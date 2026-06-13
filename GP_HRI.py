import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import MiniBatchKMeans
import time
import warnings

# ══════════════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════════════
DATA_DIR        = "Experiment_Train_Trajectory_HRI"
SCENARIO_IDS    = [1, 2, 3]          # Chỉ huấn luyện kịch bản 1, 2, 3
N_TRAJECTORIES  = 30                 # Số quỹ đạo chọn mẫu
TRAIN_RATIO     = 0.8                # 80% train / 20% validation
RANDOM_SEED     = 42

WINDOW_SIZE     = 10                 # T - số bước lịch sử
HORIZON         = 1                  # N - số bước dự đoán
MAX_EPOCHS      = 200
LR_INITIAL      = 0.1

# EarlyStopping & ReduceLR-on-Plateau
PATIENCE_ES       = 20
PATIENCE_RLR      = 10
LR_REDUCTION_FACT = 0.5

# Danh sách K-Means Ratio cần thử
KMEANS_RATIOS = [1.0, 0.1]

# Tên thư mục output tương ứng
OUTPUT_DIRS = {
    1.0: "gp_hri_kmeans_ratio_1p0",
    0.1: "gp_hri_kmeans_ratio_0p1",
}

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def get_scenario_id(filepath):
    """Đọc ScenarioId từ dòng đầu tiên dữ liệu của file CSV."""
    try:
        df = pd.read_csv(filepath, nrows=1)
        if "ScenarioId" in df.columns:
            return int(df["ScenarioId"].iloc[0])
    except Exception:
        pass
    return None


def filter_files_by_scenario(data_dir, scenario_ids):
    """Lọc danh sách file CSV có ScenarioId nằm trong scenario_ids."""
    all_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ])
    matched = []
    print(f"Đang quét {len(all_files)} file CSV để lọc ScenarioId {scenario_ids}...")
    for fname in all_files:
        sid = get_scenario_id(os.path.join(data_dir, fname))
        if sid in scenario_ids:
            matched.append(fname)
    print(f"  → Tìm thấy {len(matched)} file thoả điều kiện ScenarioId ∈ {scenario_ids}.")
    return matched


def sample_and_split(file_list, n_traj, train_ratio, seed):
    """
    Lấy mẫu ngẫu nhiên n_traj quỹ đạo từ file_list,
    sau đó chia thành train / validation theo train_ratio.
    """
    rng = np.random.default_rng(seed)
    if len(file_list) < n_traj:
        warnings.warn(
            f"Chỉ có {len(file_list)} file thoả mãn, nhỏ hơn N_TRAJECTORIES={n_traj}. "
            "Sẽ dùng toàn bộ file."
        )
        selected = list(file_list)
    else:
        idx = rng.choice(len(file_list), size=n_traj, replace=False)
        selected = [file_list[i] for i in sorted(idx)]

    n_train = int(len(selected) * train_ratio)
    train_files = selected[:n_train]
    val_files   = selected[n_train:]
    print(f"  Tổng quỹ đạo được chọn : {len(selected)}")
    print(f"  Train                  : {len(train_files)} quỹ đạo")
    print(f"  Validation             : {len(val_files)} quỹ đạo")
    return train_files, val_files


def read_and_smooth(filepath, sigma=1, apply_smoothing=True):
    """Đọc file CSV và làm mượt (Gaussian) cột X, Y, Z."""
    df = pd.read_csv(filepath)[["X", "Y", "Z"]].dropna()
    if apply_smoothing:
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
    return df.reset_index(drop=True)


def prepare_windows_from_files(file_list, T, N, data_dir):
    """Tạo cửa sổ sliding window (X_input, Y_target) từ danh sách file."""
    all_X, all_Y = [], []
    for fname in file_list:
        full_path = os.path.join(data_dir, fname)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")

        df  = read_and_smooth(full_path)
        seq = df[["X", "Y", "Z"]].values   # shape (L, 3)
        L   = seq.shape[0]

        for i in range(L - N + 1):
            X_win = seq[max(0, i - T):i]
            pad   = T - X_win.shape[0]
            if pad > 0:
                X_win = np.vstack((np.zeros((pad, 3)), X_win))
            all_X.append(X_win.flatten())           # (3*T,)
            all_Y.append(seq[i:i + N].flatten())    # (3*N,)

    return np.array(all_X, dtype=np.float64), np.array(all_Y, dtype=np.float64)


def scale_data(X_tr, Y_tr, X_val, Y_val):
    """Chuẩn hóa dữ liệu theo từng chiều (X, Y, Z)."""
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr_s  = scaler_x.fit_transform(X_tr.reshape(-1, 3)).reshape(X_tr.shape)
    X_val_s = scaler_x.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)

    Y_tr_s  = scaler_y.fit_transform(Y_tr.reshape(-1, 3)).reshape(Y_tr.shape)
    Y_val_s = scaler_y.transform(Y_val.reshape(-1, 3)).reshape(Y_val.shape)

    return X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y


def setup_kernel(input_dim):
    """Khởi tạo kernel Squared Exponential (RBF)."""
    return gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(input_dim) * 1.0,
        variance=1.0
    )

def train_gp(X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_y,
             kmeans_ratio, output_dir):
    """
    Huấn luyện mô hình GPR với K-Means sampling theo kmeans_ratio,
    sau đó lưu kết quả vào output_dir.
    """
    print(f"\n{'═'*60}")
    print(f"  KMEANS_RATIO = {kmeans_ratio:.2f}  →  lưu vào '{output_dir}'")
    print(f"{'═'*60}")

    # ── Lấy mẫu K-Means ──────────────────────────────
    n_total = X_tr_s.shape[0]
    n_keep  = max(1, int(n_total * kmeans_ratio))

    if kmeans_ratio >= 1.0 or n_keep >= n_total:
        # Sử dụng toàn bộ dữ liệu
        X_sampled = X_tr_s
        Y_sampled = Y_tr_s
        print(f"K-Means ratio = 1.0 → dùng toàn bộ {n_total} cửa sổ.")
    else:
        print(f"Lấy mẫu K-Means: {n_total} → {n_keep} cụm...")
        km = MiniBatchKMeans(n_clusters=n_keep, random_state=0,
                             n_init='auto', batch_size=256)
        km.fit(X_tr_s)
        X_sampled = km.cluster_centers_
        closest_idx, _ = pairwise_distances_argmin_min(X_sampled, X_tr_s)
        Y_sampled = Y_tr_s[closest_idx]
        print(f"  → {X_sampled.shape[0]} cửa sổ sau lấy mẫu.")

    # ── Chuyển sang TF ───────────────────────────────
    X_tr_tf  = tf.constant(X_sampled,  dtype=gpflow.default_float())
    Y_tr_tf  = tf.constant(Y_sampled,  dtype=gpflow.default_float())
    X_val_tf = tf.constant(X_val_s,    dtype=gpflow.default_float())

    # ── Xây dựng mô hình ─────────────────────────────
    input_dim = X_sampled.shape[1]
    kernel    = setup_kernel(input_dim)
    model     = gpflow.models.GPR((X_tr_tf, Y_tr_tf), kernel=kernel)
    optimizer = tf.optimizers.Adam(LR_INITIAL)

    if X_sampled.shape[0] > 2000:
        print("CẢNH BÁO: Số điểm lớn – Full GP có thể chậm (O(N³)).")

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.negative(model.log_marginal_likelihood())
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # ── Vòng lặp huấn luyện ──────────────────────────
    best_val_mse   = np.inf
    best_model_vars = None
    wait_es = wait_rlr = 0
    current_lr     = LR_INITIAL
    history_rows   = []

    total_t0 = time.time()
    print(f"\n--- Bắt đầu huấn luyện GPR —  {X_sampled.shape[0]} điểm ---")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_step()

        val_means, _ = model.predict_y(X_val_tf)
        val_pred      = scaler_y.inverse_transform(val_means.numpy().reshape(-1, 3))
        val_orig      = scaler_y.inverse_transform(Y_val_s.reshape(-1, 3))
        val_mse       = float(np.mean((val_orig - val_pred) ** 2))
        val_rmse      = float(np.sqrt(val_mse))
        ep_time       = time.time() - t0

        print(f"Epoch {epoch:03d} — NLL: {train_loss.numpy():.6f} "
              f"— val_MSE: {val_mse:.6f} — Time: {ep_time:.2f}s")

        history_rows.append({
            "epoch":      epoch,
            "train_loss": float(train_loss.numpy()),
            "val_mse":    val_mse,
            "val_rmse":   val_rmse,
        })

        # Early Stopping
        if val_mse < best_val_mse - 1e-6:
            best_val_mse    = val_mse
            best_model_vars = [v.numpy() for v in model.trainable_variables]
            wait_es = wait_rlr = 0
        else:
            wait_es  += 1
            wait_rlr += 1

            if wait_rlr >= PATIENCE_RLR:
                current_lr *= LR_REDUCTION_FACT
                optimizer.learning_rate.assign(current_lr)
                print(f"  ➜ ReduceLR → lr = {current_lr:.6f}")
                wait_rlr = 0

            if wait_es >= PATIENCE_ES:
                print(f"  ➜ EarlyStopping tại epoch {epoch}")
                if best_model_vars is not None:
                    for v, bv in zip(model.trainable_variables, best_model_vars):
                        v.assign(bv)
                    print("  ➜ Restored best model weights")
                break
    else:
        if best_model_vars is not None:
            for v, bv in zip(model.trainable_variables, best_model_vars):
                v.assign(bv)
            print(f"\n  ➜ Hoàn tất {MAX_EPOCHS} epochs. Restored best (val_MSE={best_val_mse:.6f})")

    total_dur = time.time() - total_t0
    print(f"\n  Tổng thời gian huấn luyện: {total_dur:.2f}s ({total_dur/60:.2f} phút)")

    # ── Đánh giá cuối ────────────────────────────────
    val_means_f, _ = model.predict_y(X_val_tf)
    pred_f = scaler_y.inverse_transform(val_means_f.numpy().reshape(-1, 3))
    orig_f = scaler_y.inverse_transform(Y_val_s.reshape(-1, 3))
    mse_f  = float(np.mean((orig_f - pred_f) ** 2))
    rmse_f = float(np.sqrt(mse_f))
    print(f"\n--- Kết quả cuối cùng trên Validation ---")
    print(f"  MSE  : {mse_f:.6f}")
    print(f"  RMSE : {rmse_f:.6f}")

    # ── Lưu kết quả ──────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Lưu training log
    try:
        pd.DataFrame(history_rows).to_csv(
            os.path.join(output_dir, "training_log.csv"), index=False)
        print(f"  Đã lưu training_log.csv → {output_dir}/")
    except Exception as e:
        print(f"  Cảnh báo: Không thể lưu training_log.csv — {e}")

    # Lưu scaler và model
    with open(os.path.join(output_dir, "scaler_x.pkl"), "wb") as f:
        pickle.dump(scaler_x_global, f)
    with open(os.path.join(output_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(output_dir, "gp_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Lưu metadata
    meta = {
        "kmeans_ratio":     kmeans_ratio,
        "n_train_windows":  int(X_sampled.shape[0]),
        "n_val_windows":    int(X_val_s.shape[0]),
        "window_size":      WINDOW_SIZE,
        "horizon":          HORIZON,
        "best_val_mse":     best_val_mse,
        "final_val_mse":    mse_f,
        "final_val_rmse":   rmse_f,
        "train_duration_s": round(total_dur, 2),
        "scenario_ids":     SCENARIO_IDS,
        "n_trajectories":   N_TRAJECTORIES,
        "train_ratio":      TRAIN_RATIO,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Đã lưu scaler_x.pkl, scaler_y.pkl, gp_model.pkl, metadata.json → {output_dir}/\n")
    return mse_f, rmse_f


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

# Biến global để train_gp truy cập scaler_x
scaler_x_global = None

def main():
    global scaler_x_global

    print("=" * 60)
    print("  GP HRI Training Script")
    print(f"  ScenarioId filter : {SCENARIO_IDS}")
    print(f"  N_TRAJECTORIES    : {N_TRAJECTORIES}")
    print(f"  Train / Val ratio : {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    print("=" * 60)

    # ── 1. Lọc file theo ScenarioId ──────────────────
    matched_files = filter_files_by_scenario(DATA_DIR, SCENARIO_IDS)
    if not matched_files:
        raise RuntimeError("Không tìm thấy file nào thoả điều kiện ScenarioId!")

    # ── 2. Lấy mẫu & chia train/val ──────────────────
    print(f"\n[Bước 2] Lấy mẫu {N_TRAJECTORIES} quỹ đạo và chia train/val...")
    train_files, val_files = sample_and_split(
        matched_files, N_TRAJECTORIES, TRAIN_RATIO, RANDOM_SEED
    )
    print(f"\n  Train files: {train_files}")
    print(f"  Val   files: {val_files}")

    # Lưu lại danh sách file để tham khảo
    os.makedirs("gp_hri_data_split", exist_ok=True)
    with open("gp_hri_data_split/train_file_list.json", "w") as f:
        json.dump(train_files, f, indent=2)
    with open("gp_hri_data_split/val_file_list.json", "w") as f:
        json.dump(val_files, f, indent=2)
    print("\n  Đã lưu danh sách file vào gp_hri_data_split/")

    # ── 3. Tạo cửa sổ sliding window ─────────────────
    print(f"\n[Bước 3] Tạo cửa sổ sliding window (T={WINDOW_SIZE}, N={HORIZON})...")
    X_train_full, Y_train_full = prepare_windows_from_files(
        train_files, WINDOW_SIZE, HORIZON, DATA_DIR
    )
    X_val, Y_val = prepare_windows_from_files(
        val_files, WINDOW_SIZE, HORIZON, DATA_DIR
    )
    print(f"  Cửa sổ train : {X_train_full.shape[0]}")
    print(f"  Cửa sổ val   : {X_val.shape[0]}")

    # ── 4. Chuẩn hóa dữ liệu ─────────────────────────
    print("\n[Bước 4] Chuẩn hóa dữ liệu...")
    X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y = scale_data(
        X_train_full, Y_train_full, X_val, Y_val
    )
    scaler_x_global = scaler_x  # expose to train_gp

    # ── 5. Huấn luyện với từng K-Means Ratio ─────────
    results = {}
    for ratio in KMEANS_RATIOS:
        out_dir = OUTPUT_DIRS[ratio]
        mse, rmse = train_gp(
            X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_y,
            kmeans_ratio=ratio,
            output_dir=out_dir
        )
        results[ratio] = {"val_mse": mse, "val_rmse": rmse}

    # ── 6. Tổng kết ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  TỔNG KẾT KẾT QUẢ")
    print("=" * 60)
    for ratio, res in results.items():
        print(f"  KMeans Ratio {ratio:.2f}  | "
              f"val_MSE={res['val_mse']:.6f}  | "
              f"val_RMSE={res['val_rmse']:.6f}  | "
              f"Folder: {OUTPUT_DIRS[ratio]}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
