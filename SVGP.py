import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import time

# CẤU HÌNH

DATA_DIR = "Experiment_Train_Trajectory_HRI"
TRAIN_LIST = os.path.join(DATA_DIR, "train_file_list.json")
VAL_LIST = os.path.join(DATA_DIR, "val_file_list.json")

WINDOW_SIZE = 10    # số bước history T
HORIZON = 1         # số bước dự đoán N
INDUCING_M = 400
BATCH_SIZE = 256
MAX_EPOCHS = 500
LR_INITIAL = 0.01

# EarlyStopping & ReduceLR-on-Plateau
PATIENCE_ES = 20
PATIENCE_RLR = 10
LR_REDUCTION_FACT = 0.5

# ─── HELPERS ────────────────────────────────────────────────────────────
def read_and_smooth(filename, sigma=1, apply_smooothing=True):
    df = pd.read_csv(filename)[["X","Y","Z"]].dropna()
    if apply_smooothing:
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
    return df.reset_index(drop=True)

def prepare_windows_from_files(file_list, T, N, data_dir):
    all_X, all_Y = [], []
    for fname in file_list:
        full_path = os.path.join(data_dir, fname)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")

        df = read_and_smooth(full_path)
        seq = df[["X", "Y", "Z"]].values  # 3D data
        L = seq.shape[0]

        # Bắt đầu vòng lặp từ chỉ số 0 đến L-N
        for i in range(L - N + 1):
            # Tạo cửa sổ đầu vào X_window
            X_window = seq[max(0, i - T):i]
            
            # Thêm padding nếu cửa sổ quá ngắn
            padding_needed = T - X_window.shape[0]
            if padding_needed > 0:
                padding = np.zeros((padding_needed, 3))  # 3D padding
                X_window = np.vstack((padding, X_window))
                
            all_X.append(X_window.flatten())
            all_Y.append(seq[i:i + N].flatten())

    return np.array(all_X), np.array(all_Y)

def scale_data(X_tr, Y_tr, X_val, Y_val):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr_s = scaler_x.fit_transform(X_tr.reshape(-1, 3)).reshape(X_tr.shape)  # 3D
    X_val_s = scaler_x.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)  # 3D

    Y_tr_s = scaler_y.fit_transform(Y_tr.reshape(-1, 3)).reshape(Y_tr.shape)  # 3D
    Y_val_s = scaler_y.transform(Y_val.reshape(-1, 3)).reshape(Y_val.shape)  # 3D

    return X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y

def setup_kernel(input_dim):
    # 4. Matern52 cho less smooth patterns
    matern = gpflow.kernels.Matern52(
        lengthscales=np.ones(input_dim) * 1.0,
        variance=1.0
    )
    return matern

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    # Đọc danh sách file từ JSON
    with open(TRAIN_LIST, 'r') as f:
        train_files = json.load(f)
    with open(VAL_LIST, 'r') as f:
        val_files = json.load(f)
    print(f"Số file train: {len(train_files)}, Số file val: {len(val_files)}")

    X_train_full, Y_train_full = prepare_windows_from_files(train_files, WINDOW_SIZE, HORIZON, DATA_DIR)
    X_val, Y_val = prepare_windows_from_files(val_files, WINDOW_SIZE, HORIZON, DATA_DIR)

    print(f"Tổng số cửa sổ huấn luyện ban đầu: {X_train_full.shape[0]}")

    # Bước 3: Chuẩn hóa dữ liệu đã được lấy mẫu
    X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y = scale_data(X_train_full, Y_train_full, X_val, Y_val)

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr_s, Y_tr_s)).shuffle(10000).batch(BATCH_SIZE).prefetch(1)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_s, Y_val_s)).batch(BATCH_SIZE)

    input_dim = X_tr_s.shape[1]
    output_dim = Y_tr_s.shape[1]

    # Khởi tạo Inducing Points Z bằng Kmeans
    kmeans = KMeans(n_clusters=INDUCING_M, random_state=42, n_init='auto', max_iter=300)
    Z = kmeans.fit(X_tr_s).cluster_centers_
    # Khởi tạo kernel
    kernel = setup_kernel(input_dim)
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1)
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=output_dim,
        whiten=True)
    gpflow.set_trainable(model.inducing_variable, True)
    optimizer = tf.optimizers.Adam(LR_INITIAL)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = -model.elbo((xb, yb))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Training loop
    best_val_loss = np.inf
    best_model_vars = None  # Lưu best model state
    wait_es = 0
    wait_rlr = 0
    current_lr = LR_INITIAL

    # --- Đo tổng thời gian huấn luyện ---
    total_train_start_time = time.time()

    # Callback
    for epoch in range(1, MAX_EPOCHS + 1):
        # --- Đo thời gian cho epoch hiện tại ---
        epoch_start_time = time.time()

        train_losses = []
        for xb, yb in train_ds:
            l = train_step(xb, yb)
            train_losses.append(l.numpy())
        train_loss = np.mean(train_losses)

        val_elbos = [model.elbo((xv, yv)).numpy() for xv, yv in val_ds]
        val_loss = -np.mean(val_elbos)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch:03d} — train_loss: {train_loss:.6f} — val_loss: {val_loss:.6f} — Time: {epoch_duration:.2f}s")

        # BETTER EARLY STOPPING với model restoration
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            # Lưu trạng thái model tốt nhất
            best_model_vars = [var.numpy() for var in model.trainable_variables]
            wait_es = 0
            wait_rlr = 0
        else:
            wait_es += 1
            wait_rlr += 1

            if wait_rlr >= PATIENCE_RLR:
                current_lr *= LR_REDUCTION_FACT
                optimizer.learning_rate.assign(current_lr)
                print(f"➜ ReduceLROnPlateau: giảm lr xuống {current_lr:.6f}")
                wait_rlr = 0

            if wait_es >= PATIENCE_ES:
                print(f"➜ EarlyStopping: dừng tại epoch {epoch}")
                # Khôi phục model tốt nhất
                if best_model_vars is not None:
                    for var, best_val in zip(model.trainable_variables, best_model_vars):
                        var.assign(best_val)
                    print("➜ Restored best model weights")
                    # Đánh giá lại sau khi restore
                    val_elbos_restored = [model.elbo((xv, yv)).numpy() for xv, yv in val_ds]
                    val_loss_restored = -np.mean(val_elbos_restored)
                    print(f"➜ Restored model validation loss: {val_loss_restored:.6f}")
                else:
                    print("➜ No best model to restore")
                break
    else:
        # Vòng lặp kết thúc mà không trigger EarlyStopping → khôi phục best model
        if best_model_vars is not None:
            for var, best_val in zip(model.trainable_variables, best_model_vars):
                var.assign(best_val)
            print(f"\n➜ Training hoàn tất {MAX_EPOCHS} epochs. Restored best model weights (best val_loss: {best_val_loss:.6f})")
        else:
            print(f"\n➜ Training hoàn tất {MAX_EPOCHS} epochs. Không có best model để restore.")

    # Tổng thời gian huấn luyện
    total_train_end_time = time.time()
    total_train_duration = total_train_end_time - total_train_start_time
    print(f"\n--- Tổng kết quá trình huấn luyện ---")
    print(f"Tổng thời gian huấn luyện (toàn bộ các epoch): {total_train_duration:.2f} giây ({total_train_duration / 60:.2f} phút)")
    print("\n--- Đánh giá trên tập Validation ---")
    means, _ = model.predict_y(X_val_s)
    pred_s = means.numpy()

    # Đảo ngược chuẩn hóa các dự đoán và nhãn gốc (3D)
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 3)).reshape(pred_s.shape)
    orig = scaler_y.inverse_transform(Y_val_s.reshape(-1, 3)).reshape(Y_val_s.shape)

    mse = np.mean((orig - pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Validation MSE:  {mse:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")


    os.makedirs("svgp_hri_m52_400", exist_ok=True)
    with open("svgp_hri_m52_400/scaler_x.pkl", "wb") as f:
        pickle.dump(scaler_x, f)
    with open("svgp_hri_m52_400/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    with open("svgp_hri_m52_400/svgp_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()