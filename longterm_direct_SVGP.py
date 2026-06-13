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
DATA_DIR = "Quy_dao_train_(10.10)"
TRAIN_LIST = os.path.join(DATA_DIR, "train_file_list.json")
VAL_LIST = os.path.join(DATA_DIR, "val_file_list.json")

WINDOW_SIZE = 10   # số bước history T (k)
HORIZON = 2    # số bước dự đoán N (h)
INDUCING_M = 100
BATCH_SIZE = 64
MAX_EPOCHS = 2000
LR_INITIAL = 0.2

# EarlyStopping & ReduceLR-on-Plateau
PATIENCE_ES = 20
PATIENCE_RLR = 10
LR_REDUCTION_FACT = 0.5

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1, apply_smooothing=True):
    # Giữ nguyên hàm này
    df = pd.read_csv(filename)[["x","y"]].dropna()
    if apply_smooothing:
        df["x"] = gaussian_filter1d(df["x"].values, sigma=sigma)
        df["y"] = gaussian_filter1d(df["y"].values, sigma=sigma)
    return df.reset_index(drop=True)

def prepare_windows_from_files(file_list, T, N, data_dir):
    all_X, all_Y = [], []
    for fname in file_list:
        full_path = os.path.join(data_dir, fname)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Không tìm thấy file: {full_path}")

        df = read_and_smooth(full_path)
        seq = df[["x", "y"]].values
        L = seq.shape[0]
        for i in range(T, L - N + 1):
            X_window = seq[i - T : i]
            Y_window = seq[i : i + N]
                
            all_X.append(X_window.flatten())
            all_Y.append(Y_window.flatten())

    return np.array(all_X), np.array(all_Y)

def scale_data(X_tr, Y_tr, X_val, Y_val):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr_s = scaler_x.fit_transform(X_tr)
    X_val_s = scaler_x.transform(X_val)

    Y_tr_s = scaler_y.fit_transform(Y_tr.reshape(-1, 2)).reshape(Y_tr.shape)
    Y_val_s = scaler_y.transform(Y_val.reshape(-1, 2)).reshape(Y_val.shape)

    return X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y

def setup_kernel(input_dim):
    # Giữ nguyên hàm này
    rbf = gpflow.kernels.SquaredExponential(lengthscales=np.ones(input_dim) * 3.0, variance=1.0)
    white = gpflow.kernels.White(variance=0.05)
    rbf.lengthscales.prior = tfp.distributions.Gamma(2.0, 1.0)
    rbf.variance.prior = tfp.distributions.Gamma(2.0, 1.0)
    white.variance.prior = tfp.distributions.Gamma(2.0, 10.0)
    return rbf + white

# ─── DIRECT APPROACH MAIN ───────────────────────────────────────────────
def main():
    # 1. Tải và chuẩn bị dữ liệu thô (giống bản gốc)
    train_files = load_file_list(TRAIN_LIST)
    val_files = load_file_list(VAL_LIST)

    # X_train_full: (Số lượng mẫu, T*2) | Y_train_full: (Số lượng mẫu, N*2)
    X_train_full, Y_train_full = prepare_windows_from_files(train_files, WINDOW_SIZE, HORIZON, DATA_DIR)
    X_val, Y_val = prepare_windows_from_files(val_files, WINDOW_SIZE, HORIZON, DATA_DIR)

    print(f"Tổng số cửa sổ huấn luyện: {X_train_full.shape[0]}")
    print(f"Tổng số cửa sổ validation: {X_val.shape[0]}")

    # 2. Chuẩn hóa dữ liệu
    # Lưu ý: Chuẩn hóa Y bằng cách reshape thành (N*Mẫu, 2) để scaler_y chuẩn hóa từng cặp (x, y)
    X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y = scale_data(X_train_full, Y_train_full, X_val, Y_val)

    # 3. Khởi tạo Inducing Points Z bằng Kmeans (dùng chung cho tất cả các model)
    input_dim = X_tr_s.shape[1]
    kmeans = KMeans(n_clusters=INDUCING_M, random_state=42, n_init='auto', max_iter=300)
    Z = kmeans.fit(X_tr_s).cluster_centers_

    print(f"Kích thước đầu vào (Input dim): {input_dim}")
    print(f"Số lượng mô hình (Horizon N): {HORIZON}")
    
    # Tạo thư mục lưu trữ
    os.makedirs("lt_svgp_direct_8s", exist_ok=True)
    with open("lt_svgp_direct_8s/scaler_x.pkl", "wb") as f:
        pickle.dump(scaler_x, f)
    with open("lt_svgp_direct_8s/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # 4. Vòng lặp huấn luyện cho từng bước (t+1 đến t+HORIZON)
    total_train_start_time = time.time()
    all_pred_s = []  # Lưu tất cả các dự đoán (đã chuẩn hóa)
    val_losses_per_step = [] # Lưu loss của từng bước

    for k in range(1, HORIZON + 1):
        print(f"\n--- Huấn luyện Mô hình {k}/{HORIZON} (Dự đoán bước t+{k}) ---")

        # a. Tách nhãn đầu ra Y(k) (tọa độ x, y tại bước t+k)
        # Y_tr_s có shape (Mẫu, N*2). Ta cần cột 2*(k-1) và 2*k.
        Y_k_tr_s = Y_tr_s[:, 2*(k-1):2*k]  # Shape (Mẫu, 2)
        Y_k_val_s = Y_val_s[:, 2*(k-1):2*k] # Shape (Mẫu, 2)

        # b. Tạo Dataset
        train_ds_k = tf.data.Dataset.from_tensor_slices((X_tr_s, Y_k_tr_s)).shuffle(10000).batch(BATCH_SIZE).prefetch(1)
        val_ds_k = tf.data.Dataset.from_tensor_slices((X_val_s, Y_k_val_s)).batch(BATCH_SIZE)

        # c. Khởi tạo Model M(k)
        kernel = setup_kernel(input_dim)
        likelihood = gpflow.likelihoods.Gaussian(variance=0.1)
        model = gpflow.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=Z.copy(), # Dùng bản sao của Z
            num_latent_gps=2, # Chỉ dự đoán (x, y) -> 2 GP độc lập
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

        # d. Training loop cho M(k)
        best_val_loss = np.inf
        best_model_vars = None
        wait_es = 0
        wait_rlr = 0
        current_lr = LR_INITIAL
        
        # --- Đo thời gian cho model hiện tại ---
        model_start_time = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            train_losses = []
            epoch_start_time = time.time()
            for xb, yb in train_ds_k:
                l = train_step(xb, yb)
                train_losses.append(l.numpy())
            train_loss = np.mean(train_losses)

            val_elbos = [model.elbo((xv, yv)).numpy() for xv, yv in val_ds_k]
            val_loss = -np.mean(val_elbos)
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:03d} — train_loss: {train_loss:.6f} — val_loss: {val_loss:.6f} - time: {epoch_duration:.3f}")

            # Early Stopping / Reduce LR
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_model_vars = [var.numpy() for var in model.trainable_variables]
                wait_es = 0
                wait_rlr = 0
            else:
                wait_es += 1
                wait_rlr += 1

                if wait_rlr >= PATIENCE_RLR:
                    current_lr *= LR_REDUCTION_FACT
                    optimizer.learning_rate.assign(current_lr)
                    print(f"  ➜ ReduceLROnPlateau: giảm lr xuống {current_lr:.6f}")
                    wait_rlr = 0

                if wait_es >= PATIENCE_ES:
                    # Khôi phục model tốt nhất
                    if best_model_vars is not None:
                        for var, best_val in zip(model.trainable_variables, best_model_vars):
                            var.assign(best_val)
                    print(f"  ➜ EarlyStopping: dừng tại epoch {epoch}. Khôi phục model tốt nhất.")
                    break
        
        model_end_time = time.time()
        print(f"Model {k} huấn luyện trong: {model_end_time - model_start_time:.2f}s")

        # e. Đánh giá và lưu Model M(k)
        means, _ = model.predict_y(X_val_s)
        pred_k_s = means.numpy()
        all_pred_s.append(pred_k_s)
        
        # Tính toán loss cuối cùng
        val_elbos_final = [model.elbo((xv, yv)).numpy() for xv, yv in val_ds_k]
        val_loss_final = -np.mean(val_elbos_final)
        val_losses_per_step.append(val_loss_final)

        # Lưu model
        with open(f"lt_svgp_direct_8s/svgp_model_step_{k:02d}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"  Đã lưu svgp_model_step_{k:02d}.pkl với val_loss: {val_loss_final:.6f}")

    # 5. Tổng kết quá trình huấn luyện và đánh giá toàn bộ Horizon
    total_train_end_time = time.time()
    total_train_duration = total_train_end_time - total_train_start_time

    print(f"\n\n--- Tổng kết Quá trình Huấn luyện (Direct Approach) ---")
    print(f"Tổng thời gian huấn luyện (cho {HORIZON} model): {total_train_duration:.2f} giây ({total_train_duration / 60:.2f} phút)")

    print("\n--- Đánh giá Tổng hợp trên tập Validation ---")
    # Ghép tất cả các dự đoán đã chuẩn hóa lại thành (Mẫu, N*2)
    pred_s = np.concatenate(all_pred_s, axis=1) # Shape (Mẫu, N*2)

    # Đảo ngược chuẩn hóa các dự đoán và nhãn gốc
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 2)).reshape(pred_s.shape)
    orig = scaler_y.inverse_transform(Y_val_s.reshape(-1, 2)).reshape(Y_val_s.shape)

    mse = np.mean((orig - pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"Validation Loss (ELBO) trung bình trên {HORIZON} bước: {np.mean(val_losses_per_step):.6f}")
    print(f"Validation **Tổng hợp** MSE:  {mse:.6f}")
    print(f"Validation **Tổng hợp** RMSE: {rmse:.6f}")
    print("Đã lưu scaler_x, scaler_y và tất cả các svgp_model_* vào thư mục lt_svgp_direct_8s")


if __name__ == "__main__":
    # Đặt môi trường để tắt eager execution, tăng tốc (tùy chọn)
    tf.config.experimental_run_functions_eagerly(False)
    main()