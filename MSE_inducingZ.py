import os
import json
import pickle
import numpy as np
import pandas as pd
import csv # <--- Thêm thư viện CSV
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import time
import matplotlib.pyplot as plt # <--- Thêm thư viện Matplotlib

# CẤU HÌNH

DATA_DIR = "Trajectories_GP_TRAIN"
TRAIN_LIST = os.path.join(DATA_DIR, "train_file_list.json")
VAL_LIST = os.path.join(DATA_DIR, "val_file_list.json")

WINDOW_SIZE = 10    # số bước history T
HORIZON = 1         # số bước dự đoán N
BATCH_SIZE = 64
MAX_EPOCHS = 500
LR_INITIAL = 0.01

# EarlyStopping & ReduceLR-on-Plateau
PATIENCE_ES = 20
PATIENCE_RLR = 10
LR_REDUCTION_FACT = 0.5

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1, apply_smooothing=True):
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

        # Bắt đầu vòng lặp từ chỉ số 0 đến L-N
        for i in range(L - N + 1):
            # Tạo cửa sổ đầu vào X_window
            X_window = seq[max(0, i - T):i]
            
            # Thêm padding nếu cửa sổ quá ngắn
            padding_needed = T - X_window.shape[0]
            if padding_needed > 0:
                padding = np.zeros((padding_needed, 2))
                X_window = np.vstack((padding, X_window))
                
            all_X.append(X_window.flatten())
            all_Y.append(seq[i:i + N].flatten())

    return np.array(all_X), np.array(all_Y)

def scale_data(X_tr, Y_tr, X_val, Y_val):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr_s = scaler_x.fit_transform(X_tr.reshape(-1, 2)).reshape(X_tr.shape)
    X_val_s = scaler_x.transform(X_val.reshape(-1, 2)).reshape(X_val.shape)

    Y_tr_s = scaler_y.fit_transform(Y_tr.reshape(-1, 2)).reshape(Y_tr.shape)
    Y_val_s = scaler_y.transform(Y_val.reshape(-1, 2)).reshape(Y_val.shape)

    return X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y

def setup_kernel(input_dim):
    # RBF với lengthscales lớn hơn để smooth hơn
    rbf = gpflow.kernels.SquaredExponential(lengthscales=np.ones(input_dim) * 3.0, variance=1.0)
    # White noise lớn hơn để regularize
    white = gpflow.kernels.White(variance=0.05)
    # Set priors để regularization
    rbf.lengthscales.prior = tfp.distributions.Gamma(2.0, 1.0)
    rbf.variance.prior = tfp.distributions.Gamma(2.0, 1.0)
    white.variance.prior = tfp.distributions.Gamma(2.0, 10.0)
    return rbf + white

# ─── HÀM HUẤN LUYỆN MỚI ────────────────────────────────────────────────
def train_svgp(inducing_m, X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_y, 
               batch_size, max_epochs, lr_initial, patience_es, 
               patience_rlr, lr_reduction_fact):
    """
    Huấn luyện mô hình SVGP với số lượng Inducing Points (M) được chỉ định.
    Trả về danh sách RMSE trên tập validation theo từng epoch.
    """
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN VỚI INDUCING_M = {inducing_m} ---")
    
    # 1. Chuẩn bị Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr_s, Y_tr_s)).shuffle(10000).batch(batch_size).prefetch(1)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_s, Y_val_s)).batch(batch_size)

    input_dim = X_tr_s.shape[1]
    output_dim = Y_tr_s.shape[1]

    # 2. Khởi tạo Mô hình SVGP
    kmeans = KMeans(n_clusters=inducing_m, random_state=42, n_init='auto', max_iter=300)
    sample_size = min(inducing_m, X_tr_s.shape[0])
    
    # Chỉ chạy KMeans nếu số lượng mẫu đủ lớn hơn M
    if X_tr_s.shape[0] > inducing_m:
        Z = kmeans.fit(X_tr_s).cluster_centers_
    else:
        sample_indices = np.random.choice(X_tr_s.shape[0], sample_size, replace=False)
        Z = X_tr_s[sample_indices]

    kernel = setup_kernel(input_dim)
    likelihood = gpflow.likelihoods.Gaussian(variance=0.1)
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=output_dim,
        whiten=True)
    gpflow.set_trainable(model.inducing_variable, True)
    optimizer = tf.optimizers.Adam(lr_initial)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            loss = -model.elbo((xb, yb))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    def calculate_mse(model, X_val_s, Y_val_s, scaler_y):
        """Tính RMSE trên toàn bộ tập Validation (đảo ngược chuẩn hóa)"""
        means, _ = model.predict_y(X_val_s)
        pred_s = means.numpy()

        # Đảo ngược chuẩn hóa các dự đoán và nhãn gốc
        pred = scaler_y.inverse_transform(pred_s.reshape(-1, 2)).reshape(pred_s.shape)
        orig = scaler_y.inverse_transform(Y_val_s.reshape(-1, 2)).reshape(Y_val_s.shape)

        mse = np.mean((orig - pred) ** 2)
        return mse

    # 3. Vòng lặp Huấn luyện
    best_val_loss = np.inf
    best_model_vars = None
    wait_es = 0
    wait_rlr = 0
    current_lr = lr_initial
    
    # Danh sách lưu kết quả MSE sau mỗi epoch
    mse_history = [] 

    for epoch in range(1, max_epochs + 1):
        # Huấn luyện
        train_losses = [train_step(xb, yb).numpy() for xb, yb in train_ds]
        train_loss = np.mean(train_losses)

        # Đánh giá Validation Loss theo ELBO
        val_elbos = [model.elbo((xv, yv)).numpy() for xv, yv in val_ds]
        val_loss = -np.mean(val_elbos)

        # Theo dõi MSE trên Validation (sau khi đảo chuẩn hóa) để log/CSV
        current_mse = calculate_mse(model, X_val_s, Y_val_s, scaler_y)
        mse_history.append(current_mse)

        print(f"M={inducing_m} — Epoch {epoch:03d} — train_loss: {train_loss:.6f} — val_loss: {val_loss:.6f} — val_MSE: {current_mse:.6f}")

        # BETTER EARLY STOPPING
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            # Lưu trạng thái model tốt nhất
            best_model_vars = [var.numpy() for var in model.trainable_variables]
            wait_es = 0
            wait_rlr = 0
        else:
            wait_es += 1
            wait_rlr += 1

            if wait_rlr >= patience_rlr:
                current_lr *= lr_reduction_fact
                optimizer.learning_rate.assign(current_lr)
                print(f"➜ M={inducing_m}: ReduceLROnPlateau: giảm lr xuống {current_lr:.6f}")
                wait_rlr = 0

            if wait_es >= patience_es:
                print(f"➜ M={inducing_m}: EarlyStopping: dừng tại epoch {epoch}")
                # KHÔNG khôi phục model, vì ta cần lưu lại lịch sử RMSE đã đạt được.    
                # Tuy nhiên, ta CÓ THỂ khôi phục để đảm bảo RMSE cuối cùng được báo cáo là tốt nhất.
                if best_model_vars is not None:
                    for var, best_val in zip(model.trainable_variables, best_model_vars):
                        var.assign(best_val)
                    print(f"➜ M={inducing_m}: Restored best model weights (cho báo cáo MSE cuối cùng)")
                break

    # Báo cáo MSE tốt nhất quan sát được trong quá trình huấn luyện
    final_mse = float(np.min(mse_history)) if len(mse_history) > 0 else np.nan
    print(f"--- M={inducing_m} Kết quả MSE tốt nhất: {final_mse:.6f} ---")
    
    return mse_history

# ─── HÀM CHÍNH THỰC HIỆN THỬ NGHIỆM VÀ XUẤT FILE ──────────────────────
def main():
    # Danh sách các Inducing Points (M) bạn muốn thử nghiệm
    inducing_points_list = [50, 100, 200, 500] 
    
    # 1. Chuẩn bị Dữ liệu chung (chỉ làm một lần)
    train_files = load_file_list(TRAIN_LIST)
    val_files = load_file_list(VAL_LIST)

    X_train_full, Y_train_full = prepare_windows_from_files(train_files, WINDOW_SIZE, HORIZON, DATA_DIR)
    X_val, Y_val = prepare_windows_from_files(val_files, WINDOW_SIZE, HORIZON, DATA_DIR)

    # Chuẩn hóa Dữ liệu (chỉ làm một lần)
    X_tr_s, Y_tr_s, X_val_s, Y_val_s, scaler_x, scaler_y = scale_data(X_train_full, Y_train_full, X_val, Y_val)

    # Dictionary lưu trữ lịch sử MSE: {M: [mse_epoch_1, mse_epoch_2, ...]}
    results_history = {}
    
    # 2. Chạy Thử nghiệm
    for M in inducing_points_list:
        mse_history = train_svgp(
            inducing_m=M, 
            X_tr_s=X_tr_s, Y_tr_s=Y_tr_s, 
            X_val_s=X_val_s, Y_val_s=Y_val_s, 
            scaler_y=scaler_y, 
            batch_size=BATCH_SIZE, 
            max_epochs=MAX_EPOCHS, 
            lr_initial=LR_INITIAL, 
            patience_es=PATIENCE_ES, 
            patience_rlr=PATIENCE_RLR, 
            lr_reduction_fact=LR_REDUCTION_FACT
        )
        # Lưu kết quả
        results_history[M] = mse_history
        
    # 3. Xuất ra File CSV
    output_filename = "svgp_mse_by_iteration.csv"
    max_len = max(len(h) for h in results_history.values())
    
    # Tạo tiêu đề
    header = ['Epoch'] + [f'MSE_M{M}' for M in inducing_points_list]
    
    # Chuẩn bị dữ liệu để ghi
    data_rows = []
    for i in range(max_len):
        row = [i + 1] # Cột Epoch
        for M in inducing_points_list:
            # Lấy MSE hoặc None nếu epoch đó chưa được chạy (do ES)
            mse_val = results_history[M][i] if i < len(results_history[M]) else '' 
            row.append(mse_val)
        data_rows.append(row)

    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
        
    print(f"\n✅ Đã xuất lịch sử MSE theo từng epoch vào file: {output_filename}")
    
    plt.title('MSE trên Validation theo Epoch cho các giá trị M khác nhau')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Validation)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()