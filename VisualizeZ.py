import os
import pickle
import json
import numpy as np
import pandas as pd
import gpflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

# CẤU HÌNH
CHECKPOINT_DIR = "checkpoints_gp_3target"
WINDOW_SIZE = 10

# Thiết lập font Times New Roman cho đồ thị
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 11.5
plt.rcParams['legend.fontsize'] = 10

def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1, apply_smoothing=True):
    df = pd.read_csv(filename)[["x", "y"]].dropna()
    if apply_smoothing:
        df["x"] = gaussian_filter1d(df["x"].values, sigma=sigma)
        df["y"] = gaussian_filter1d(df["y"].values, sigma=sigma)
    return df.values

def prepare_windows_from_files(file_list, T, data_dir):
    all_X = []
    for fname in file_list:
        full_path = os.path.join(data_dir, fname)
        if not os.path.isfile(full_path):
            print(f"Không tìm thấy file: {full_path}. Bỏ qua.")
            continue
        
        seq = read_and_smooth(full_path)
        L = seq.shape[0]

        for i in range(L):
            X_window = seq[max(0, i - T):i]
            padding_needed = T - X_window.shape[0]
            if padding_needed > 0:
                padding = np.zeros((padding_needed, 2))
                X_window = np.vstack((padding, X_window))
            all_X.append(X_window.flatten())
    return np.array(all_X)

def load_scalers_and_model(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "full_gp_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler_x, model

def main():
    # Bước 1: Tải model và scaler
    scaler_x, model = load_scalers_and_model(CHECKPOINT_DIR)

    # Bước 2: Trích xuất và giải chuẩn hóa các điểm hỗ trợ Z
    inducing_points_scaled = model.inducing_variable.Z.numpy()
    
    # Lấy số lượng điểm Z
    num_z_points = inducing_points_scaled.shape[0]
    
    # Reshape và giải chuẩn hóa
    inducing_points_unscaled = scaler_x.inverse_transform(inducing_points_scaled.reshape(-1, 2))
    
    # Bước 3: Vẽ đồ thị
    plt.figure(figsize=(4.5, 4.5))
    # Vẽ các điểm hỗ trợ Z với chú thích bổ sung
    label_z = f'Các điểm (Z) ({num_z_points} điểm)'
    plt.scatter(inducing_points_unscaled[:, 0], inducing_points_unscaled[:, 1], s=50, c='blue', marker='.', label=label_z)
    
    plt.title('Vị trí của các điểm hỗ trợ (Z)')
    plt.xlabel('Tọa độ x')
    plt.ylabel('Tọa độ y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()