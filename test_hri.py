import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Thiết lập font Times New Roman cho đồ thị
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12 #title
plt.rcParams['axes.labelsize'] = 12 #2 trục x và y
plt.rcParams['xtick.labelsize'] = 12 #các số trên trục x
plt.rcParams['ytick.labelsize'] = 12 #các số trên trục y
plt.rcParams['legend.fontsize'] = 10 #chú thích

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_DIR = "Experiment_Test_Trajectory_HRI"
TEST_LIST = os.path.join(DATA_DIR, "test_file_list.json")
CHECKPOINT = "svgp_hri_m52_400"
WINDOW_SIZE = 10
HORIZON = 1  # Trong bài toán này, HORIZON luôn là 1 để dự đoán điểm tiếp theo

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1):
    """Đọc file CSV với các cột X, Y, Z"""
    df = pd.read_csv(filename)[["X", "Y", "Z"]].dropna()
    return df.reset_index(drop=True)

# Hàm tạo cửa sổ với padding cho dữ liệu 3D
# start_point: điểm đầu tiên của quỹ đạo, dùng làm giá trị padding
def create_sequences_with_padding(data_sequence, window_size, horizon, start_point):
    X_sequences = []
    Y_targets = []
    L = data_sequence.shape[0]  # Số điểm của quỹ đạo (L)
    num_features = data_sequence.shape[1]  # Số chiều của mỗi điểm (3: X, Y, Z)
    
    # Bắt đầu từ i=1 (bỏ qua điểm đầu tiên vì đã biết)
    for i in range(1, L):
        # Tạo cửa sổ với padding bằng start_point
        current_window = np.tile(start_point, (window_size, 1)).astype(data_sequence.dtype)
        actual_points_end_idx = i
        actual_points_start_idx = max(0, i - window_size)

        num_actual_points = actual_points_end_idx - actual_points_start_idx

        if num_actual_points > 0:
            current_window[window_size - num_actual_points : window_size] = data_sequence[actual_points_start_idx : actual_points_end_idx]
        
        X_sequences.append(current_window.flatten())
        Y_targets.append(data_sequence[i : i + 1].flatten())

    return np.array(X_sequences), np.array(Y_targets)

def load_scalers_and_model(checkpoint_dir=CHECKPOINT):
    with open(os.path.join(checkpoint_dir, "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "svgp_model.pkl"), "rb") as f:
        model = pickle.load(f)
    print("Đã tải scaler_x, scaler_y và model thành công.")
    return scaler_x, scaler_y, model

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    scaler_x, scaler_y, model = load_scalers_and_model()
    test_files = load_file_list(TEST_LIST)
    
    all_original_targets_global = []
    all_predicted_targets_global = []
    total_prediction_time_global = 0
    
    # Biến cho khoảng tin cậy (3D)
    all_lower_bound_x_global = []
    all_upper_bound_x_global = []
    all_lower_bound_y_global = []
    all_upper_bound_y_global = []
    all_lower_bound_z_global = []
    all_upper_bound_z_global = []

    print(f"\n--- Bắt đầu dự đoán trên tập Test ({len(test_files)} files) ---")
    
    for fname in test_files:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(full_path):
            print(f"Cảnh báo: Không tìm thấy file: {full_path}. Bỏ qua.")
            continue
        
        print(f"\nĐang xử lý quỹ đạo: {fname}")
        df = read_and_smooth(full_path)
        full_sequence = df[["X", "Y", "Z"]].values  # Dữ liệu gốc 3D
        L = full_sequence.shape[0]
        
        if L < 2:  # Cần ít nhất 2 điểm (1 làm start, 1 để dự đoán)
            print(f"Quỹ đạo quá ngắn ({L} điểm). Yêu cầu ít nhất 2 điểm. Bỏ qua.")
            continue

        # Lấy điểm đầu tiên làm START_POINT cho quỹ đạo này
        start_point = full_sequence[0]  # (3,) - điểm đầu tiên (X, Y, Z)
        print(f"  START_POINT cho quỹ đạo này: {start_point}")
        
        # Tạo cửa sổ với padding bằng điểm đầu tiên
        X_test_traj, Y_test_traj = create_sequences_with_padding(full_sequence, WINDOW_SIZE, HORIZON, start_point)
        
        # Thêm điểm đầu tiên vào kết quả (không dự đoán, chỉ ghi nhận)
        first_point = full_sequence[0]

        current_trajectory_original_targets = []
        current_trajectory_predicted_targets = []
        
        current_trajectory_lower_bound_x = []
        current_trajectory_upper_bound_x = []
        current_trajectory_lower_bound_y = []
        current_trajectory_upper_bound_y = []
        current_trajectory_lower_bound_z = []
        current_trajectory_upper_bound_z = []

        start_time_trajectory = time.time()
        
        # Thêm điểm đầu tiên vào kết quả (prediction = observation = first_point)
        current_trajectory_original_targets.append(first_point)
        current_trajectory_predicted_targets.append(first_point.copy())
        # Khoảng tin cậy cho điểm đầu tiên = 0 (vì đã biết chính xác)
        current_trajectory_lower_bound_x.append(first_point[0])
        current_trajectory_upper_bound_x.append(first_point[0])
        current_trajectory_lower_bound_y.append(first_point[1])
        current_trajectory_upper_bound_y.append(first_point[1])
        current_trajectory_lower_bound_z.append(first_point[2])
        current_trajectory_upper_bound_z.append(first_point[2])
        
        all_original_targets_global.append(first_point)
        all_predicted_targets_global.append(first_point.copy())
        all_lower_bound_x_global.append(first_point[0])
        all_upper_bound_x_global.append(first_point[0])
        all_lower_bound_y_global.append(first_point[1])
        all_upper_bound_y_global.append(first_point[1])
        all_lower_bound_z_global.append(first_point[2])
        all_upper_bound_z_global.append(first_point[2])

        # Dự đoán các điểm còn lại (từ điểm thứ 2 trở đi)
        for i in range(X_test_traj.shape[0]):
            X_history = X_test_traj[i]
            Y_true = Y_test_traj[i]    

            # Reshape X_history để khớp với cách scaler_x được fit trong training (3D)
            X_window_reshaped = X_history.reshape(WINDOW_SIZE, 3)  # (10, 3)
            X_input_s = scaler_x.transform(X_window_reshaped).flatten().reshape(1, -1)  # (1, 30)
            
            mean_s, var_s = model.predict_y(X_input_s)
            
            predicted_point_unscaled = scaler_y.inverse_transform(mean_s.numpy()).flatten()
            
            # Xử lý phương sai cho 3D
            var_s_reshaped = var_s.numpy().reshape(-1, 3) if var_s.shape[1] > 3 else var_s.numpy()
            scale_squared_for_variance = np.square(scaler_y.scale_.reshape(1, -1))
            unscaled_variance = var_s_reshaped * scale_squared_for_variance
                
            std_dev_unscaled = np.sqrt(unscaled_variance).flatten()  # (3,)
            
            # Tính toán khoảng tin cậy 95% cho từng chiều (3D)
            lower_bound_x = predicted_point_unscaled[0] - 1.96 * std_dev_unscaled[0]
            upper_bound_x = predicted_point_unscaled[0] + 1.96 * std_dev_unscaled[0]
            lower_bound_y = predicted_point_unscaled[1] - 1.96 * std_dev_unscaled[1]
            upper_bound_y = predicted_point_unscaled[1] + 1.96 * std_dev_unscaled[1]
            lower_bound_z = predicted_point_unscaled[2] - 1.96 * std_dev_unscaled[2]
            upper_bound_z = predicted_point_unscaled[2] + 1.96 * std_dev_unscaled[2]
            
            current_trajectory_original_targets.append(Y_true)
            current_trajectory_predicted_targets.append(predicted_point_unscaled)

            current_trajectory_lower_bound_x.append(lower_bound_x)
            current_trajectory_upper_bound_x.append(upper_bound_x)
            current_trajectory_lower_bound_y.append(lower_bound_y)
            current_trajectory_upper_bound_y.append(upper_bound_y)
            current_trajectory_lower_bound_z.append(lower_bound_z)
            current_trajectory_upper_bound_z.append(upper_bound_z)
            
            all_original_targets_global.append(Y_true)
            all_predicted_targets_global.append(predicted_point_unscaled)
            all_lower_bound_x_global.append(lower_bound_x)
            all_upper_bound_x_global.append(upper_bound_x)
            all_lower_bound_y_global.append(lower_bound_y)
            all_upper_bound_y_global.append(upper_bound_y)
            all_lower_bound_z_global.append(lower_bound_z)
            all_upper_bound_z_global.append(upper_bound_z)
            
        end_time_trajectory = time.time()
        elapsed_time_trajectory = end_time_trajectory - start_time_trajectory
        total_prediction_time_global += elapsed_time_trajectory
        
        if current_trajectory_original_targets:
            current_orig = np.array(current_trajectory_original_targets)
            current_pred = np.array(current_trajectory_predicted_targets)
            current_lower_x = np.array(current_trajectory_lower_bound_x)
            current_upper_x = np.array(current_trajectory_upper_bound_x)
            current_lower_y = np.array(current_trajectory_lower_bound_y)
            current_upper_y = np.array(current_trajectory_upper_bound_y)
            current_lower_z = np.array(current_trajectory_lower_bound_z)
            current_upper_z = np.array(current_trajectory_upper_bound_z)

            mse_single_trajectory = np.mean((current_orig - current_pred)**2)
            rmse_single_trajectory = np.sqrt(mse_single_trajectory)
            
            coverage_x = np.mean((current_orig[:, 0] >= current_lower_x) & (current_orig[:, 0] <= current_upper_x)) * 100
            coverage_y = np.mean((current_orig[:, 1] >= current_lower_y) & (current_orig[:, 1] <= current_upper_y)) * 100
            coverage_z = np.mean((current_orig[:, 2] >= current_lower_z) & (current_orig[:, 2] <= current_upper_z)) * 100
            
            print(f"  Số điểm dự đoán: {len(current_orig)}")
            print(f"  MSE quỹ đạo: {mse_single_trajectory:.6f}")
            print(f"  RMSE quỹ đạo: {rmse_single_trajectory:.6f}")
            print(f"  Thời gian dự đoán: {elapsed_time_trajectory:.4f} giây")

        if current_trajectory_predicted_targets:
            time_indices = np.arange(len(current_orig)) / (191/12)  # 12s / 191 frames

            # --- Đồ thị cho trục X ---
            fig_x, ax_x = plt.subplots(figsize=(4.3, 4.3))
            ax_x.plot(time_indices, current_orig[:, 0], '-', color='blue', label='Observation', linewidth=1.5)
            ax_x.plot(time_indices, current_pred[:, 0], '--', color='red', label='Prediction', linewidth=1.5)
            ax_x.fill_between(time_indices, current_lower_x, current_upper_x, color='gray', alpha=0.2, label='95% CI')
            ax_x.set_xlabel('Timestamp (s)')
            ax_x.set_ylabel('X (m)')
            ax_x.legend()
            ax_x.grid(True)
            plt.tight_layout()
            plt.show()

            # --- Đồ thị cho trục Y ---
            fig_y, ax_y = plt.subplots(figsize=(4.3, 4.3))
            ax_y.plot(time_indices, current_orig[:, 1], '-', color='blue', label='Observation', linewidth=1.5)
            ax_y.plot(time_indices, current_pred[:, 1], '--', color='red', label='Prediction', linewidth=1.5)
            ax_y.fill_between(time_indices, current_lower_y, current_upper_y, color='gray', alpha=0.2, label='95% CI')
            ax_y.set_xlabel('Timestamp (s)')
            ax_y.set_ylabel('Y (m)')
            ax_y.legend()
            ax_y.grid(True)
            plt.tight_layout()
            plt.show()

            # --- Đồ thị cho trục Z ---
            fig_z, ax_z = plt.subplots(figsize=(4.3, 4.3))
            ax_z.plot(time_indices, current_orig[:, 2], '-', color='blue', label='Observation', linewidth=1.5)
            ax_z.plot(time_indices, current_pred[:, 2], '--', color='red', label='Prediction', linewidth=1.5)
            ax_z.fill_between(time_indices, current_lower_z, current_upper_z, color='gray', alpha=0.2, label='95% CI')
            ax_z.set_xlabel('Timestamp (s)')
            ax_z.set_ylabel('Z (m)')
            ax_z.legend()
            ax_z.grid(True)
            plt.tight_layout()
            plt.show()

            # --- Đồ thị 3D ---
            fig_3d = plt.figure(figsize=(6, 6))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot(current_orig[:, 0], current_orig[:, 1], current_orig[:, 2], '-', color='blue', label='Observation', linewidth=1.5)
            ax_3d.plot(current_pred[:, 0], current_pred[:, 1], current_pred[:, 2], '--', color='red', label='Prediction', linewidth=1.5)
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.legend()
            plt.tight_layout()
            plt.show()

    if all_original_targets_global:
        all_original_targets_global = np.array(all_original_targets_global)
        all_predicted_targets_global = np.array(all_predicted_targets_global)
        
        all_lower_bound_x_global = np.array(all_lower_bound_x_global)
        all_upper_bound_x_global = np.array(all_upper_bound_x_global)
        all_lower_bound_y_global = np.array(all_lower_bound_y_global)
        all_upper_bound_y_global = np.array(all_upper_bound_y_global)
        all_lower_bound_z_global = np.array(all_lower_bound_z_global)
        all_upper_bound_z_global = np.array(all_upper_bound_z_global)

        coverage_global_x = np.mean((all_original_targets_global[:, 0] >= all_lower_bound_x_global) & (all_original_targets_global[:, 0] <= all_upper_bound_x_global)) * 100
        coverage_global_y = np.mean((all_original_targets_global[:, 1] >= all_lower_bound_y_global) & (all_original_targets_global[:, 1] <= all_upper_bound_y_global)) * 100
        coverage_global_z = np.mean((all_original_targets_global[:, 2] >= all_lower_bound_z_global) & (all_original_targets_global[:, 2] <= all_upper_bound_z_global)) * 100
        
        mse_global = np.mean((all_original_targets_global - all_predicted_targets_global)**2)
        rmse_global = np.sqrt(mse_global)
        
        print(f"\n--- Kết quả tổng thể trên tập Test ---")
        print(f"Tổng số điểm dự đoán: {len(all_original_targets_global)}")
        print(f"Test MSE:  {mse_global:.6f}")
        print(f"Test RMSE: {rmse_global:.6f}")
        print(f"Độ bao phủ KTC 95% (X): {coverage_global_x:.2f}%")
        print(f"Độ bao phủ KTC 95% (Y): {coverage_global_y:.2f}%")
        print(f"Độ bao phủ KTC 95% (Z): {coverage_global_z:.2f}%")
        print(f"Tổng thời gian dự đoán: {total_prediction_time_global:.4f} giây")
    else:
        print("\nKhông có điểm nào được dự đoán trên tập test. Vui lòng kiểm tra lại dữ liệu test và cấu hình.")

if __name__ == "__main__":
    main()