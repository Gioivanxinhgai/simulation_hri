import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
TEST_FOLDER = "Tra_test"
TEST_LIST   = os.path.join(TEST_FOLDER, "test_file_list.json")
CHECKPOINT  = "gp_full"
OUTPUT_DIR  = "Test_Results_SVGP"  # Thư mục lưu file .csv kết quả
WINDOW_SIZE = 10
HORIZON     = 1 # Trong bài toán này, HORIZON luôn là 1 để dự đoán điểm tiếp theo
START_POINT = 0.0

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1):
    df = pd.read_csv(filename)[["x", "y"]].dropna()
    df["x"] = gaussian_filter1d(df["x"].values, sigma=sigma)
    df["y"] = gaussian_filter1d(df["y"].values, sigma=sigma)
    return df.reset_index(drop=True)

# Hàm tương tự từ code train để tạo cửa sổ với padding
def create_sequences_with_padding(data_sequence, window_size, horizon, padding_value=START_POINT):
    X_sequences = []
    Y_targets = []
    L = data_sequence.shape[0] # Số điểm của quỹ đạo (L)
    num_features = data_sequence.shape[1] # Số chiều của mỗi điểm (ở đây là 2: x, y)
    for i in range(L): # Lặp qua tất cả các chỉ số của data_sequence
        current_window = np.full((window_size, num_features), padding_value, dtype=data_sequence.dtype)
        actual_points_end_idx = i
        actual_points_start_idx = max(0, i - window_size) # Bắt đầu từ 0 hoặc window_size điểm trước i

        num_actual_points = actual_points_end_idx - actual_points_start_idx

        if num_actual_points > 0:
            current_window[window_size - num_actual_points : window_size] = data_sequence[actual_points_start_idx : actual_points_end_idx]
        
        X_sequences.append(current_window.flatten()) # Làm phẳng cửa sổ để phù hợp với đầu vào của GP
        Y_targets.append(data_sequence[i : i + 1].flatten()) # Điểm mục tiêu là chính điểm hiện tại 'i'

    return np.array(X_sequences), np.array(Y_targets)

def load_scalers_and_model(checkpoint_dir=CHECKPOINT):
    with open(os.path.join(checkpoint_dir, "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "gp_full.pkl"), "rb") as f:
        model = pickle.load(f)
    print("Đã tải scaler_x, scaler_y và model thành công.")
    return scaler_x, scaler_y, model

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scaler_x, scaler_y, model = load_scalers_and_model()
    test_files = load_file_list(TEST_LIST)
    
    all_original_targets_global = []
    all_predicted_targets_global = []
    total_prediction_time_global = 0
    all_lower_bound_x_global = []
    all_upper_bound_x_global = []
    all_lower_bound_y_global = []
    all_upper_bound_y_global = []

    print("\n--- Bắt đầu dự đoán trên tập Test ---")
    
    for fname in test_files:
        full_path = os.path.join(TEST_FOLDER, fname)
        if not os.path.isfile(full_path):
            print(f"Cảnh báo: Không tìm thấy file: {full_path}. Bỏ qua.")
            continue
        
        print(f"\nĐang xử lý quỹ đạo: {fname}")
        df = read_and_smooth(full_path)
        full_sequence = df[["x","y"]].values # Dữ liệu gốc của quỹ đạo
        L = full_sequence.shape[0]
        
        if L < HORIZON:
            print(f"Quỹ đạo quá ngắn ({L} điểm). Yêu cầu ít nhất {HORIZON} điểm để tạo target. Bỏ qua.")
            continue

        X_test_traj, Y_test_traj = create_sequences_with_padding(full_sequence, WINDOW_SIZE, HORIZON)

        current_trajectory_original_targets = []
        current_trajectory_predicted_targets = []
        
        current_trajectory_lower_bound_x = []
        current_trajectory_upper_bound_x = []
        current_trajectory_lower_bound_y = []
        current_trajectory_upper_bound_y = []
        current_trajectory_var_x = []
        current_trajectory_var_y = []

        start_time_trajectory = time.time()

        for i in range(X_test_traj.shape[0]):
            X_history = X_test_traj[i]
            Y_true = Y_test_traj[i]    

            # Reshape X_history để khớp với cách scaler_x được fit trong training
            X_window_reshaped = X_history.reshape(WINDOW_SIZE, 2)  # (10, 2)
            X_input_s = scaler_x.transform(X_window_reshaped).flatten().reshape(1, -1)  # (1, 20)
            
            mean_s, var_s = model.predict_y(X_input_s) #var_s = diag(cov) trong công thức gốc
            
            predicted_point_unscaled = scaler_y.inverse_transform(mean_s.numpy()).flatten() # mean_s đã là (1, 2)
            
            # Xử lý var_s
            var_s_reshaped = var_s.numpy().reshape(-1, 2) if var_s.shape[1] > 2 else var_s.numpy()  # (1, 2)
            scale_squared_for_variance = np.square(scaler_y.scale_.reshape(1, -1))  # (1, 2)
            unscaled_variance = var_s_reshaped * scale_squared_for_variance  # (1, 2)
            
            std_dev_unscaled = np.sqrt(unscaled_variance).flatten()  # (2,)
            # Tính toán khoảng tin cậy 95% cho từng chiều (giá trị Z = 1.96 cho 95%)
            lower_bound_x = predicted_point_unscaled[0] - 1.96 * std_dev_unscaled[0]
            upper_bound_x = predicted_point_unscaled[0] + 1.96 * std_dev_unscaled[0]
            lower_bound_y = predicted_point_unscaled[1] - 1.96 * std_dev_unscaled[1]
            upper_bound_y = predicted_point_unscaled[1] + 1.96 * std_dev_unscaled[1]
            
            current_trajectory_original_targets.append(Y_true)
            current_trajectory_predicted_targets.append(predicted_point_unscaled)
            # Lưu phương sai (đơn vị gốc) cho từng trục
            current_trajectory_var_x.append(unscaled_variance.flatten()[0])
            current_trajectory_var_y.append(unscaled_variance.flatten()[1])

            current_trajectory_lower_bound_x.append(lower_bound_x)
            current_trajectory_upper_bound_x.append(upper_bound_x)
            current_trajectory_lower_bound_y.append(lower_bound_y)
            current_trajectory_upper_bound_y.append(upper_bound_y)
            
            all_original_targets_global.append(Y_true)
            all_predicted_targets_global.append(predicted_point_unscaled)
            all_lower_bound_x_global.append(lower_bound_x)
            all_upper_bound_x_global.append(upper_bound_x)
            all_lower_bound_y_global.append(lower_bound_y)
            all_upper_bound_y_global.append(upper_bound_y)
            
        end_time_trajectory = time.time()
        elapsed_time_trajectory = end_time_trajectory - start_time_trajectory
        total_prediction_time_global += elapsed_time_trajectory
        
        if current_trajectory_original_targets:
            current_orig    = np.array(current_trajectory_original_targets)
            current_pred    = np.array(current_trajectory_predicted_targets)
            current_lower_x = np.array(current_trajectory_lower_bound_x)
            current_upper_x = np.array(current_trajectory_upper_bound_x)
            current_lower_y = np.array(current_trajectory_lower_bound_y)
            current_upper_y = np.array(current_trajectory_upper_bound_y)
            current_var_x   = np.array(current_trajectory_var_x)
            current_var_y   = np.array(current_trajectory_var_y)

            mse_single_trajectory = np.mean((current_orig - current_pred)**2)
            rmse_single_trajectory = np.sqrt(mse_single_trajectory)
            
            coverage_x = np.mean((current_orig[:, 0] >= current_lower_x) & (current_orig[:, 0] <= current_upper_x)) * 100
            coverage_y = np.mean((current_orig[:, 1] >= current_lower_y) & (current_orig[:, 1] <= current_upper_y)) * 100
            
            print(f" Số điểm dự đoán cho quỹ đạo này (bao gồm padding): {len(current_orig)}")
            print(f" MSE quỹ đạo: {mse_single_trajectory:.6f}")
            print(f" RMSE quỹ đạo: {rmse_single_trajectory:.6f}")
            #print(f" Độ bao phủ KTC 95% (trục x): {coverage_x:.2f}%")
            #print(f" Độ bao phủ KTC 95% (trục y): {coverage_y:.2f}%")
            print(f" Thời gian dự đoán cho quỹ đạo này: {elapsed_time_trajectory:.4f} giây")

            # Lưu kết quả dự đoán và phương sai theo từng trục ra file .csv
            # time_indices giả sử dữ liệu được lấy mẫu với tần số 10 Hz -> bước thời gian 0.1 s
            time_indices = np.arange(len(current_orig)) / 10.0
            result_df = pd.DataFrame({
                "t": time_indices,
                "x_obs": current_orig[:, 0],
                "y_obs": current_orig[:, 1],
                "x_pred": current_pred[:, 0],
                "y_pred": current_pred[:, 1],
                "var_x": current_var_x,
                "var_y": current_var_y,
                "lower_x": current_lower_x,
                "upper_x": current_upper_x,
                "lower_y": current_lower_y,
                "upper_y": current_upper_y,
            })
            base_name = os.path.splitext(os.path.basename(fname))[0]
            out_path = os.path.join(OUTPUT_DIR, f"pred_var_{base_name}.csv")
            result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f" Đã lưu kết quả dự đoán và phương sai vào: {out_path}")

        if current_trajectory_predicted_targets:
            # Lấy chỉ số thời gian cho trục x của đồ thị
            time_indices = np.arange(len(current_orig))/10

            # --- Đồ thị cho trục X (xuất hiện riêng) ---
            fig_x, ax_x = plt.subplots(figsize=(4.3, 4.3)) # figsize được đổi thành vuông vắn
            ax_x.plot(time_indices, current_orig[:, 0], '-', color='blue', label='Observation', linewidth=1.5)
            ax_x.plot(time_indices, current_pred[:, 0], '--', color='red', label='Prediction', linewidth=1.5)
            ax_x.fill_between(
                time_indices,
                current_lower_x,
                current_upper_x,
                color='gray', alpha=0.2, label='KTC 95% (X)'
            ) 
            ax_x.set_xlabel('time(s)')
            ax_x.set_ylabel('x(m)')
            ax_x.legend()
            ax_x.grid(True)
            ax_x.autoscale(enable=True, axis='both', tight=True)
            ax_x.margins(x=0.005, y=0.0)
            plt.show()

            # --- Đồ thị cho trục Y (xuất hiện riêng) ---
            fig_y, ax_y = plt.subplots(figsize=(4.3, 4.3)) # figsize được đổi thành vuông vắn
            ax_y.plot(time_indices, current_orig[:, 1], '-', color='blue',label='Observation', linewidth=1.5)
            ax_y.plot(time_indices, current_pred[:, 1], '--', color='red', label='Prediction', linewidth=1.5)
            ax_y.fill_between(
                time_indices,
                current_lower_y,
                current_upper_y,
                color='gray', alpha=0.2, label='95% CI'
            )
            ax_y.set_xlabel('Time(s)')
            ax_y.set_ylabel('y(m)')
            ax_y.legend()
            ax_y.grid(True)
            ax_y.autoscale(enable=True, axis='both', tight=True)
            ax_y.margins(x=0.005, y=0.0)
            plt.show()

            # --- Đồ thị (X, Y) với trục thời gian phía trên ---
            fig_xy, ax_xy = plt.subplots(figsize=(4.3, 4.3))
            
            # 1. Tạo vùng KTC 95% (Giữ nguyên logic của bạn)
            upper_right_boundary = np.stack([current_upper_x, current_upper_y], axis=1)
            lower_left_boundary_reversed = np.stack([current_lower_x, current_lower_y], axis=1)[::-1]
            polygon_boundary = np.vstack([upper_right_boundary, lower_left_boundary_reversed])
            
            ax_xy.fill(
                polygon_boundary[:, 0], 
                polygon_boundary[:, 1], 
                color='gray', 
                alpha=0.2, 
                label='95% CI'
            )
            
            # 2. Vẽ quỹ đạo thực tế và dự đoán (Trục X bottom, Trục Y left)
            ax_xy.plot(current_orig[:, 0], current_orig[:, 1], '-', color='blue', label='Observation', linewidth=1.5)
            ax_xy.plot(current_pred[:, 0], current_pred[:, 1], '--', color='red', label='Prediction', linewidth=1.5)

            ax_xy.set_xlabel('x(m)')
            ax_xy.set_ylabel('y(m)')
            # 4. Cấu hình chung
            ax_xy.legend(loc='upper left')
            ax_xy.grid(True)
            ax_xy.autoscale(enable=True, axis='both', tight=False)
            ax_xy.margins(x=0.008, y=0.005)
            
            plt.tight_layout() # Đảm bảo các nhãn không bị đè lên nhau
            plt.show()

    if all_original_targets_global:
        all_original_targets_global = np.array(all_original_targets_global)
        all_predicted_targets_global = np.array(all_predicted_targets_global)
        
        all_lower_bound_x_global = np.array(all_lower_bound_x_global)
        all_upper_bound_x_global = np.array(all_upper_bound_x_global)
        all_lower_bound_y_global = np.array(all_lower_bound_y_global)
        all_upper_bound_y_global = np.array(all_upper_bound_y_global)

        coverage_global_x = np.mean((all_original_targets_global[:, 0] >= all_lower_bound_x_global) & (all_original_targets_global[:, 0] <= all_upper_bound_x_global)) * 100
        coverage_global_y = np.mean((all_original_targets_global[:, 1] >= all_lower_bound_y_global) & (all_original_targets_global[:, 1] <= all_upper_bound_y_global)) * 100
        
        mse_global  = np.mean((all_original_targets_global - all_predicted_targets_global)**2)
        rmse_global = np.sqrt(mse_global)
        print(f"\n--- Kết quả tổng thể trên tập Test ---")
        print(f"Tổng số điểm dự đoán (bao gồm padding): {len(all_original_targets_global)}")
        print(f"Test MSE:  {mse_global:.6f}")
        print(f"Test RMSE: {rmse_global:.6f}")
        print(f"Độ bao phủ KTC 95% (trục x): {coverage_global_x:.2f}%")
        print(f"Độ bao phủ KTC 95% (trục y): {coverage_global_y:.2f}%")
        print(f"Tổng thời gian dự đoán (bao gồm I/O, chuẩn hóa): {total_prediction_time_global:.4f} giây")
    else:
        print("\nKhông có điểm nào được dự đoán trên tập test. Vui lòng kiểm tra lại dữ liệu test và cấu hình.")

if __name__ == "__main__":
    main()