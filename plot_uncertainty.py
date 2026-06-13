import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import time

# Thiết lập font Times New Roman cho đồ thị
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_DIR = "Experiment_Test_Trajectory_HRI"
TEST_LIST = os.path.join(DATA_DIR, "test_file_list.json")
CHECKPOINT = "svgp_hri_rbf+m52"
WINDOW_SIZE = 10
HORIZON = 1

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_file_list(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_and_smooth(filename, sigma=1):
    """Đọc file CSV, tự động phát hiện 2D (x,y) hoặc 3D (X,Y,Z)."""
    df = pd.read_csv(filename).dropna()
    
    # Phát hiện tự động: 3D nếu có cột X, Y, Z (viết hoa); 2D nếu có x, y (viết thường)
    if {'X', 'Y', 'Z'}.issubset(df.columns):
        cols = ['X', 'Y', 'Z']
    elif {'x', 'y'}.issubset(df.columns):
        cols = ['x', 'y']
    else:
        raise ValueError(f"Không tìm thấy cột tọa độ phù hợp trong file: {filename}")
    
    for c in cols:
        df[c] = gaussian_filter1d(df[c].values, sigma=sigma)
    
    return df[cols].reset_index(drop=True), cols

def create_sequences_with_padding(data_sequence, window_size, horizon, start_point):
    X_sequences = []
    Y_targets = []
    L = data_sequence.shape[0]
    num_features = data_sequence.shape[1]
    
    for i in range(1, L):
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
    with open(os.path.join(checkpoint_dir, "scaler_uncertainty.pkl"), "rb") as f:
        uncertainty_scaler = pickle.load(f)
    u_min = uncertainty_scaler["u_min"]
    u_max = uncertainty_scaler["u_max"]
    print(f"Loaded scaler_x, scaler_y, model and uncertainty scaler (u_min={u_min:.6f}, u_max={u_max:.6f}).")
    return scaler_x, scaler_y, model, u_min, u_max

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    scaler_x, scaler_y, model, u_min, u_max = load_scalers_and_model()
    test_files = load_file_list(TEST_LIST)
    
    print(f"\n--- Starting Uncertainty calculation on Test set ({len(test_files)} files) ---")
    
    for fname in test_files:
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(full_path):
            print(f"Warning: File not found: {full_path}. Skipping.")
            continue
        
        print(f"\nProcessing trajectory: {fname}")
        df, cols = read_and_smooth(full_path)
        full_sequence = df.values
        L = full_sequence.shape[0]
        num_features = full_sequence.shape[1]  # 2 hoặc 3 tự động
        
        print(f"  Detected {num_features}D data (columns: {cols})")
        
        if L < 2:
            print(f"Trajectory too short ({L} points). Skipping.")
            continue

        start_point = full_sequence[0]
        print(f"  START_POINT: {start_point}")
        
        X_test_traj, Y_test_traj = create_sequences_with_padding(full_sequence, WINDOW_SIZE, HORIZON, start_point)
        
        # Storage cho combined variance
        variance_combined_list = [0.0]  # Điểm đầu tiên có variance = 0
        
        # Dự đoán các điểm còn lại
        for i in range(X_test_traj.shape[0]):
            X_history = X_test_traj[i]
            
            X_window_reshaped = X_history.reshape(WINDOW_SIZE, num_features)
            X_input_s = scaler_x.transform(X_window_reshaped).flatten().reshape(1, -1)
            
            mean_s, var_s = model.predict_y(X_input_s)
            
            # Tính combined variance (σ² = Σ σ²_i cho tất cả các trục)
            scale_squared = np.square(scaler_y.scale_)
            var_unscaled = var_s.numpy().flatten() * scale_squared
            combined_var = np.sum(var_unscaled)
            
            variance_combined_list.append(combined_var)
        
        # Convert to numpy arrays
        variance_combined = np.array(variance_combined_list)
        std_combined = np.sqrt(variance_combined)
        
        # ─── Min-Max Normalization (u_min, u_max từ scaler_uncertainty) ──
        if u_max - u_min > 0:
            std_normalized = (std_combined - u_min) / (u_max - u_min)
            std_normalized = np.clip(std_normalized, 0, 1)  # Clip tại [0, 1]
        else:
            std_normalized = np.zeros_like(std_combined)
        
        # Time indices (assuming 10 Hz sampling rate)
        time_indices = np.arange(len(variance_combined)) / 10.0
        
        # ═══════════════════════════════════════════════════════════════════
        # PLOT: Combined Uncertainty (Normalized)
        # ═══════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(time_indices, std_normalized, '-', color='tab:red', linewidth=1.5, label='Normalized Uncertainty')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Uncertainty [0, 1]')
        ax.set_title(f'Combined Uncertainty (Min-Max Normalized, {num_features}D) - {fname}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # ═══════════════════════════════════════════════════════════════════
        # PRINT STATISTICS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  --- Combined Uncertainty Statistics ---")
        print(f"  u_min = {u_min:.6f} m, u_max = {u_max:.6f} m")
        print(f"  Normalized range: [{np.min(std_normalized):.4f}, {np.max(std_normalized):.4f}]")

    print("\n--- Hoàn tất ---")

if __name__ == "__main__":
    main()
