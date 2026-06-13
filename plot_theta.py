import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Thêm thư mục chứa file config.py và shared_control_lib.py vào sys.path nếu chạy file ngoài
sys.path.append(r"d:\LAB")

from shared_control_lib import add_control_mode_background, control_mode_legend_patches
from config import PHI_ANGLE

# Thiết lập Font chữ chuẩn bài báo
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

def plot_theta_from_csv(csv_path):
    """
    Đọc dữ liệu từ file CSV quỹ đạo và vẽ lại đồ thị góc Theta.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File không tồn tại - {csv_path}")
        return

    # 1. Đọc dữ liệu
    df = pd.read_csv(csv_path)
    
    trajectory_name = os.path.basename(csv_path).replace('.csv', '')
    time_array = df['time_s'].values
    modes = df['mode'].values
    
    # Lấy mảng theta đã được tính trong CSV 
    # (Trường hợp CSV không có thì mới tính lại từ f_h và f_r)
    if 'theta_deg' in df.columns:
        theta_arr = df['theta_deg'].values
    else:
        print("Không tìm thấy cột theta_deg trong CSV, tính toán lại...")
        from config import FORCE_DEADZONE
        n_steps = len(df)
        f_h_history = df[['F_h_X', 'F_h_Y', 'F_h_Z']].values
        f_r_history = df[['F_r_X', 'F_r_Y', 'F_r_Z']].values
        theta_arr = np.zeros(n_steps)
        norms_h = np.linalg.norm(f_h_history, axis=1)
        
        for i in range(n_steps):
            f_h = f_h_history[i]
            f_r = f_r_history[i]
            norm_h = norms_h[i]
            norm_r = np.linalg.norm(f_r)
            if norm_r == 0:
                cos_theta = 0.0
            elif norm_h < FORCE_DEADZONE:
                cos_theta = 1.0
            else:
                cos_theta = np.clip(np.dot(f_h, f_r) / (norm_h * norm_r), -1.0, 1.0)
            theta_arr[i] = np.arccos(cos_theta) * (180.0 / np.pi)

    # 2. Lam min (Low-pass filter) voi thong so SIGMA = 3.0 giong ham goc
    SIGMA_SMOOTH = 3.0
    theta_smoothed = gaussian_filter1d(theta_arr, sigma=SIGMA_SMOOTH)

    # 3. Tinh running cumulative mean theo Eq.(52):
    #    Angle(t) = (1/t) * integral_0^t  arccos( f_r^T f_h / (||f_r|| ||f_h||) ) dtau
    #    Su dung tich phan bang trapz tich luy
    n_steps = len(time_array)
    cumulative_integral = np.zeros(n_steps)
    for k in range(1, n_steps):
        # trapz tich luy tu 0 -> k
        cumulative_integral[k] = np.trapz(theta_smoothed[:k+1], time_array[:k+1])
    
    running_mean_theta = np.zeros(n_steps)
    running_mean_theta[0] = theta_smoothed[0]  # tai t=0 chua co tich phan, lay gia tri tuc thoi
    for k in range(1, n_steps):
        T_k = time_array[k] - time_array[0]
        if T_k > 0:
            running_mean_theta[k] = cumulative_integral[k] / T_k
        else:
            running_mean_theta[k] = theta_smoothed[k]

    # Gia tri mean cuoi cung (toan bo quy dao)
    T_total = time_array[-1] - time_array[0]
    final_mean_theta = cumulative_integral[-1] / T_total if T_total > 0 else 0.0

    # 4. Ve hai do thi rieng biet
    save_dir = os.path.dirname(csv_path)
    mode_patches = control_mode_legend_patches()

    # 4a. Do thi Theta tuc thoi (smoothed)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    add_control_mode_background(ax1, list(modes), time_array, zorder=0)

    ax1.plot(time_array, theta_smoothed, '--', color='tab:blue', linewidth=2.0,
            label='Theta (instantaneous)', zorder=2)
    ax1.axhline(PHI_ANGLE, color='tab:orange', linewidth=2.5, label=f'Standard', zorder=2)
    
    ax1.set_ylabel('Theta [deg]')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 180)
    
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + mode_patches, labels=l1 + [p.get_label() for p in mode_patches], loc='upper left')
    
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    ax1.tick_params(width=1.5)
    
    plt.tight_layout()
    save_path1 = os.path.join(save_dir, f'theta_instant_replot_{trajectory_name}.png')
    fig1.savefig(save_path1, dpi=150)
    print(f"Da luu do thi Theta tuc thoi tai: {save_path1}")

    # 4b. Do thi Running Mean Theta theo Eq.(52)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    add_control_mode_background(ax2, list(modes), time_array, zorder=0)

    ax2.plot(time_array, running_mean_theta, '-', color='tab:blue', linewidth=2.0,
            label=f'Theta (average)', zorder=3)
    ax2.axhline(PHI_ANGLE, color='tab:orange', linewidth=2.5, label=f'Standard', zorder=2)
    
    ax2.set_ylabel('Theta [deg]')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 180)
    
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=h2 + mode_patches, labels=l2 + [p.get_label() for p in mode_patches], loc='upper left')
    
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    ax2.tick_params(width=1.5)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f'theta_running_mean_replot_{trajectory_name}.png')
    fig2.savefig(save_path2, dpi=150)
    print(f"Da luu do thi Running Mean Theta tai: {save_path2}")

    print(f"Final Mean Theta (Eq.52): {final_mean_theta:.2f} degrees")
    plt.close('all')

if __name__ == "__main__":
    csv_file = r"d:\LAB\MJM_CURRENT_10_10_60_5_5_0.75_3_150_1000000_1400.00_GTCHEAT\csv_logs\trajectory_048.csv"
    plot_theta_from_csv(csv_file)
