import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings("ignore")

# --- Import tham số tập trung từ config.py ---
from config import (
    GMM_CHECKPOINT_DIR, GMM_MODEL_PATH, GMM_SCALER_PATH,
    SVGP_CHECKPOINT_DIR, TEST_FOLDER,
    P_HIGH, P_HYSTERESIS, DT,
    SVGP_HISTORY_SIZE, GMM_WINDOW_SIZE, TAU_SOFTMAX, SIGMA,
    FITTS_A, FITTS_B, FITTS_W,
    GOALS, GROUND_TRUTH, PHI_ANGLE
)

class ControlMode:
    FOLLOWER = "FOLLOWER"
    LEADER = "LEADER"

MODE_BACKGROUND_COLORS = {
    ControlMode.FOLLOWER: None,       # Nền trắng (không tô)
    ControlMode.LEADER: "#d62728",    # Màu đỏ nhạt / Pastel Red
}
MODE_BACKGROUND_ALPHA = 0.3
MIN_SHADE_STEPS = 1  # Chỉ tô màu nếu mode kéo dài ít nhất 3 bước (tránh rối mắt)


def add_control_mode_background(ax, modes, time_steps, zorder=0):
    n = len(modes)
    if n == 0 or len(time_steps) != n:
        return
    i = 0
    while i < n:
        current_mode = modes[i]
        start_idx = i
        while i < n and modes[i] == current_mode:
            i += 1
        end_idx = i
        color = MODE_BACKGROUND_COLORS.get(current_mode)
        if color is not None and (end_idx - start_idx) >= MIN_SHADE_STEPS:
            t0 = float(time_steps[start_idx])
            t1 = float(time_steps[end_idx - 1]) if end_idx > start_idx else t0
            if end_idx < n:
                t1 = float(time_steps[end_idx])
            else:
                dt = float(time_steps[1] - time_steps[0]) if n > 1 else 1.0
                t1 = t1 + dt
            ax.axvspan(t0, t1, facecolor=color, alpha=MODE_BACKGROUND_ALPHA, zorder=zorder, linewidth=0)
        
        i = end_idx


def control_mode_legend_patches():
    """Patch cho legend: giải thích nền LEADER / FOLLOWER / SHARED."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor="white", edgecolor="gray", alpha=0.5, label="FOLLOWER"),
        Patch(facecolor=MODE_BACKGROUND_COLORS[ControlMode.LEADER],
              alpha=MODE_BACKGROUND_ALPHA + 0.1, edgecolor="none", label="LEADER"),
    ]

# --- Font ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10


# === HELPERS ===

def load_gmm_system(model_path, scaler_path):
    """Load GMM models và scaler."""
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return models, scaler

def load_svgp_system(checkpoint_dir):
    """Load SVGP model và scalers."""
    try:
        with open(os.path.join(checkpoint_dir, "scaler_x.pkl"), "rb") as f:
            scaler_x = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "scaler_y.pkl"), "rb") as f:
            scaler_y = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "svgp_model.pkl"), "rb") as f:
            model = pickle.load(f)
        return scaler_x, scaler_y, model
    except FileNotFoundError as e:
        print(f"[!] Error loading SVGP checkpoints: {e}")
        return None, None, None

def read_and_smooth_3d(filename, sigma=1):
    """Đọc file CSV và làm mịn dữ liệu 3D. Trả về df và tên quỹ đạo."""
    try:
        if not os.path.exists(filename):
            return None, None
        df = pd.read_csv(filename)
        
        trajectory_name = os.path.splitext(os.path.basename(filename))[0]
        
        required_cols = {'X', 'Y', 'Z'}
        if not required_cols.issubset(df.columns):
            return None, None
        
        df = df.dropna()
        
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
        
        return df, trajectory_name
    except Exception as e:
        print(f"  [!] Error reading file: {e}")
        return None, None

def dynamic_minimum_jerk_trajectory(x_0, v_0, x_f, t_total, dt, a_0=None):
    if a_0 is None:
        a_0 = np.zeros_like(x_0)
    
    t_total = max(t_total, dt) # Đảm bảo T >= dt để tránh chia cho 0
    n_steps = int(t_total / dt) + 1
    t_array = np.linspace(0, t_total, n_steps)
    t = t_array[:, np.newaxis]

    T = t_total
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    delta_x = x_f - x_0
    
    c0 = x_0
    c1 = v_0
    c2 = a_0 / 2.0
    c3 = (20 * delta_x - (12 * v_0) * T - (3 * a_0) * T2) / (2 * T3)
    c4 = (-30 * delta_x + (16 * v_0) * T + (3 * a_0) * T2) / (2 * T4)
    c5 = (12 * delta_x - (6 * v_0) * T - a_0 * T2) / (2 * T5)
    trajectory = c0 + c1*t + c2*(t**2) + c3*(t**3) + c4*(t**4) + c5*(t**5)
    
    return trajectory



# ──────────────────────────────────────────────────────────────────────────────
# PAPER MJM — Implement đúng theo phương trình (28) và (30) trong bài báo
# ──────────────────────────────────────────────────────────────────────────────

def _mjm_paper_position(tau: float, x_0: np.ndarray, x_f: np.ndarray) -> np.ndarray:
    """
    Phương trình (30): X_r(t) = X_0 + (X_0 - X_f)(15τ^4 - 6τ^5 - 10τ^3)
    Ghi chú: Trong bài báo đây là kết quả tích phân của Eq.28.
    """
    poly = 15 * tau**4 - 6 * tau**5 - 10 * tau**3
    return x_0 + (x_0 - x_f) * poly


def _mjm_paper_velocity_normalized(tau: float, x_0: np.ndarray, x_f: np.ndarray) -> np.ndarray:
    """
    Phương trình (28): dX/dτ = (X_0 - X_f)(60τ^3 - 30τ^4 - 30τ^2)
    Vận tốc thực: dX/dt = dX/dτ * (1/t_f)
    """
    dpoly_dtau = 60 * tau**3 - 30 * tau**4 - 30 * tau**2
    return (x_0 - x_f) * dpoly_dtau


def _find_tau_from_position(x_current: np.ndarray,
                             x_0: np.ndarray, x_f: np.ndarray,
                             tau_prev: float = 0.0,
                             tol: float = 1e-6) -> float:
    """
    Giải ngược phương trình (30) để tìm τ ứng với vị trí x_current.

    Từ Eq.30: X_r - X_0 = (X_0 - X_f) * g(τ)  với g(τ) = 15τ⁴ - 6τ⁵ - 10τ³
    Đặt p(τ) = -g(τ) = 10τ³ - 15τ⁴ + 6τ⁵  (progress function)
      → p'(τ) = 30τ²(1-τ)² ≥ 0  ⟹ đơn điệu TĂNG trên [0,1]
      → p(0) = 0,  p(1) = 1

    Chiếu x_current lên trục X_0→X_f để tính progress, rồi bisection tìm τ.
    """
    direction = x_f - x_0
    dist_total = np.linalg.norm(direction)
    if dist_total < 1e-9:
        return 1.0  # Đã ở đích

    # Chiếu x_current lên trục X_0→X_f
    unit = direction / dist_total
    proj_current = np.dot(x_current - x_0, unit)

    # X_r - X_0 = (X_0 - X_f) * g(τ)  →  chiếu lên unit:
    #   proj = dot((X_0 - X_f)*g(τ), unit) = -dist_total * g(τ) = dist_total * p(τ)
    # ⟹ target_p = proj_current / dist_total
    target_p = np.clip(proj_current / dist_total, 0.0, 1.0)

    # Bisection tìm τ* sao cho p(τ) = target_p
    # p(τ) = 10τ³ - 15τ⁴ + 6τ⁵  đơn điệu tăng [0,1] → [0,1]
    lo, hi = max(tau_prev - 0.05, 0.0), 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        p_mid = 10 * mid**3 - 15 * mid**4 + 6 * mid**5
        if p_mid < target_p:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def paper_mjm_step(x_current: np.ndarray,
                   x_0: np.ndarray, x_f: np.ndarray,
                   t_f: float, dt: float,
                   tau_prev: float = 0.0) -> tuple:
    """
    Một bước MJM theo cách bài báo (phương trình 28, 29, 30).

    Quy trình:
      1. Tìm τ_current ứng với x_current bằng cách giải ngược Eq.30
      2. Nhích τ lên một bước: τ_next = τ_current + dt/t_f
      3. Tính X_d(t+1) từ Eq.30 tại τ_next
      4. Tính V_d(t+1) từ Eq.28 tại τ_next (scale về m/s)

    Tham số:
      x_current : vị trí robot hiện tại (3,)
      x_0       : điểm xuất phát của hành trình (ghi nhận lúc vào LEADER lần đầu)
      x_f       : điểm đích (3,)
      t_f       : tổng thời gian hành trình = 12s (cố định theo dataset)
      dt        : time-step (s)
      tau_prev  : τ ở bước trước (để khởi điểm bisection nhanh hơn)

    Trả về:
      (x_desired, v_desired, tau_next)
        x_desired : vị trí mong muốn ở bước t+1  (3,)
        v_desired : vận tốc mong muốn ở bước t+1 (3,)   [m/s]
        tau_next  : τ mới (truyền lại cho bước sau)
    """
    tau_current = _find_tau_from_position(x_current, x_0, x_f, tau_prev=tau_prev)
    tau_next    = min(tau_current + dt / t_f, 1.0)

    x_desired = _mjm_paper_position(tau_next, x_0, x_f)
    v_desired = _mjm_paper_velocity_normalized(tau_next, x_0, x_f) / t_f  # chuyển sang m/s

    return x_desired, v_desired, tau_next


def fitts_law_duration(x_current, x_goal, a=FITTS_A, b=FITTS_B, w=FITTS_W):
    D = np.linalg.norm(x_current - x_goal)
    if D < w / 2:  
        return DT * 5  
    t_f = a + b * np.log2(2 * D / w)
    return max(t_f, DT * 5)

def predict_next_point_svgp(points_history, scaler_x, scaler_y, model, 
                            history_size, padding_value=None):
    L = points_history.shape[0]
    num_features = points_history.shape[1]
    
    if padding_value is None:
        padding_value = points_history[0] if L > 0 else np.zeros(num_features)
    
    current_window = np.tile(padding_value, (history_size, 1)).astype(points_history.dtype)
    num_actual_points = min(L, history_size)
    
    if num_actual_points > 0:
        current_window[history_size - num_actual_points:] = \
            points_history[L - num_actual_points:]
    
    X_window_reshaped = current_window.reshape(history_size, num_features)
    X_input_scaled = scaler_x.transform(X_window_reshaped).flatten().reshape(1, -1)
    
    # Ẩn dòng có tính variance để tăng tốc độ dự đoán (chỉ cần mean)
    # mean_scaled, var_scaled = model.predict_y(X_input_scaled)
    
    mean_scaled, _ = model.predict_f(X_input_scaled)
    predicted_point = scaler_y.inverse_transform(mean_scaled.numpy()).flatten()
    
    return predicted_point


def stable_softmax(log_scores_dict, tau=1.0):
    scores = np.array(list(log_scores_dict.values()))
    keys = list(log_scores_dict.keys())
    scores = scores / tau
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    probs = exp_scores / np.sum(exp_scores)
    return dict(zip(keys, probs))

def compute_goal_probabilities(points_history, gmm_models, gmm_scaler, 
                               window_size, tau=1.0):
    L = len(points_history)
    if L == 0:
        n_targets = len(gmm_models)
        return {tid: 1.0/n_targets for tid in gmm_models}
    
    points_scaled = gmm_scaler.transform(points_history)
    start_idx = max(0, L - window_size)
    window_points = points_scaled[start_idx:L]
    
    rolling_logs = {}
    for tid, model in gmm_models.items():
        rolling_logs[tid] = np.sum(model.score_samples(window_points))
    
    probs = stable_softmax(rolling_logs, tau=tau)
    return probs


def plot_simulation_results(test_data, results, trajectory_name, true_target, save_dir=None):
    """Vẽ 4 plot: Probability, Per-Axis, Interaction Forces và Velocity."""
    modes = list(results['modes'])
    probs = results['probabilities']
    robot_traj = np.asarray(results['robot_trajectory'])
    test_data = np.asarray(test_data)
    L = min(test_data.shape[0], len(robot_traj), len(modes))
    test_data = test_data[:L]
    robot_traj = robot_traj[:L]
    modes = modes[:L]
    probs = {tid: list(prob_list)[:L] for tid, prob_list in probs.items()}
    time_steps = np.arange(L) * DT
    mode_patches_common = control_mode_legend_patches()

    # Plot 1: Goal Probabilities (nền FOLLOWER / SHARED / LEADER)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    add_control_mode_background(ax1, modes, time_steps, zorder=0)

    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    for tid, prob_list in probs.items():
        ax1.plot(time_steps, prob_list, linewidth=1.5, zorder=2,
                color=colors.get(tid, None), label=f'Target {tid}')

    ax1.axhline(y=P_HIGH, color='r', linestyle='--', zorder=2,
                linewidth=1, alpha=0.7, label=f'P_HIGH ({P_HIGH})')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Goal Probability Evolution ({trajectory_name})')
    ax1.set_ylim(0, 1.1)
    h1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=h1 + mode_patches_common, labels=lab1 + [p.get_label() for p in mode_patches_common],
               loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=1)
    plt.tight_layout()
    if save_dir:
        fig1.savefig(os.path.join(save_dir, f'prob_{trajectory_name}.png'), dpi=150)
    plt.close('all')

    # Plot 3: Interaction Forces (f_h và f_r)
    f_h_history = results.get('f_h_history', None)
    f_r_history = results.get('f_r_history', None)
    
    if f_h_history is not None:
        f_h_history = np.asarray(f_h_history)
        if f_r_history is not None:
            f_r_history = np.asarray(f_r_history)
        Lf = min(L, len(f_h_history))
        f_h_history = f_h_history[:Lf]
        if f_r_history is not None and len(f_r_history) >= Lf:
            f_r_history = f_r_history[:Lf]
        modes_f = modes[:Lf]
        ts_f = time_steps[:Lf]

        fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axis_labels_fh = ['Force X (N)', 'Force Y (N)', 'Force Z (N)']
        mode_patches_f = mode_patches_common

        for idx, (ax, label) in enumerate(zip(axes3, axis_labels_fh)):
            add_control_mode_background(ax, modes_f, ts_f, zorder=0)
            ax.plot(ts_f, f_h_history[:, idx], 'g-', linewidth=1.5, label='Human Force (f_h)', zorder=2)

            if f_r_history is not None and len(f_r_history) == len(f_h_history):
                ax.plot(ts_f, f_r_history[:, idx], 'm--', linewidth=1.5, alpha=0.8,
                       label='Robot Force (f_r)', zorder=2)

            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, zorder=1)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

            if idx == 0:
                h, lab = ax.get_legend_handles_labels()
                ax.legend(handles=h + mode_patches_f, labels=lab + [p.get_label() for p in mode_patches_f],
                          loc='upper left')
                ax.set_title(f'Interaction Forces: Human vs Robot ({trajectory_name})')
            else:
                ax.legend(loc='upper left')

        axes3[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_dir:
            fig3.savefig(os.path.join(save_dir, f'forces_{trajectory_name}.png'), dpi=150)
        plt.close('all')

    # Plot 4: Velocity Comparison (Ground Truth vs Prediction)
    n = L
    if n > 2:
        gt_vel = np.diff(test_data[:n], axis=0) / DT
        pred_vel = np.diff(robot_traj[:n], axis=0) / DT
        time_vel = np.arange(len(gt_vel)) * DT

        fig4, axes4 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        vel_labels = ['Velocity X (m/s)', 'Velocity Y (m/s)', 'Velocity Z (m/s)']
        mode_patches_v = mode_patches_common
        modes_vel = modes[1:n]
        time_steps_vel = time_vel

        for idx, (ax, label) in enumerate(zip(axes4, vel_labels)):
            add_control_mode_background(ax, modes_vel, time_steps_vel, zorder=0)
            ax.plot(time_vel, gt_vel[:, idx], 'b-', linewidth=1.2, label='Ground Truth', zorder=2)
            ax.plot(time_vel, pred_vel[:, idx], 'k--', linewidth=1.2, alpha=0.9,
                    label='Prediction (from x_ref)', zorder=2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, zorder=1)
            if idx == 0:
                hv, lv = ax.get_legend_handles_labels()
                ax.legend(handles=hv + mode_patches_v, labels=lv + [p.get_label() for p in mode_patches_v],
                          loc='upper left')
                ax.set_title(f'Velocity Comparison ({trajectory_name})')
            else:
                ax.legend(loc='upper left')

        axes4[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        if save_dir:
            fig4.savefig(os.path.join(save_dir, f'velocity_{trajectory_name}.png'), dpi=150)
        plt.close('all')

def calculate_and_plot_hri_metrics(f_h_history, f_r_history, dt, trajectory_name, modes=None, save_dir=None, assist_energy_history=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from config import FORCE_DEADZONE
    
    n_steps = len(f_h_history)
    theta_arr = np.zeros(n_steps)
    assist_arr = np.zeros(n_steps)

    norms_h = np.linalg.norm(f_h_history, axis=1)

    # 2. VÒNG LẶP TÍNH TOÁN THEO THỜI GIAN
    for i in range(n_steps):
        f_h = f_h_history[i]
        f_r = f_r_history[i]
        
        norm_h = norms_h[i]
        norm_r = np.linalg.norm(f_r)
        
        # --- Tính Assist Index ---
        if norm_h > FORCE_DEADZONE:
            assist_arr[i] = np.dot(f_r, f_h) / norm_h
        else:
            assist_arr[i] = 0.0

        # --- Tính Theta (theo quy ước bài báo) ---
        if norm_r == 0:
            cos_theta = 0.0   # Robot không lực → trung lập (theta = 90°)
        elif norm_h < FORCE_DEADZONE:
            cos_theta = 1.0   # Người thả lỏng → không xung đột (theta = 0°)
        else:
            cos_theta = np.clip(np.dot(f_h, f_r) / (norm_h * norm_r), -1.0, 1.0)

        theta_arr[i] = np.arccos(cos_theta) * (180.0 / np.pi)

    # 3. LỌC NHIỄU (LOW-PASS FILTER)
    SIGMA_SMOOTH = 3.0
    theta_smoothed = gaussian_filter1d(theta_arr, sigma=SIGMA_SMOOTH)
    assist_smoothed = gaussian_filter1d(assist_arr, sigma=SIGMA_SMOOTH)

    # 4. TÍNH GIÁ TRỊ TRUNG BÌNH (INTEGRAL MEAN)
    time_array = np.arange(n_steps) * dt
    T_total = time_array[-1] if n_steps > 0 else 0
    
    mean_theta = np.trapz(theta_smoothed, time_array) / T_total if T_total > 0 else 0
    mean_assist = np.trapz(assist_smoothed, time_array) / T_total if T_total > 0 else 0
    
    print(f"\n   [HRI METRICS - {trajectory_name}]")
    print(f"   Mean Theta:        {mean_theta:.2f} degrees")
    print(f"   Mean Assist Index: {mean_assist:.4f} N")

    # Xử lý Assist Energy nếu có
    has_assist_energy = assist_energy_history is not None and len(assist_energy_history) == n_steps
    n_plots = 3 if has_assist_energy else 2
    
    if has_assist_energy:
        ae_smoothed = gaussian_filter1d(assist_energy_history, sigma=SIGMA_SMOOTH)
        mean_ae = np.trapz(ae_smoothed, time_array) / T_total if T_total > 0 else 0
        print(f"   Mean Assist Energy: {mean_ae:.6f}")

    # 5. VẼ ĐỒ THỊ (STYLE BÀI BÁO)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Subplot (a) và (b): background đỏ LEADER như cũ
    mode_patches = [control_mode_legend_patches() for _ in range(n_plots)]
    if modes is not None and len(modes) == n_steps:
        for ax in axes[:2]:
            add_control_mode_background(ax, list(modes), time_array, zorder=0)

    # Đồ thị (a): Theta
    axes[0].plot(time_array, theta_smoothed, '--', color='tab:blue', linewidth=2.5, label='Theta', zorder=2)
    axes[0].axhline(PHI_ANGLE, color='tab:orange', linewidth=2.5, label='Standard', zorder=2)
    axes[0].set_ylabel('Theta [deg]')
    axes[0].set_ylim(0, 180)
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0 + mode_patches[0], labels=l0 + [p.get_label() for p in mode_patches[0]],
                   loc='upper left')
    axes[0].set_title('(a)', y=-0.25)
    
    # Đồ thị (b): Assist Index (f_r · f_h / |f_h|)
    axes[1].plot(time_array, assist_smoothed, '--', color='tab:blue', linewidth=2.5, label='Assist_index', zorder=2)
    axes[1].axhline(0, color='tab:orange', linewidth=2.5, label='Standard', zorder=2)
    axes[1].set_ylabel('Assist_index')
    
    # Auto-scale trục Y cho Assist Index (bỏ qua 1 giây đầu để tránh gai nhọn)
    valid_idx = int(1.0 / dt) if len(assist_smoothed) > int(1.0/dt) else 0
    if len(assist_smoothed[valid_idx:]) > 0:
        y_max = max(np.max(np.abs(assist_smoothed[valid_idx:])) * 1.5, 2.0)
    else:
        y_max = 2.0
    axes[1].set_ylim(-y_max, y_max)
    
    h1, l1 = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=h1 + mode_patches[1], labels=l1 + [p.get_label() for p in mode_patches[1]],
                   loc='upper left')
    axes[1].set_title('(b)', y=-0.3)

    # Đồ thị (c): Assist Energy = -f_h^T(x_d - x)
    if has_assist_energy:
        from matplotlib.patches import Patch

        # Background xanh nhạt chỉ ở đoạn LEADER có A > 0
        if modes is not None and len(modes) == n_steps:
            start_idx = None
            for i in range(n_steps):
                is_leader_positive = (modes[i] == ControlMode.LEADER and assist_energy_history[i] > 0)
                if is_leader_positive:
                    if start_idx is None:
                        start_idx = i
                else:
                    if start_idx is not None:
                        axes[2].axvspan(time_array[start_idx], time_array[max(0, i - 1)],
                                        color='blue', alpha=0.15, zorder=0)
                        start_idx = None
            if start_idx is not None:
                axes[2].axvspan(time_array[start_idx], time_array[-1],
                                color='blue', alpha=0.15, zorder=0)

        axes[2].plot(time_array, ae_smoothed, '-', color='tab:green', linewidth=1.5, label=r'$A = -f_h^T(x_d - x)$', zorder=2)
        axes[2].axhline(0, color='tab:orange', linewidth=1.5, linestyle='--', label='Zero line', zorder=1)
        axes[2].set_ylabel('Assist Energy')
        axes[2].set_xlabel('Time (s)')
        
        # Auto-scale
        if len(ae_smoothed[valid_idx:]) > 0:
            ae_max = max(np.max(np.abs(ae_smoothed[valid_idx:])) * 1.5, 0.01)
        else:
            ae_max = 0.01
        axes[2].set_ylim(-ae_max, ae_max)
        
        # Fill dưới đường cong: xanh lá khi A < 0 (giúp), xanh dương khi A > 0 (cản)
        axes[2].fill_between(time_array, ae_smoothed, 0,
                             where=np.array(ae_smoothed) < 0,
                             alpha=0.2, color='green', label='Ho tro (A<0)')
        axes[2].fill_between(time_array, ae_smoothed, 0,
                             where=np.array(ae_smoothed) > 0,
                             alpha=0.2, color='blue', label='Can tro (A>0)')
        
        # Legend riêng cho subplot (c)
        leader_bg_patch = Patch(color='blue', alpha=0.15, label='LEADER & A>0')
        h2, l2 = axes[2].get_legend_handles_labels()
        axes[2].legend(handles=h2 + [leader_bg_patch],
                       labels=l2 + [leader_bg_patch.get_label()],
                       loc='upper left', fontsize=8)
        axes[2].set_title('(c)', y=-0.3)
    
    # Format viền và tick dày dặn
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)
        
    axes[0].set_title(f'HRI Metrics ({trajectory_name})')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'hri_metrics_{trajectory_name}.png'), dpi=150)
    plt.close('all')
    
    return mean_theta, mean_assist

def plot_phi(phi_history, trajectory_name, save_dir=None):
    from config import PHI_ANGLE
    t = np.arange(len(phi_history)) * DT
    
    phi_threshold = np.cos(np.radians(PHI_ANGLE))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, phi_history, 'k-', linewidth=1.0, label='Φ')
    ax.axhline(phi_threshold, color='r', linestyle='--', linewidth=1.2, label=f'Φ={phi_threshold:.2f} (conflict, {PHI_ANGLE}°)')
    ax.fill_between(t, phi_history, phi_threshold,
                    where=np.array(phi_history) < phi_threshold,
                    alpha=0.3, color='red', label='Conflict region')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Φ')
    ax.set_title(f'Disagreement Index ({trajectory_name})')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'phi_{trajectory_name}.png'), dpi=150)
    plt.close('all')

def plot_xd_vs_xr(x_d, x_r, modes, trajectory_name, save_dir=None,
                  label_d='$x_d$ (Desired)', label_r='$x_r$ (Actual)',
                  title_prefix='Trajectory Tracking Comparison', filename_prefix='xd_vs_xr',
                  n_keep=None, scenario_name=None):
    """Vẽ đồ thị so sánh giữa x_d (quỹ đạo đặt) và x_r (quỹ đạo thực hành).

    Tham số bổ sung:
      n_keep        : Nếu truyền vào, cả hai quỹ đạo và modes sẽ được cắt tại n_keep
                      để loại bỏ phần đuôi đứng im sau điểm dừng sớm.
      scenario_name : Tên kịch bản (từ SCENARIO_NAMES) để hiển thị trên title.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ── Cắt phần đuôi thừa nếu được yêu cầu ─────────────────────────────────
    if n_keep is not None:
        x_d    = x_d[:n_keep]
        x_r    = x_r[:n_keep]
        modes  = list(modes)[:n_keep]

    n_d = len(x_d)
    n_r = len(x_r)
    n_modes = len(modes)
    n_full = max(n_d, n_r)

    t_d    = np.arange(n_d)    * DT
    t_r    = np.arange(n_r)    * DT
    t_modes = np.arange(min(n_modes, n_full)) * DT

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    coords = ['X', 'Y', 'Z']
    mode_patches = control_mode_legend_patches()

    for i, ax in enumerate(axes):
        add_control_mode_background(ax, list(modes)[:min(n_modes, n_full)], t_modes, zorder=0)

        ax.plot(t_d, x_d[:n_d, i], 'k--', linewidth=1.5, label=label_d, zorder=2)
        ax.plot(t_r, x_r[:n_r, i], 'r-',  linewidth=1.2, label=label_r, zorder=2)

        if n_d < n_r:
            ax.axvline(x=(n_d - 1) * DT, color='green', linestyle=':', linewidth=1.5,
                       alpha=0.8, zorder=3)

        ax.set_ylabel(f'{coords[i]} (m)')
        ax.grid(True, alpha=0.3)

        if i == 0:
            h, l = ax.get_legend_handles_labels()
            extra_handles = list(mode_patches)
            extra_labels  = [p.get_label() for p in mode_patches]
            if n_d < n_r:
                from matplotlib.lines import Line2D
                goal_line = Line2D([0], [0], color='green', linestyle=':', linewidth=1.5,
                                   label=f'MJM Goal Reached ({(n_d-1)*DT:.1f}s)')
                extra_handles.append(goal_line)
                extra_labels.append(goal_line.get_label())
            ax.legend(handles=h + extra_handles, labels=l + extra_labels,
                      loc='upper left', ncol=2)
            # ── Title: thêm scenario_name nếu có ─────────────────────────────
            if scenario_name:
                ax.set_title(f'{title_prefix}\n{trajectory_name}  |  {scenario_name}')
            else:
                ax.set_title(f'{title_prefix} ({trajectory_name})')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{filename_prefix}_{trajectory_name}.png'), dpi=150)
    plt.close('all')

