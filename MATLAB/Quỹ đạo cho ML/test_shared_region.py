import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings

warnings.filterwarnings("ignore")

# --- Paths ---
GMM_CHECKPOINT_DIR  = "gmm_model_hri"
GMM_MODEL_PATH      = os.path.join(GMM_CHECKPOINT_DIR, "gmm_models.pkl")
GMM_SCALER_PATH     = os.path.join(GMM_CHECKPOINT_DIR, "scaler.pkl")
SVGP_CHECKPOINT_DIR = "svgp_hri_m52"
TEST_FOLDER = "Experiment_Test_Trajectory_HRI"

# --- Parameters ---
P_LOW = 0.6
P_HIGH = 0.80
HYSTERESIS_DELTA = 0.1  # Dải trễ chống chattering
DT = 12 / 189
SIGMA = 1.0
SVGP_HISTORY_SIZE = 10
GMM_WINDOW_SIZE = 5
TAU_SOFTMAX = 8

# THÊM HẰNG SỐ LỰC (Độ cứng cánh tay người)
K_E = 50.0

# --- Fitts' Law: t_f = a + b * log2(2D/W) ---
FITTS_A = 2.186582
FITTS_B = 1.476623
FITTS_W = 0.08

# --- Goals ---
GOALS = {
    1: np.array([0.3796, 0.9779, 0.0169]),
    2: np.array([0.1703, 1.2863, 0.0238]),
    3: np.array([-0.0784, 1.0867, 0.0256]),
}

GROUND_TRUTH = {
    1: 1, 4: 1, 9: 1, 11: 1, 15: 1, 17: 1,
    2: 2, 5: 2, 7: 2, 12: 2, 13: 2, 18: 2,
    3: 3, 6: 3, 8: 3, 10: 3, 14: 3, 16: 3
}

class ControlMode:
    FOLLOWER = "FOLLOWER"
    LEADER = "LEADER"

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
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return models, scaler

def load_svgp_system(checkpoint_dir):
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
    try:
        if not os.path.exists(filename):
            return None, None
        df = pd.read_csv(filename)
        
        required_cols = {'X', 'Y', 'Z', 'ScenarioId'}
        if not required_cols.issubset(df.columns):
            return None, None
        
        df = df.dropna()
        scenario_id = int(df['ScenarioId'].iloc[0])
        
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
        
        return df, scenario_id
    except Exception as e:
        print(f"  [!] Error reading file: {e}")
        return None, None


# === MINIMUM JERK TRAJECTORY ===

def dynamic_minimum_jerk_trajectory(x_0, v_0, x_f, t_total, dt):
    """
    Tạo quỹ đạo Minimum Jerk (Quintic Spline) đảm bảo tính liên tục C1.
    Tích hợp cứng điều kiện biên v_f = 0 và a_f = 0 tại thời điểm đích.
    """
    t_total = max(t_total, dt * 5) # Đảm bảo T không quá nhỏ
    n_steps = int(t_total / dt) + 1
    t_array = np.linspace(0, t_total, n_steps)
    t = t_array[:, np.newaxis]

    T = t_total
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    c0 = x_0
    c1 = v_0
    c2 = np.zeros_like(x_0)
    c3 = 10 * (x_f - x_0) / T3 - 6 * v_0 / T2
    c4 = -15 * (x_f - x_0) / T4 + 8 * v_0 / T3
    c5 = 6 * (x_f - x_0) / T5 - 3 * v_0 / T4

    trajectory = c0 + c1*t + c2*(t**2) + c3*(t**3) + c4*(t**4) + c5*(t**5)
    return trajectory

def fitts_law_duration(x_current, x_goal, a=FITTS_A, b=FITTS_B, w=FITTS_W):
    D = np.linalg.norm(x_current - x_goal)
    if D < w / 2:
        return DT * 5
    t_f = a + b * np.log2(2 * D / w)
    return max(t_f, DT * 5)


# === SVGP PREDICTION ===

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
    
    mean_scaled, _ = model.predict_y(X_input_scaled)
    predicted_point = scaler_y.inverse_transform(mean_scaled.numpy()).flatten()
    
    return predicted_point


# === GMM PROBABILITY ===

def stable_softmax(log_scores_dict, tau=1.0):
    scores = np.array(list(log_scores_dict.values()))
    keys = list(log_scores_dict.keys())
    scores = scores / tau
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    probs = exp_scores / np.sum(exp_scores)
    return dict(zip(keys, probs))

def linear_saturation(prob, p_low, p_high):
    if prob <= p_low:
        return 0.0
    elif prob >= p_high:
        return 1.0
    else:
        return (prob - p_low) / (p_high - p_low)

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


# === MAIN CONTROL LOOP ===

def run_control_loop(test_data, gmm_models, gmm_scaler, 
                     svgp_model, svgp_scaler_x, svgp_scaler_y,
                     true_target_id):
    """Chạy vòng lặp điều khiển mô phỏng Offline."""
    n_steps = len(test_data)
    current_mode = ControlMode.FOLLOWER
    
    # Storage
    robot_trajectory = []
    modes = []
    probability_history = {tid: [] for tid in gmm_models}
    transitions = []
    svgp_predictions = []
    alpha_history = []
    f_h_history = [] 
    
    # Timing
    timing = {
        'gmm_total': 0.0, 'svgp_total': 0.0, 'mjm_total': 0.0,
        'gmm_count': 0, 'svgp_count': 0, 'mjm_count': 0
    }
    
    # State Variables cho Array MJM và Trigger
    current_goal_id = None
    mjm_trajectory = None
    mjm_step_index = 0
    trajectory_start_pos = test_data[0].copy()
    prev_mode = ControlMode.FOLLOWER
    
    print(f"\n{'='*70}")
    print(f"  CONTROL LOOP SIMULATION (OFFLINE - PHASE-TRIGGERED MJM)")
    print(f"  Total timesteps: {n_steps} | P_LOW: {P_LOW} | P_HIGH: {P_HIGH}")
    print(f"{'='*70}\n")
    
    for t in range(n_steps):
        # Ước lượng và LỌC vận tốc hiện tại của robot (v_current)
        if len(robot_trajectory) >= 5:
            # Lấy trung bình 4 frames để khử nhiễu
            v_current = (robot_trajectory[-1] - robot_trajectory[-5]) / (4 * DT)
        elif len(robot_trajectory) >= 2:
            v_current = (robot_trajectory[-1] - robot_trajectory[-2]) / DT
        else:
            v_current = np.zeros(3)
            
        # KẸP (Clamp) tốc độ tối đa (1.0 m/s) để tránh nổ MJM
        v_norm = np.linalg.norm(v_current)
        if v_norm > 1.0: 
            v_current = (v_current / v_norm) * 1.0
            
        # Tính toán lực f_h
        true_goal_pos = GOALS[true_target_id]
        f_h = -K_E * (test_data[t] - true_goal_pos)
        f_h_history.append(f_h)
        
        # Xử lý t=0
        if t == 0:
            x_ref = test_data[0].copy()
            robot_trajectory.append(x_ref)
            modes.append(ControlMode.FOLLOWER)
            alpha_history.append(0.0)
            svgp_predictions.append(x_ref.copy())
            for tid in gmm_models:
                probability_history[tid].append(1.0 / len(gmm_models))
            prev_mode = ControlMode.FOLLOWER
            continue
        
        # BƯỚC 1: GMM Classification
        points_history = test_data[0:t+1]
        t_gmm_start = time.perf_counter()
        
        probs = compute_goal_probabilities(
            points_history, gmm_models, gmm_scaler,
            window_size=GMM_WINDOW_SIZE, tau=TAU_SOFTMAX
        )
        timing['gmm_total'] += time.perf_counter() - t_gmm_start
        timing['gmm_count'] += 1
        
        for tid in gmm_models:
            probability_history[tid].append(probs[tid])
        
        max_prob_goal = max(probs, key=probs.get)
        max_prob = probs[max_prob_goal]
        
        # BƯỚC 2: STATE MACHINE (với Hysteresis)
        x_svgp = None
        alpha = 0.0
        
        if current_mode == ControlMode.FOLLOWER:
            threshold_low = P_LOW          
            threshold_high = P_HIGH        
        elif current_mode == ControlMode.LEADER:
            threshold_low = P_LOW - HYSTERESIS_DELTA   
            threshold_high = P_HIGH - HYSTERESIS_DELTA 
        else:  
            threshold_low = P_LOW - HYSTERESIS_DELTA   
            threshold_high = P_HIGH                    
        
        if max_prob < threshold_low:
            # ── VÙNG 1: PURE FOLLOWER ──
            current_mode = ControlMode.FOLLOWER
            alpha = 0.0
            
            t_svgp_start = time.perf_counter()
            x_svgp = predict_next_point_svgp(
                points_history, svgp_scaler_x, svgp_scaler_y,
                svgp_model, history_size=SVGP_HISTORY_SIZE
            )
            timing['svgp_total'] += time.perf_counter() - t_svgp_start
            timing['svgp_count'] += 1
            
            x_ref = x_svgp.copy()
            current_goal_id = None
            mjm_trajectory = None
            mjm_step_index = 0
            
        else:
            # ── VÙNG 2 & 3: LEADER hoặc SHARED ──
            if max_prob > threshold_high:
                current_mode = ControlMode.LEADER
                alpha = 1.0
            else:
                current_mode = "LEADER"
                # Thử nghiệm Ablation Study của cậu
                alpha = linear_saturation(max_prob, P_LOW, P_HIGH) 

            goal_changed     = (max_prob_goal != current_goal_id)
            exited_follower  = (prev_mode == ControlMode.FOLLOWER)
            shared_to_leader = (prev_mode == "SHARED" and current_mode == ControlMode.LEADER)

            if goal_changed or exited_follower or shared_to_leader:
                current_goal_id = max_prob_goal
                trajectory_start_pos = robot_trajectory[-1].copy()
                goal_pos = GOALS[current_goal_id]
                
                t_fitts = fitts_law_duration(trajectory_start_pos, goal_pos)
                t_mjm_start = time.perf_counter()
                
                mjm_trajectory = dynamic_minimum_jerk_trajectory(
                    x_0=trajectory_start_pos, 
                    v_0=v_current,       # Đã được lọc và kẹp an toàn
                    x_f=goal_pos, 
                    t_total=t_fitts, 
                    dt=DT
                )
                timing['mjm_total'] += time.perf_counter() - t_mjm_start
                timing['mjm_count'] += 1
                
                # ── [SỬA LỖI 2]: LUÔN BẮT ĐẦU TỪ INDEX 1 (Tránh đứng khựng 1 frame) ──
                mjm_step_index = 1

            if mjm_trajectory is not None and mjm_step_index < len(mjm_trajectory):
                current_mjm_point = mjm_trajectory[mjm_step_index]
                mjm_step_index += 1
            else:
                current_mjm_point = GOALS[current_goal_id]

            if current_mode == ControlMode.LEADER:
                x_ref = current_mjm_point.copy()
            else:
                t_svgp_start = time.perf_counter()
                x_svgp = predict_next_point_svgp(
                    points_history, svgp_scaler_x, svgp_scaler_y,
                    svgp_model, history_size=SVGP_HISTORY_SIZE
                )
                timing['svgp_total'] += time.perf_counter() - t_svgp_start
                timing['svgp_count'] += 1
                
                x_ref = (1.0 - alpha) * x_svgp + alpha * current_mjm_point
        
        # Lưu trữ
        robot_trajectory.append(x_ref.copy())
        modes.append(current_mode)
        alpha_history.append(alpha)
        
        if x_svgp is not None:
            svgp_predictions.append(x_svgp.copy())
        else:
            svgp_predictions.append(np.array([np.nan, np.nan, np.nan]))
            
        prev_mode = current_mode
    
    # Timing summary
    print(f"\n[OK] Simulation completed. Total transitions: {len(transitions)}")
    print(f"\n--- TIMING SUMMARY ---")
    print(f"GMM Classification: {timing['gmm_total']:.2f} s ({timing['gmm_count']} calls)")
    print(f"SVGP Prediction:    {timing['svgp_total']:.2f} s ({timing['svgp_count']} calls)")
    print(f"MJM Generation:     {timing['mjm_total']:.2f} s ({timing['mjm_count']} calls)")
    
    return {
        'robot_trajectory': np.array(robot_trajectory),
        'modes': modes,
        'probabilities': probability_history,
        'transitions': transitions,
        'svgp_predictions': np.array(svgp_predictions),
        'alpha_history': np.array(alpha_history),
        'f_h_history': np.array(f_h_history),
        'timing': timing
    }

# === VISUALIZATION ===

def plot_simulation_results(test_data, results, scenario_id, true_target):
    modes = results['modes']
    probs = results['probabilities']
    time_steps = np.arange(len(modes)) / (191/12)
    
    # Plot 1: Goal Probabilities
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    for tid, prob_list in probs.items():
        ax1.plot(time_steps, prob_list, linewidth=1.5, color=colors.get(tid, None), label=f'Target {tid}')
    
    ax1.axhline(y=P_HIGH, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'P_HIGH ({P_HIGH})')
    ax1.axhline(y=P_LOW, color='purple', linestyle='--', linewidth=1, alpha=0.7, label=f'P_LOW ({P_LOW})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Goal Probability Evolution (ScenarioId {scenario_id})')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Per-Axis (X, Y, Z)
    fig2, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    robot_traj = results['robot_trajectory']
    axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
    
    def get_mode_color(mode):
        if mode == ControlMode.FOLLOWER: return 'green'
        elif mode == ControlMode.LEADER: return 'red'
        else: return 'orange'
    
    for idx, (ax, label) in enumerate(zip(axes, axis_labels)):
        ax.plot(time_steps, test_data[:, idx], 'b-', linewidth=0.8, label='Ground Truth')
        i = 0
        legend_added = {'FOLLOWER': False, 'SHARED': False, 'LEADER': False}
        
        while i < len(modes):
            current_mode = modes[i]
            start_idx = i
            while i < len(modes) and modes[i] == current_mode:
                i += 1
            end_idx = i
            
            color = get_mode_color(current_mode)
            if current_mode == ControlMode.FOLLOWER: mode_name = 'FOLLOWER'
            elif current_mode == ControlMode.LEADER: mode_name = 'LEADER'
            else: mode_name = 'SHARED'
            
            lbl = f'x_ref ({mode_name})' if not legend_added[mode_name] else None
            legend_added[mode_name] = True
            seg_end = min(end_idx + 1, len(robot_traj))
            ax.plot(time_steps[start_idx:seg_end], robot_traj[start_idx:seg_end, idx], 
                    color=color, linestyle='--', linewidth=1.5, alpha=1, label=lbl)
        
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best')
            ax.set_title(f'Per-Axis Trajectory Comparison (ScenarioId {scenario_id})')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    # Plot 3: Velocity Comparison
    n = min(len(test_data), len(robot_traj))
    if n > 2:
        gt_vel = np.diff(test_data[:n], axis=0) / DT
        pred_vel = np.diff(robot_traj[:n], axis=0) / DT
        time_vel = np.arange(len(gt_vel)) * DT

        fig4, axes4 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        vel_labels = ['Velocity X (m/s)', 'Velocity Y (m/s)', 'Velocity Z (m/s)']

        for idx, (ax, label) in enumerate(zip(axes4, vel_labels)):
            ax.plot(time_vel, gt_vel[:, idx], 'b-', linewidth=1.2, label='Ground Truth')
            ax.plot(time_vel, pred_vel[:, idx], 'r--', linewidth=1.2, alpha=0.85, label='Prediction (x_ref)')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='best')
                ax.set_title(f'Velocity Comparison: Ground Truth vs Prediction (ScenarioId {scenario_id})')
        axes4[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

# === MAIN ===
def load_file_list(json_path):
    import json
    if not os.path.exists(json_path): return []
    with open(json_path, 'r') as f: return json.load(f)

def main():
    print("=" * 70)
    print("  HUMAN-ROBOT SHARED CONTROL SIMULATION")
    print("=" * 70)
    
    print("\n[*] Loading models...")
    gmm_models, gmm_scaler = load_gmm_system(GMM_MODEL_PATH, GMM_SCALER_PATH)
    svgp_scaler_x, svgp_scaler_y, svgp_model = load_svgp_system(SVGP_CHECKPOINT_DIR)
    
    file_list_path = os.path.join(TEST_FOLDER, "test_file_list.json")
    file_names = load_file_list(file_list_path)
    
    if not file_names: return
    
    all_metrics = []
    
    for i, test_file in enumerate(file_names):
        print(f"\n[{i+1}/{len(file_names)}] Processing: {test_file}")
        test_path = os.path.join(TEST_FOLDER, test_file)
        df, scenario_id = read_and_smooth_3d(test_path, SIGMA)
        
        if df is None: continue
        
        test_data = df[['X', 'Y', 'Z']].values
        true_target = GROUND_TRUTH.get(scenario_id, None)
        
        results = run_control_loop(
            test_data, gmm_models, gmm_scaler,
            svgp_model, svgp_scaler_x, svgp_scaler_y,
            true_target_id=true_target 
        )
        
        robot_traj = np.array(results['robot_trajectory'])
        ground_truth = test_data[:len(robot_traj)]
        errors = robot_traj - ground_truth
        
        mse = np.mean(np.sum(errors ** 2, axis=1))
        mae = np.mean(np.sqrt(np.sum(errors ** 2, axis=1)))
        
        print(f"   [EVALUATION] MAE: {mae:.4f} m | MSE: {mse:.6f}")
        plot_simulation_results(test_data, results, scenario_id, true_target)
        
if __name__ == "__main__":
    main()