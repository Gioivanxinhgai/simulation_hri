"""
Script tim tham so a, b cho Fitts' Law tu du lieu HMFR.

FITTS' LAW: t_f = a + b * log2(2A/W)

Trong do:
- t_f: Thoi gian di chuyen CON LAI tu transition point den X_f (seconds)
- A: Amplitude = khoang cach CUC BO tu transition point den X_f (dich)
- W: Target Width = do rong muc tieu (trong thi nghiem)
- a, b: tham so can tim qua hoi quy tuyen tinh
- ID = log2(2A/W): Index of Difficulty (Fitts)

LUU Y: Dung LOCAL distance (transition -> goal) de CONSISTENT voi python_matlab_bridge.py
       noi fitts_law_duration(x_current, goal) dung khoang cach tu vi tri HIEN TAI den goal.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_FOLDER = "Experiment_Train_Trajectory_HRI"
TRAIN_FILE_LIST = os.path.join(TRAIN_FOLDER, "train_file_list.json")
GMM_MODEL_PATH = "gmm_model_hri"

# FITTS' LAW PARAMETERS
W = 0.08  # Target Width (m) - do rong muc tieu trong thi nghiem

# Thresholds
P_HIGH = 0.80
GMM_WINDOW_SIZE = 5
TAU_SOFTMAX = 8

TOTAL_TIME = 12.0
DT = 12.0 / 191

# Toa do Goals
GOALS = {
    1: np.array([0.3796, 0.9779, 0.0169]),
    2: np.array([0.1703, 1.2863, 0.0238]),
    3: np.array([-0.0784, 1.0867, 0.0256]),
}

SCENARIO_TARGET_MAP = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 3,
    7: 2, 8: 2, 9: 1, 10: 2, 11: 1, 12: 2,
    13: 3, 14: 3, 15: 1, 16: 3, 17: 2, 18: 3
}

# ============================================================================
# FITTS' LAW FUNCTIONS
# ============================================================================

def fitts_index_of_difficulty(A, W):
    """
    Tinh Index of Difficulty (ID) theo Fitts' Law.
    ID = log2(2A/W)
    
    Parameters:
    - A: Amplitude (khoang cach den dich)
    - W: Target Width (do rong muc tieu)
    """
    if A <= 0:
        return 0.0
    return np.log2(2 * A / W)

def fitts_movement_time(a, b, A, W):
    """
    Tinh thoi gian di chuyen theo Fitts' Law.
    t_f = a + b * log2(2A/W)
    """
    ID = fitts_index_of_difficulty(A, W)
    return a + b * ID

# ============================================================================
# GMM FUNCTIONS (same as before)
# ============================================================================

def load_gmm_models():
    models = {}
    scaler = None
    
    models_path = os.path.join(GMM_MODEL_PATH, "gmm_models.pkl")
    if os.path.exists(models_path):
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        print(f"   [OK] Loaded GMM models: {list(models.keys())}")
    
    scaler_path = os.path.join(GMM_MODEL_PATH, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   [OK] Loaded scaler")
    
    return models, scaler

def softmax(log_scores, tau=1.0):
    scores = np.array(list(log_scores.values())) / tau
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    probs = exp_scores / np.sum(exp_scores)
    return dict(zip(log_scores.keys(), probs))

def compute_gmm_probabilities(points, gmm_models, scaler, tau=TAU_SOFTMAX):
    if len(points) == 0:
        return {1: 1/3, 2: 1/3, 3: 1/3}
    
    log_scores = {1: 0.0, 2: 0.0, 3: 0.0}
    
    for point in points:
        point_arr = np.array(point).reshape(1, -1)
        if scaler is not None:
            point_arr = scaler.transform(point_arr)
        for target_id, model in gmm_models.items():
            log_scores[target_id] += model.score(point_arr)
    
    return softmax(log_scores, tau)

def find_leader_transition(trajectory_df, gmm_models, scaler, true_target):
    n_frames = len(trajectory_df)
    
    for t in range(GMM_WINDOW_SIZE, n_frames):
        window_start = max(0, t - GMM_WINDOW_SIZE)
        window_data = trajectory_df.iloc[window_start:t+1]
        
        points = [[row['X'], row['Y'], row['Z']] for _, row in window_data.iterrows()]
        probs = compute_gmm_probabilities(points, gmm_models, scaler)
        
        max_prob = max(probs.values())
        max_goal = max(probs, key=probs.get)
        
        if max_prob > P_HIGH:
            return {
                'frame': t,
                'prob': max_prob,
                'predicted_goal': max_goal,
                'true_goal': true_target,
                'position': np.array([
                    trajectory_df.iloc[t]['X'],
                    trajectory_df.iloc[t]['Y'],
                    trajectory_df.iloc[t]['Z']
                ]),
                'time_remaining': (n_frames - t) * DT
            }
    
    return None

# ============================================================================
# ANALYZE
# ============================================================================

def analyze_leader_transitions():
    print("[*] Loading GMM models...")
    gmm_models, scaler = load_gmm_models()
    
    if not gmm_models:
        print("[!] ERROR: Could not load GMM models")
        return None
    
    print(f"   [OK] Loaded {len(gmm_models)} GMM models")
    
    with open(TRAIN_FILE_LIST, 'r') as f:
        train_files = json.load(f)
    
    print(f"[*] Analyzing {len(train_files)} training files...")
    
    transitions = []
    no_transition_count = 0
    
    for i, filename in enumerate(train_files):
        filepath = os.path.join(TRAIN_FOLDER, filename)
        
        if not os.path.exists(filepath):
            continue
        
        df = pd.read_csv(filepath)
        scenario_id = df['ScenarioId'].iloc[0]
        true_target = SCENARIO_TARGET_MAP.get(scenario_id)
        
        if true_target is None:
            continue
        
        trans = find_leader_transition(df, gmm_models, scaler, true_target)
        
        if trans is not None:
            # Lay diem bat dau quy dao (X_0) va goal
            x_0 = np.array([df.iloc[0]['X'], df.iloc[0]['Y'], df.iloc[0]['Z']])
            goal_pos = GOALS[trans['predicted_goal']]
            
            # === LOCAL Distance cho Fitts' Law ===
            # A = khoang cach tu transition position den X_f (goal)
            # CONSISTENT voi python_matlab_bridge.py: fitts_law_duration(x_current, goal)
            A_global = np.linalg.norm(goal_pos - x_0)    # Global (for reference)
            A_local = np.linalg.norm(goal_pos - trans['position'])  # Local Amplitude
            
            # Tinh Index of Difficulty voi LOCAL distance
            ID = fitts_index_of_difficulty(A_local, W)
            
            trans['x_0'] = x_0
            trans['A_global'] = A_global  # Global Amplitude (X_0 -> goal, reference)
            trans['A_local'] = A_local    # Local Amplitude (transition -> goal)
            trans['A'] = A_local          # Use LOCAL for regression
            trans['W'] = W
            trans['ID'] = ID
            trans['t_total'] = TOTAL_TIME  # Tong thoi gian quy dao
            trans['filename'] = filename
            transitions.append(trans)
        else:
            no_transition_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(train_files)} files...")
    
    print(f"\n[*] Results:")
    print(f"   Transitions found: {len(transitions)}")
    print(f"   No transition (P never > {P_HIGH}): {no_transition_count}")
    
    return pd.DataFrame(transitions)

def perform_fitts_regression(df):
    """
    Hoi quy Fitts' Law: t_f = a + b * ID
    Trong do ID = log2(2A/W)
    """
    X = df['ID'].values.reshape(-1, 1)
    y = df['time_remaining'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    a = model.intercept_
    b = model.coef_[0]
    r2 = model.score(X, y)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    return {
        'a': a,
        'b': b,
        'r2': r2,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'W': W
    }

def plot_fitts_results(df, params):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitts' Law Regression
    ax1 = axes[0]
    correct = df[df['predicted_goal'] == df['true_goal']]
    incorrect = df[df['predicted_goal'] != df['true_goal']]
    
    ax1.scatter(correct['ID'], correct['time_remaining'], 
                c='green', alpha=0.6, s=50, label=f'Correct ({len(correct)})')
    ax1.scatter(incorrect['ID'], incorrect['time_remaining'], 
                c='red', alpha=0.6, s=50, label=f'Wrong ({len(incorrect)})')
    
    # Regression line
    id_range = np.linspace(df['ID'].min(), df['ID'].max(), 100)
    t_pred = params['a'] + params['b'] * id_range
    ax1.plot(id_range, t_pred, 'k-', linewidth=2, 
             label=f"t_f = {params['a']:.3f} + {params['b']:.3f} * ID")
    
    ax1.set_xlabel('Index of Difficulty (ID = log2(2A/W))', fontsize=12)
    ax1.set_ylabel('Movement Time t_f (s)', fontsize=12)
    ax1.set_title(f"Fitts' Law Regression\nR^2 = {params['r2']:.4f}, W = {W}m", fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Amplitude vs Time
    ax2 = axes[1]
    scatter = ax2.scatter(df['A'], df['time_remaining'], 
                          c=df['ID'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax2, label='ID')
    
    ax2.set_xlabel('Amplitude A (m)', fontsize=12)
    ax2.set_ylabel('Movement Time t_f (s)', fontsize=12)
    ax2.set_title('Amplitude vs Movement Time\n(Color = ID)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitts_law_regression.png', dpi=150)
    plt.show()
    
    print("\n[*] Plot saved: fitts_law_regression.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  FITTS' LAW PARAMETER REGRESSION")
    print("  Model: t_f = a + b * log2(2A/W)")
    print("="*70)
    print(f"\n  Target Width W = {W} m")
    
    # 1. Analyze
    df = analyze_leader_transitions()
    
    if df is None or len(df) == 0:
        print("[!] ERROR: No transitions found!")
        return None, None
    
    # 2. Statistics
    print("\n" + "="*70)
    print("  STATISTICS")
    print("="*70)
    
    print(f"\n  Amplitude (A = distance to goal):")
    print(f"     Mean: {df['A'].mean():.4f} m")
    print(f"     Std:  {df['A'].std():.4f} m")
    print(f"     Min:  {df['A'].min():.4f} m")
    print(f"     Max:  {df['A'].max():.4f} m")
    
    print(f"\n  Index of Difficulty (ID = log2(2A/W)):")
    print(f"     Mean: {df['ID'].mean():.2f} bits")
    print(f"     Std:  {df['ID'].std():.2f} bits")
    print(f"     Min:  {df['ID'].min():.2f} bits")
    print(f"     Max:  {df['ID'].max():.2f} bits")
    
    print(f"\n  Movement Time (t_f):")
    print(f"     Mean: {df['time_remaining'].mean():.2f} s")
    print(f"     Std:  {df['time_remaining'].std():.2f} s")
    
    accuracy = (df['predicted_goal'] == df['true_goal']).mean() * 100
    print(f"\n  Goal prediction accuracy: {accuracy:.1f}%")
    
    # 3. Fitts' Law Regression
    print("\n" + "="*70)
    print("  FITTS' LAW REGRESSION")
    print("  t_f = a + b * log2(2A/W)")
    print("="*70)
    
    params = perform_fitts_regression(df)
    
    print(f"\n  PARAMETERS:")
    print(f"     a = {params['a']:.4f} seconds")
    print(f"     b = {params['b']:.4f} seconds/bit")
    print(f"     W = {params['W']:.4f} m (fixed)")
    
    print(f"\n  GOODNESS OF FIT:")
    print(f"     R^2 = {params['r2']:.4f}")
    print(f"     p-value = {params['p_value']:.2e}")
    print(f"     Std error = {params['std_err']:.4f}")
    
    print("\n" + "="*70)
    print("  FINAL MODEL")
    print("="*70)
    print(f"\n  t_f = {params['a']:.4f} + {params['b']:.4f} * log2(2A / {W})")
    
    print(f"\n  Example calculations (W = {W}m):")
    for A in [0.05, 0.1, 0.2, 0.5, 1.0]:
        ID = fitts_index_of_difficulty(A, W)
        t = params['a'] + params['b'] * ID
        print(f"     A = {A:.2f}m, ID = {ID:.2f} bits -> t_f = {t:.2f}s")
    
    # 4. Plot
    print("\n[*] Generating plots...")
    plot_fitts_results(df, params)
    
    return df, params

if __name__ == "__main__":
    df, params = main()
