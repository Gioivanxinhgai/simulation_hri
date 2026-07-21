import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
import time

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# ─── 1. CẤU HÌNH (CONFIGURATION) ────────────────────────────────────────
TEST_FOLDER     = "Experiment_Test_Trajectory_HRI"

# Đường dẫn model GMM cho HRI
CHECKPOINT_DIR  = "gmm_model_hri"
MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "gmm_models.pkl")
SCALER_PATH     = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

SIGMA           = 1.0 
WINDOW_SIZE     = 5  # Kích thước Sliding Window

# Tham số làm mịn Softmax (Temperature)
TAU_SOFTMAX     = 8 # τ: Giá trị khuyến nghị 5 ~ 20.

# Thư mục lưu đồ thị (Plots)
PLOT_SAVE_DIR   = "gmm_plots"

# --- Định nghĩa Ground Truth cho từng Trajectory ID ---
# Target 1: ScenarioId 1, 4, 9, 11, 15, 17
# Target 2: ScenarioId 2, 5, 7, 12, 13, 18
# Target 3: ScenarioId 3, 6, 8, 10, 14, 16
GROUND_TRUTH = {
    1: 1, 4: 1, 9: 1, 11: 1, 15: 1, 17: 1,
    2: 2, 5: 2, 7: 2, 12: 2, 13: 2, 18: 2,
    3: 3, 6: 3, 8: 3, 10: 3, 14: 3, 16: 3
}

# --- Thiết lập Font chữ Times New Roman ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# ─── 2. HELPERS ─────────────────────────────────────────────────────────
def read_and_smooth_3d(filename, sigma=1):
    """Đọc file CSV và làm mịn dữ liệu 3D (X, Y, Z)."""
    try:
        if not os.path.exists(filename): 
            return None, None
        df = pd.read_csv(filename)
        
        # Kiểm tra các cột cần thiết
        required_cols = {'X', 'Y', 'Z', 'ScenarioId'}
        if not required_cols.issubset(df.columns):
            print(f"  ⚠️ File thiếu cột: {required_cols - set(df.columns)}")
            return None, None
        
        df = df.dropna()
        
        # Lấy ScenarioId từ dòng đầu tiên
        scenario_id = int(df['ScenarioId'].iloc[0])
        
        # Làm mịn dữ liệu 3D
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
        
        return df, scenario_id
    except Exception as e:
        print(f"  ❌ Lỗi đọc file: {e}")
        return None, None

def load_system(model_path, scaler_path):
    """Load model GMM và scaler."""
    with open(model_path, 'rb') as f: 
        models = pickle.load(f)
    with open(scaler_path, 'rb') as f: 
        scaler = pickle.load(f)
    return models, scaler

def stable_softmax(log_scores_dict, tau=1.0):
    """Chuyển đổi Log-Likelihood sang Probability (Softmax) với Temperature."""
    scores = np.array(list(log_scores_dict.values()))
    keys = list(log_scores_dict.keys())
    
    # Áp dụng Temperature: scores / tau
    scores = scores / tau
    
    # Softmax ổn định
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    probs = exp_scores / np.sum(exp_scores)
    return dict(zip(keys, probs))

# ─── 3. LOGIC SLIDING WINDOW (3D) ────────────────────────────────────────
def analyze_sliding_window_3d(df, models, scaler, window_size=10, tau=1.0):
    """
    Tính toán xác suất tại mỗi thời điểm t dựa trên cửa sổ trượt với dữ liệu 3D.
    """
    # Sử dụng 3 chiều: X, Y, Z
    points = df[['X', 'Y', 'Z']].values
    points_scaled = scaler.transform(points)
    n_steps = len(points)
    
    # 1. Tính log-likelihood cho từng điểm
    step_log_probs = {}
    for tid, model in models.items():
        step_log_probs[tid] = model.score_samples(points_scaled)
    
    # 2. Tính tổng trong cửa sổ trượt (Rolling Sum)
    rolling_logs = {tid: np.zeros(n_steps) for tid in models}
    
    for tid in models:
        raw_logs = step_log_probs[tid]
        for t in range(n_steps):
            # Lấy cửa sổ K điểm gần nhất
            start_idx = max(0, t - window_size + 1)
            window_sum = np.sum(raw_logs[start_idx : t + 1])
            rolling_logs[tid][t] = window_sum
            
    # 3. Chuyển đổi sang Xác suất theo thời gian
    history_probs = {tid: [] for tid in models}
    
    for t in range(n_steps):
        current_scores = {tid: rolling_logs[tid][t] for tid in models}
        # Áp dụng Softmax + Temperature
        probs = stable_softmax(current_scores, tau=tau)
        
        for tid in models:
            history_probs[tid].append(probs[tid])
            
    # Dự đoán cuối cùng (dựa trên cửa sổ cuối)
    final_scores = {tid: rolling_logs[tid][-1] for tid in models}
    pred_target = max(final_scores, key=final_scores.get)
    
    return pred_target, rolling_logs, history_probs

# ─── 4. VẼ ĐỒ THỊ ──────────────────────────────────────────────────────────
def plot_evolution(df, history_probs, tau, scenario_id, pred_target, true_target, save_path=None):
    """Vẽ đồ thị Probability theo thời gian (Softmax + Tau)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    time_steps = np.arange(len(df)) / 12  # Giả sử 10Hz sampling rate
    
    colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}
    
    for tid, probs in history_probs.items():
        ax.plot(time_steps, probs, linewidth=1.5, 
                label=f'Target {tid}', color=colors.get(tid, None))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Probability")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    ax.grid(True, linestyle='-', alpha=0.5)
    
    # Tiêu đề với thông tin ScenarioId và kết quả
    status = "✓" if pred_target == true_target else "✗"
    ax.set_title(f"ScenarioId {scenario_id} | Pred: Target {pred_target} | True: Target {true_target} [{status}]")
    
    plt.tight_layout()
    # plt.show() # Tắt show để không bị block khi chạy hàng loạt
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

# ─── 5. MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"=" * 80)

    print(f"=" * 80)
    
    # 1. Load System
    if not os.path.exists(MODEL_PATH):
        print("❌ Missing model checkpoints.")
        print(f"   Expected: {MODEL_PATH}")
        exit()
    if not os.path.exists(SCALER_PATH):
        print("❌ Missing scaler file.")
        print(f"   Expected: {SCALER_PATH}")
        exit()
        
    models, scaler = load_system(MODEL_PATH, SCALER_PATH)
    print(f"\n✅ Loaded {len(models)} GMM models: {list(models.keys())}")
    
    # 2. Liệt kê tất cả file trajectory trong thư mục test
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ Test folder not found: {TEST_FOLDER}")
        exit()
    
    # Lọc các file .csv có định dạng trajectory_XXX.csv
    all_files = os.listdir(TEST_FOLDER)
    trajectory_files = sorted([f for f in all_files if f.startswith("trajectory_") and f.endswith(".csv")])
    
    if not trajectory_files:
        print("⚠️ No trajectory files found.")
        exit()
        
    print(f"📂 Found {len(trajectory_files)} trajectory files.\n")
    print(f"{'File Name':<25} | {'ScenarioId':>10} | {'True Target':>11} | {'Prediction':>10} | {'Status':>6} | {'Time (s)':>10}")
    print("-" * 95)
    
    # Tạo thư mục lưu CSV
    PROB_SAVE_DIR = "gmm_prob_logs"
    if not os.path.exists(PROB_SAVE_DIR):
        os.makedirs(PROB_SAVE_DIR)
    print(f"📁 Probability results will be saved to: {PROB_SAVE_DIR}\n")
    
    # Tạo thư mục lưu Đồ thị (Plots)
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)
    print(f"📁 Plot images will be saved to: {PLOT_SAVE_DIR}\n")
    
    # 3. Main Loop
    total_classification_time = 0
    results = []  # Lưu kết quả để tổng hợp
    
    for fname in trajectory_files:
        full_path = os.path.join(TEST_FOLDER, fname)
        df, scenario_id = read_and_smooth_3d(full_path, SIGMA)
        
        if df is not None and scenario_id is not None:
            # Lấy Ground Truth
            true_target = GROUND_TRUTH.get(scenario_id, None)
            
            if true_target is None:
                print(f"{fname:<25} | {scenario_id:>10} | {'Unknown':>11} | {'Skipped':>10} | {'⚠️':>6} | {'-':>10}")
                continue
            
            start_time = time.time()
            
            # Phân tích Sliding Window 3D
            pred_target, rolling_logs, history_probs = analyze_sliding_window_3d(
                df, models, scaler, window_size=WINDOW_SIZE, tau=TAU_SOFTMAX
            )
            
            elapsed_time = time.time() - start_time
            total_classification_time += elapsed_time
            
            # Kiểm tra kết quả
            is_correct = (pred_target == true_target)
            status = "✓" if is_correct else "✗"
            
            print(f"{fname:<25} | {scenario_id:>10} | Target {true_target:>4} | Target {pred_target:>3} | {status:>6} | {elapsed_time:>10.4f}")
            
            results.append({
                'file': fname,
                'scenario_id': scenario_id,
                'true_target': true_target,
                'pred_target': pred_target,
                'correct': is_correct,
                'time': elapsed_time
            })
            
            # Lưu xác suất ra file CSV
            prob_df = pd.DataFrame()
            prob_df['Time(s)'] = np.arange(len(df)) / 12
            for tid in sorted(history_probs.keys()):
                prob_df[f'Prob_Target_{tid}'] = history_probs[tid]
            
            prob_csv_path = os.path.join(PROB_SAVE_DIR, f"prob_{fname}")
            prob_df.to_csv(prob_csv_path, index=False)
            
            # Vẽ đồ thị Probability Evolution
            plot_img_path = os.path.join(PLOT_SAVE_DIR, f"plot_{fname.replace('.csv', '.png')}")
            plot_evolution(df, history_probs, TAU_SOFTMAX, scenario_id, pred_target, true_target, save_path=plot_img_path)
            
        else:
            print(f"{fname:<25} | {'Error':>10} | {'-':>11} | {'-':>10} | {'❌':>6} | {'-':>10}")
    
    # ─── 6. TỔNG KẾT ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    if results:
        total = len(results)
        correct = sum(r['correct'] for r in results)
        accuracy = correct / total * 100
        
        print(f"\n📊 Total trajectories tested: {total}")
        print(f"✅ Correct predictions: {correct}")
        print(f"❌ Incorrect predictions: {total - correct}")
        print(f"🎯 Accuracy: {accuracy:.2f}%")
        print(f"⏱️  Total classification time: {total_classification_time:.4f} seconds")
        print(f"⏱️  Average time per trajectory: {total_classification_time/total:.4f} seconds")
        
        # Thống kê theo từng Target
        print("\n📈 Breakdown by Target:")
        for target_id in sorted(set(GROUND_TRUTH.values())):
            target_results = [r for r in results if r['true_target'] == target_id]
            if target_results:
                target_correct = sum(r['correct'] for r in target_results)
                target_total = len(target_results)
                target_acc = target_correct / target_total * 100
                print(f"   Target {target_id}: {target_correct}/{target_total} ({target_acc:.1f}%)")
        
        # Hiển thị các trường hợp sai
        incorrect = [r for r in results if not r['correct']]
        if incorrect:
            print("\n⚡ Incorrect predictions:")
            for r in incorrect:
                print(f"   - {r['file']}: ScenarioId {r['scenario_id']} | True: Target {r['true_target']} | Pred: Target {r['pred_target']}")
    else:
        print("⚠️ No valid results to summarize.")
    
    print(f"\n✅ Test completed.")
