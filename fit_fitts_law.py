"""
fit_fitts_law.py  (v3 – GMM-aware)
─────────────────────────────────────────────────────────────────────────────
Fit Fitts' Law:  T = a + b * ID,  với  ID = log2(2*D/W)

Cách tiếp cận ĐÚNG với kiến trúc Outer Loop:
  - Với mỗi quỹ đạo training, ta chạy lại logic Role Arbitration (GMM + Sliding
    Window + N_CONFIDENCE_REQUIRED bước tự tin) để tìm đúng thời điểm t_leader
    mà robot sẽ nhận quyền LEADER.
  - Tại thời điểm đó, ta lấy DUY NHẤT MỘT điểm dữ liệu cho mỗi quỹ đạo:
      D  = ||x_f - x(t_leader)||          (khoảng cách còn lại đến đích)
      T  = (n_keep - 1 - t_leader) * DT   (thời gian còn lại đến khi người dừng)
      ID = log2(2 * D / W)
  - Đưa toàn bộ các cặp (ID, T) vào Linear Regression → thu được a, b.

  → Nhất quán với outer_loop.py: Fitts' Law được gọi đúng 1 lần khi vào LEADER
    để ước lượng t_f từ vị trí hiện tại đến đích.

Phụ thuộc: config.py, shared_control_lib.py (compute_goal_probabilities,
            load_gmm_system, read_and_smooth_3d)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

# ── Import tham số và hàm dùng chung ─────────────────────────────────────────
from config import (
    GMM_MODEL_PATH, GMM_SCALER_PATH,
    GMM_WINDOW_SIZE, TAU_SOFTMAX, SIGMA,
    P_HIGH, N_CONFIDENCE_REQUIRED,
    GOALS, GROUND_TRUTH,
    DT,
    FITTS_A, FITTS_B, FITTS_W,
)
from shared_control_lib import (
    load_gmm_system,
    compute_goal_probabilities,
    read_and_smooth_3d,
)

# ── Tham số riêng của script này ─────────────────────────────────────────────
TRAIN_FOLDER = 'Experiment_Train_Trajectory_HRI'  # Thư mục chứa CSV training
TRAIN_LIST   = None   # Để None → dùng glob; hoặc đặt đường dẫn tới JSON list

W            = FITTS_W   # Độ rộng mục tiêu (m) — dùng đúng giá trị config
V_THRESH     = 0.02      # Ngưỡng vận tốc đứng yên (m/s) để cắt đuôi quỹ đạo

# Tham số so sánh (config.py hiện tại)
A_CURRENT = FITTS_A
B_CURRENT = FITTS_B

# ── Load GMM ─────────────────────────────────────────────────────────────────
print(f"Loading GMM from: {GMM_MODEL_PATH}")
gmm_models, gmm_scaler = load_gmm_system(GMM_MODEL_PATH, GMM_SCALER_PATH)
print(f"  → {len(gmm_models)} targets: {list(gmm_models.keys())}")
print(f"  GMM window size: {GMM_WINDOW_SIZE},  P_HIGH: {P_HIGH},  N_confidence: {N_CONFIDENCE_REQUIRED}\n")

# ── Thu thập file training ────────────────────────────────────────────────────
if TRAIN_LIST and os.path.exists(TRAIN_LIST):
    with open(TRAIN_LIST, 'r') as f:
        files = [os.path.join(TRAIN_FOLDER, fn) for fn in json.load(f)]
else:
    files = sorted(glob.glob(os.path.join(TRAIN_FOLDER, 'trajectory_*.csv')))

print(f"Training files found: {len(files)}")
print(f"W = {W} m,  V_THRESH = {V_THRESH*100:.1f} cm/s\n")


# ── Hàm phụ: lấy ground-truth target từ tên file ─────────────────────────────
def get_target_from_filename(filepath):
    """
    Trích ScenarioId từ tên file (dạng trajectory_<id>_*.csv hoặc traj_<id>.csv)
    và tra GROUND_TRUTH để lấy target id (1, 2, 3).
    Trả về (target_id, goal_pos) hoặc (None, None) nếu không tìm được.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Thử tách số đầu tiên sau "trajectory_"
    parts = basename.replace('trajectory_', '').split('_')
    for p in parts:
        try:
            scenario_id = int(p)
            if scenario_id in GROUND_TRUTH:
                tid = GROUND_TRUTH[scenario_id]
                return tid, GOALS[tid]
        except ValueError:
            continue
    return None, None


# ── Vòng lặp chính: tìm thời điểm LEADER cho từng quỹ đạo ───────────────────
ID_list   = []
T_list    = []
info_list = []  # Để debug / in kết quả

skipped_no_csv       = 0
skipped_no_leader    = 0
skipped_short        = 0
skipped_wrong_target = 0  # GMM tự tin sai mục tiêu (hay gặp ở Change scenarios)

for fpath in files:
    # 1. Đọc và làm mịn
    df, traj_name = read_and_smooth_3d(fpath, sigma=SIGMA)
    if df is None or len(df) < GMM_WINDOW_SIZE + N_CONFIDENCE_REQUIRED + 5:
        skipped_short += 1
        continue

    coords = df[['X', 'Y', 'Z']].values  # (N, 3) — đã được làm mịn

    # 2. Lấy ground-truth target (đích đến) từ tên file
    target_id, goal_pos = get_target_from_filename(fpath)

    # 3. Cắt đuôi đứng yên (v < V_THRESH) để tìm điểm dừng thực tế
    vels = np.linalg.norm(np.diff(coords, axis=0), axis=1) / DT
    if np.all(vels < V_THRESH):
        skipped_no_csv += 1
        continue
    last_active = int(np.where(vels >= V_THRESH)[0][-1])
    n_keep = last_active + 2          # điểm cuối cùng hợp lệ + 1
    coords = coords[:n_keep]

    x_f = coords[-1]                  # điểm dừng thực tế = đích

    # Nếu không có GROUND_TRUTH, dùng x_f làm đích
    if goal_pos is None:
        goal_pos = x_f

    # 4. Chạy GMM Sliding Window để tìm t_leader
    #    Logic giống outer_loop.py: cần N_CONFIDENCE_REQUIRED bước liên tiếp
    #    với max_prob >= P_HIGH trên cùng 1 target.
    confidence_count = 0
    confident_target = None
    t_leader = None

    for t in range(GMM_WINDOW_SIZE, n_keep):
        # Lịch sử vị trí người đến bước t
        history = coords[max(0, t - GMM_WINDOW_SIZE):t]  # shape (w, 3)
        probs = compute_goal_probabilities(
            history, gmm_models, gmm_scaler,
            window_size=GMM_WINDOW_SIZE,
            tau=TAU_SOFTMAX,
        )
        best_tid  = max(probs, key=probs.get)
        best_prob = probs[best_tid]

        if best_prob >= P_HIGH:
            if best_tid == confident_target:
                confidence_count += 1
            else:
                confident_target  = best_tid
                confidence_count  = 1
        else:
            confidence_count = 0
            confident_target = None

        if confidence_count >= N_CONFIDENCE_REQUIRED:
            t_leader = t  # Bước đầu tiên robot được phép vào LEADER
            break

    if t_leader is None:
        skipped_no_leader += 1
        continue

    # 5a. Kiểm tra GMM phân loại đúng mục tiêu không
    #     Quan trọng với Change scenarios (7–18): GMM có thể đạt tự tin sớm
    #     nhưng hướng về đích BAN ĐẦU (sai), chứ không phải đích THỰC SỰ.
    #     Fitts' Law chỉ có ý nghĩa vật lý khi D là khoảng cách đến đúng đích
    #     người đang thực sự di chuyển tới.
    if target_id is not None and confident_target != target_id:
        skipped_wrong_target += 1
        continue

    # 5b. Tính khoảng cách và thời gian còn lại từ t_leader đến đích
    D_remaining = np.linalg.norm(goal_pos - coords[t_leader])
    T_remaining = (n_keep - 1 - t_leader) * DT

    # Bỏ qua nếu khoảng cách quá nhỏ (degenerate)
    if D_remaining < W / 2 or T_remaining < DT * 2:
        skipped_no_leader += 1
        continue

    ID = np.log2(2.0 * D_remaining / W)
    ID_list.append(ID)
    T_list.append(T_remaining)
    info_list.append({
        'file':     os.path.basename(fpath),
        'gt_target': target_id,       # Ground-truth target (từ GROUND_TRUTH)
        'gmm_target': confident_target, # Target mà GMM tự tin tại t_leader
        't_leader': t_leader,
        't_total':  n_keep,
        'D_rem':    D_remaining,
        'T_rem':    T_remaining,
        'ID':       ID,
    })

# ── Thống kê ─────────────────────────────────────────────────────────────────
print(f"{'─'*60}")
print(f"  Trajectories processed:  {len(files)}")
print(f"  Valid samples:           {len(ID_list)}")
print(f"  Skipped (too short):     {skipped_short}")
print(f"  Skipped (always still):  {skipped_no_csv}")
print(f"  Skipped (no LEADER):     {skipped_no_leader}")
print(f"  Skipped (GMM wrong tgt): {skipped_wrong_target}  ← Change/Obstacle scenarios")
print(f"{'─'*60}\n")

if len(ID_list) < 3:
    print("[!] Không đủ mẫu để fit. Kiểm tra lại thư mục training và GROUND_TRUTH mapping.")
    exit(1)

ID_arr = np.array(ID_list)
T_arr  = np.array(T_list)

print(f"ID range:      [{ID_arr.min():.3f}, {ID_arr.max():.3f}]  (mean={ID_arr.mean():.3f})")
print(f"T_remaining:   min={T_arr.min():.2f}s  max={T_arr.max():.2f}s  mean={T_arr.mean():.2f}s")

# In bảng chi tiết từng quỹ đạo
print(f"\n{'File':<35} {'GT':>4} {'GMM':>4}  {'t_L':>5}  {'D(m)':>7}  {'T(s)':>6}  {'ID':>5}")
print('─' * 78)
for info in info_list:
    print(f"  {info['file']:<33} {str(info['gt_target']):>4} {str(info['gmm_target']):>4}  "
          f"{info['t_leader']:>5}  {info['D_rem']:>7.4f}  "
          f"{info['T_rem']:>6.3f}  {info['ID']:>5.3f}")

# ── Linear Regression ─────────────────────────────────────────────────────────
reg   = LinearRegression().fit(ID_arr.reshape(-1, 1), T_arr)
A_new = float(reg.intercept_)
B_new = float(reg.coef_[0])
R2    = reg.score(ID_arr.reshape(-1, 1), T_arr)

T_pred_new = A_new + B_new * ID_arr
T_pred_cur = A_CURRENT + B_CURRENT * ID_arr

rmse_new = np.sqrt(np.mean((T_arr - T_pred_new) ** 2))
rmse_cur = np.sqrt(np.mean((T_arr - T_pred_cur) ** 2))
mae_new  = np.mean(np.abs(T_arr - T_pred_new))
mae_cur  = np.mean(np.abs(T_arr - T_pred_cur))

print(f"\n{'═'*70}")
print(f"  {'':32}  {'a':>8}  {'b':>8}  {'R²':>6}  {'RMSE':>7}  {'MAE':>7}")
print(f"  {'NEW  (GMM LEADER moment)':32}  {A_new:>8.4f}  {B_new:>8.4f}  "
      f"{R2:>6.3f}  {rmse_new:>7.3f}s  {mae_new:>7.3f}s")
print(f"  {'CURRENT (config.py)':32}  {A_CURRENT:>8.4f}  {B_CURRENT:>8.4f}  "
      f"{'---':>6}  {rmse_cur:>7.3f}s  {mae_cur:>7.3f}s")
print(f"{'═'*70}")
print(f"\nSuggested update in config.py:")
print(f"  FITTS_A = {A_new:.4f}")
print(f"  FITTS_B = {B_new:.4f}")
print(f"  FITTS_W = {W}  (unchanged)")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ID_plot = np.linspace(max(0.1, ID_arr.min() - 0.5), ID_arr.max() + 0.5, 300)

# Left: Scatter + Regression lines
ax = axes[0]
ax.scatter(ID_arr, T_arr, s=40, alpha=0.7, color='steelblue', zorder=3,
           label=f'Samples at LEADER moment  (N={len(ID_arr)})')
ax.plot(ID_plot, A_new + B_new * ID_plot, 'r-', linewidth=2, zorder=4,
        label=f'NEW:     T = {A_new:.3f} + {B_new:.3f}·ID   (R²={R2:.3f})')
ax.plot(ID_plot, A_CURRENT + B_CURRENT * ID_plot, 'k--', linewidth=2, zorder=4,
        label=f'CURRENT: T = {A_CURRENT} + {B_CURRENT}·ID')
ax.set_xlabel('ID = log₂(2·D / W)')
ax.set_ylabel('T_remaining (s)')
ax.set_title("Fitts' Law Re-fit\n(sampled at GMM LEADER transition)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Residuals histogram
ax2 = axes[1]
ax2.hist(T_arr - T_pred_new, bins=max(10, len(ID_arr) // 3),
         alpha=0.7, color='steelblue', label=f'NEW   RMSE={rmse_new:.3f}s')
ax2.hist(T_arr - T_pred_cur, bins=max(10, len(ID_arr) // 3),
         alpha=0.5, color='tomato',    label=f'CURRENT  RMSE={rmse_cur:.3f}s')
ax2.axvline(0, color='black', linewidth=1, linestyle='--')
ax2.set_xlabel('Residual: T_actual − T_predicted (s)')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Residuals')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Fitts' Law Re-fit (GMM-aware)  |  N_samples={len(ID_arr)}  "
    f"|  P_HIGH={P_HIGH}  |  N_conf={N_CONFIDENCE_REQUIRED}",
    fontweight='bold')
plt.tight_layout()
plt.savefig('fitts_law_refit_gmm.png', dpi=150)
print(f'\nPlot saved: fitts_law_refit_gmm.png')
plt.show()
