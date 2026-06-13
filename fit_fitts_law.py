"""
fit_fitts_law.py  (v2)
─────────────────────────────────────────────────────────────────────────────
Fit Fitts' Law T = a + b*ID  với  ID = log2(2*D/W)

Cách tiếp cận ĐÚNG:
  - Mỗi quỹ đạo được cắt để loại bỏ phần đuôi "đứng yên" (v < V_THRESH)
  - Với mỗi bước thời gian t dọc theo quỹ đạo đã cắt, ta tính:
      D_remaining(t) = ||x_f - x(t)||      (khoảng cách còn lại tới đích)
      T_remaining(t) = (n_keep - 1 - t)*DT (thời gian còn lại)
  - Tập hợp tất cả các cặp (ID_remaining, T_remaining) từ mọi quỹ đạo
    để fit linear regression
  → Nhất quán với cách dùng trong outer_loop.py: tại thời điểm LEADER,
    Fitts' Law cần dự đoán "từ vị trí hiện tại mất bao lâu để đến đích"
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

# ── Tham số ───────────────────────────────────────────────────────────────────
FOLDER   = 'Experiment_Train_Trajectory_HRI'
DT       = 12.0 / 191
SIGMA    = 1          # Gaussian smoothing (giong config.py)
W        = 0.08       # Target width (m)
V_THRESH = 0.05       # 2 cm/s -- nguong dung yen

# Sample: lay moi STEP_EVERY buoc de tranh qua nhieu diem tuong quan
STEP_EVERY     = 3
# Chi lay cac buoc trong [MIN_FRAC, MAX_FRAC] cua quy dao (tranh diem dau/cuoi nhieu)
MIN_T_REMAIN   = 0.2  # Loai diem co T_remaining < 0.2s (sap den dich)
MIN_D_REMAIN   = W / 2  # Loai diem co D < W/2 (gan dich, degenerate)

# Tham so hien tai trong config.py (de so sanh)
A_CURRENT = 4.3939
B_CURRENT = 0.5224

# ── Thu thap du lieu ──────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(FOLDER, 'trajectory_*.csv')))
print(f'Training files: {len(files)}')
print(f'V_THRESH={V_THRESH*100:.0f} cm/s  W={W} m  STEP_EVERY={STEP_EVERY}\n')

ID_list = []
T_list  = []
skipped = 0

for f in files:
    df = pd.read_csv(f)
    if not {'X', 'Y', 'Z'}.issubset(df.columns):
        skipped += 1
        continue
    coords = df[['X', 'Y', 'Z']].values.copy()
    n = len(coords)
    if n < 10:
        skipped += 1
        continue

    # Lam min
    for i in range(3):
        coords[:, i] = gaussian_filter1d(coords[:, i], sigma=SIGMA)

    # Cat duoi dung yen
    vels = np.linalg.norm(np.diff(coords, axis=0), axis=1) / DT
    if np.all(vels < V_THRESH):
        skipped += 1
        continue
    last_active = int(np.where(vels >= V_THRESH)[0][-1])
    n_keep = last_active + 2
    coords = coords[:n_keep]

    x_f = coords[-1]  # Diem dich = diem dung sau khi cat

    # Lay mau doc theo quy dao
    for t in range(0, n_keep - 1, STEP_EVERY):
        T_remaining = (n_keep - 1 - t) * DT
        if T_remaining < MIN_T_REMAIN:
            continue

        D_remaining = np.linalg.norm(x_f - coords[t])
        if D_remaining < MIN_D_REMAIN:
            continue

        ID = np.log2(2 * D_remaining / W)
        ID_list.append(ID)
        T_list.append(T_remaining)

ID_arr = np.array(ID_list)
T_arr  = np.array(T_list)

print(f'Valid sample points: {len(ID_arr)}  (skipped {skipped} trajectories)')
print(f'T_remaining: mean={T_arr.mean():.2f}s  '
      f'min={T_arr.min():.2f}s  max={T_arr.max():.2f}s')
D_remaining_approx = (W / 2) * (2 ** ID_arr)  # D = W/2 * 2^ID
print(f'D_remaining: mean={D_remaining_approx.mean()*100:.1f} cm  '
      f'min={D_remaining_approx.min()*100:.1f} cm  max={D_remaining_approx.max()*100:.1f} cm')
print(f'ID range: [{ID_arr.min():.2f}, {ID_arr.max():.2f}]  mean={ID_arr.mean():.2f}')

# ── Linear Regression ─────────────────────────────────────────────────────────
reg  = LinearRegression().fit(ID_arr.reshape(-1, 1), T_arr)
A_new = reg.intercept_
B_new = reg.coef_[0]
R2    = reg.score(ID_arr.reshape(-1, 1), T_arr)

T_pred_new = A_new + B_new * ID_arr
T_pred_cur = A_CURRENT + B_CURRENT * ID_arr

rmse_new = np.sqrt(np.mean((T_arr - T_pred_new)**2))
rmse_cur = np.sqrt(np.mean((T_arr - T_pred_cur)**2))
mae_new  = np.mean(np.abs(T_arr - T_pred_new))
mae_cur  = np.mean(np.abs(T_arr - T_pred_cur))

print()
print('=== FITTS LAW FIT RESULTS ===')
print(f'  {"":30}  {"a":>8}  {"b":>8}  {"R2":>6}  {"RMSE":>7}  {"MAE":>7}')
print(f'  {"NEW  (D_remaining, T_remaining)":30}  {A_new:>8.4f}  {B_new:>8.4f}  {R2:>6.3f}  {rmse_new:>7.3f}s  {mae_new:>7.3f}s')
print(f'  {"CURRENT (config.py)":30}  {A_CURRENT:>8.4f}  {B_CURRENT:>8.4f}  {"---":>6}  {rmse_cur:>7.3f}s  {mae_cur:>7.3f}s')
print()
print(f'Suggested update in config.py:')
print(f'  FITTS_A = {A_new:.4f}')
print(f'  FITTS_B = {B_new:.4f}')
print(f'  FITTS_W = {W}  (unchanged)')

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter + Regression lines
ID_plot = np.linspace(ID_arr.min() - 0.2, ID_arr.max() + 0.2, 300)
ax = axes[0]
ax.scatter(ID_arr, T_arr, s=4, alpha=0.2, color='steelblue', label='Samples (D_remaining, T_remaining)')
ax.plot(ID_plot, A_new + B_new * ID_plot, 'r-', linewidth=2,
        label=f'NEW: T = {A_new:.3f} + {B_new:.3f}.ID  (R2={R2:.3f})')
ax.plot(ID_plot, A_CURRENT + B_CURRENT * ID_plot, 'k--', linewidth=2,
        label=f'CURRENT: T = {A_CURRENT} + {B_CURRENT}.ID')
ax.set_xlabel('ID = log2(2 * D_remaining / W)')
ax.set_ylabel('T_remaining (s)')
ax.set_title("Fitts' Law Re-fit (remaining distance & time)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Residuals comparison
ax2 = axes[1]
ax2.hist(T_arr - T_pred_new, bins=60, alpha=0.7, color='steelblue',
         label=f'NEW   RMSE={rmse_new:.3f}s')
ax2.hist(T_arr - T_pred_cur, bins=60, alpha=0.5, color='tomato',
         label=f'CURRENT  RMSE={rmse_cur:.3f}s')
ax2.axvline(0, color='black', linewidth=1, linestyle='--')
ax2.set_xlabel('Residual: T_actual - T_predicted (s)')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Residuals')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Fitts' Law Re-fit  |  V_thresh={V_THRESH*100:.0f} cm/s  |"
    f"  N_samples={len(ID_arr)}  N_traj={len(files)-skipped}",
    fontweight='bold')
plt.tight_layout()
plt.savefig('fitts_law_refit.png', dpi=150)
print(f'\nPlot saved: fitts_law_refit.png')
plt.show()
