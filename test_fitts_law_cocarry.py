"""
test_fitts_law_cocarry.py
----------------------------------------------------------------------
Test Fitts' Law accuracy on the cocarry test set.

Uses parameters fitted by train_fitts_law_cocarry.py
(loaded from fitts_params_cocarry.json).

For each test trajectory, samples (D_remaining, T_remaining) and compares
T_predicted = a + b * log2(2D/W) against T_actual.

Also compares with the HRI-trained parameters from config.py.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# -- Parameters ----------------------------------------------------------------
DATA_FOLDER  = 'Filtered_Cocarry_Logs'
TEST_LIST    = os.path.join(DATA_FOLDER, 'test_file_list.json')
PARAMS_FILE  = os.path.join(DATA_FOLDER, 'fitts_params_cocarry.json')

SIGMA        = 1        # Gaussian smoothing sigma
V_THRESH     = 0.02     # Velocity threshold for tail trimming (m/s)
STEP_EVERY   = 3        # Sub-sample every N steps
MIN_T_REMAIN = 0.2      # Discard points with T_remaining < 0.2s


# -- Load cocarry-trained parameters ------------------------------------------
if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    A_COCARRY = params['a']
    B_COCARRY = params['b']
    W_COCARRY = params['W']
    print(f"Loaded cocarry Fitts' Law parameters from {PARAMS_FILE}")
    print(f"  a = {A_COCARRY:.4f},  b = {B_COCARRY:.4f},  W = {W_COCARRY}")
else:
    print(f"[!] {PARAMS_FILE} not found. Run train_fitts_law_cocarry.py first.")
    exit(1)

W = W_COCARRY
MIN_D_REMAIN = W / 2

# -- Load test file list -------------------------------------------------------
with open(TEST_LIST, 'r', encoding='utf-8') as f:
    file_list = json.load(f)

print(f"\nTest files: {len(file_list)}")

# -- Collect samples -----------------------------------------------------------
ID_list   = []
T_list    = []
info_list = []

skipped = 0

for fname in file_list:
    fpath = os.path.join(DATA_FOLDER, fname)
    if not os.path.exists(fpath):
        skipped += 1
        continue

    df = pd.read_csv(fpath)
    if not {'meas_x', 'meas_y', 'meas_z', 'wall_time'}.issubset(df.columns):
        skipped += 1
        continue

    # Parse timestamps
    df['wall_time'] = pd.to_datetime(df['wall_time'])
    timestamps = (df['wall_time'] - df['wall_time'].iloc[0]).dt.total_seconds().values

    coords = df[['meas_x', 'meas_y', 'meas_z']].values.copy()
    n = len(coords)
    if n < 10:
        skipped += 1
        continue

    # Smooth
    for i in range(3):
        coords[:, i] = gaussian_filter1d(coords[:, i], sigma=SIGMA)

    # Compute per-step DT
    dt_array = np.diff(timestamps)
    if len(dt_array) == 0 or np.median(dt_array) < 1e-6:
        skipped += 1
        continue
    DT_file = np.median(dt_array)

    # Trim stationary tail
    vels = np.linalg.norm(np.diff(coords, axis=0), axis=1) / DT_file
    if np.all(vels < V_THRESH):
        skipped += 1
        continue
    last_active = int(np.where(vels >= V_THRESH)[0][-1])
    n_keep = last_active + 2
    coords = coords[:n_keep]
    timestamps = timestamps[:n_keep]

    x_f = coords[-1]

    # Sample along trajectory
    n_samples_this = 0
    for t in range(0, n_keep - 1, STEP_EVERY):
        T_remaining = timestamps[n_keep - 1] - timestamps[t]
        if T_remaining < MIN_T_REMAIN:
            continue

        D_remaining = np.linalg.norm(x_f - coords[t])
        if D_remaining < MIN_D_REMAIN:
            continue

        ID = np.log2(2.0 * D_remaining / W)
        ID_list.append(ID)
        T_list.append(T_remaining)
        n_samples_this += 1

    if n_samples_this > 0:
        task_time_actual = timestamps[n_keep - 1] - timestamps[0]
        
        # Dự đoán thời gian hoàn thành tổng cộng từ vị trí xuất phát (t=0)
        D_init = np.linalg.norm(x_f - coords[0])
        ID_init = np.log2(2.0 * D_init / W)
        task_time_pred = A_COCARRY + B_COCARRY * ID_init

        info_list.append({
            'file':      fname,
            'n_keep':    n_keep,
            'DT':        DT_file,
            'task_time_actual': task_time_actual,
            'task_time_pred':   task_time_pred,
            'n_samples': n_samples_this,
        })

# -- Statistics ----------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Test trajectories:       {len(info_list)}")
print(f"  Skipped:                 {skipped}")
print(f"  Total test samples:      {len(ID_list)}")
print(f"{'='*60}\n")

if len(ID_list) < 3:
    print("[!] Not enough test samples.")
    exit(1)

ID_arr = np.array(ID_list)
T_arr  = np.array(T_list)

# -- Predictions ---------------------------------------------------------------
T_pred = A_COCARRY + B_COCARRY * ID_arr

rmse = np.sqrt(np.mean((T_arr - T_pred) ** 2))
mae  = np.mean(np.abs(T_arr - T_pred))

# R2
ss_res = np.sum((T_arr - T_pred) ** 2)
ss_tot = np.sum((T_arr - T_arr.mean()) ** 2)
r2     = 1 - ss_res / ss_tot

# Per-trajectory table
print(f"{'File':<53} {'n':>4}  {'T_actual':>9}  {'T_pred':>9}  {'Error':>8}")
print('-' * 88)
for info in info_list:
    err = info['task_time_pred'] - info['task_time_actual']
    print(f"  {info['file']:<51} {info['n_keep']:>4}  "
          f"{info['task_time_actual']:>7.2f} s  {info['task_time_pred']:>7.2f} s  {err:>+7.2f} s")

# -- Results -------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  FITTS' LAW TEST RESULTS (Cocarry Test Set)")
print(f"{'='*70}")
print(f"  {'Model':<30} {'a':>8} {'b':>8} {'R2':>7} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*70}")
print(f"  {'COCARRY-trained':<30} {A_COCARRY:>8.4f} {B_COCARRY:>8.4f} "
      f"{r2:>7.4f} {rmse:>8.4f}s {mae:>7.4f}s")
print(f"{'='*70}")

# -- Plot ----------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ID_plot = np.linspace(max(0.1, ID_arr.min() - 0.5), ID_arr.max() + 0.5, 300)

# Left: Scatter + regression line
ax = axes[0]
ax.scatter(ID_arr, T_arr, s=8, alpha=0.3, color='steelblue', zorder=3,
           label=f'Test samples (N={len(ID_arr)})')
ax.plot(ID_plot, A_COCARRY + B_COCARRY * ID_plot, 'r-', linewidth=2, zorder=4,
        label=f'T={A_COCARRY:.3f}+{B_COCARRY:.3f}*ID (R2={r2:.3f})')
ax.set_xlabel('ID = log2(2D / W)')
ax.set_ylabel('T_remaining (s)')
ax.set_title("Fitts' Law Test (Cocarry Test Set)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Residuals
ax2 = axes[1]
ax2.hist(T_arr - T_pred, bins=40, alpha=0.7, color='steelblue',
         label=f'RMSE={rmse:.3f}s')
ax2.axvline(0, color='black', linewidth=1, linestyle='--')
ax2.set_xlabel('Residual: T_actual - T_predicted (s)')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Residuals Comparison')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"Fitts' Law Test (Cocarry)  |  N_test={len(ID_arr)}  |  N_traj={len(info_list)}",
    fontweight='bold')
plt.tight_layout()
