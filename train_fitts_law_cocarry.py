"""
train_fitts_law_cocarry.py
----------------------------------------------------------------------
Fit Fitts' Law:  T = a + b * ID,   ID = log2(2*D/W)

Adapted for cocarry dataset (single target, no GMM needed).
  - Robot always knows the destination -> LEADER from the start.
  - Sample (D_remaining, T_remaining) along each trajectory.
  - Fit Linear Regression to obtain a, b.

Data source: Filtered_Cocarry_Logs/train_file_list.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

# -- Parameters ----------------------------------------------------------------
DATA_FOLDER  = 'Filtered_Cocarry_Logs'
TRAIN_LIST   = os.path.join(DATA_FOLDER, 'train_file_list.json')

SIGMA        = 1        # Gaussian smoothing sigma
W            = 0.2     # Target width (m) -- same as HRI config
V_THRESH     = 0.005     # Velocity threshold for tail trimming (m/s)
STEP_EVERY   = 3        # Sub-sample every N steps to reduce correlation
MIN_T_REMAIN = 0.2      # Discard points with T_remaining < 0.2s
MIN_D_REMAIN = W / 2    # Discard points with D < W/2

# -- Load file list ------------------------------------------------------------
with open(TRAIN_LIST, 'r', encoding='utf-8') as f:
    file_list = json.load(f)

print(f"Training files: {len(file_list)}")
print(f"W = {W} m,  V_THRESH = {V_THRESH*100:.1f} cm/s,  STEP_EVERY = {STEP_EVERY}\n")

# -- Collect samples -----------------------------------------------------------
ID_list   = []
T_list    = []
info_list = []

skipped_short = 0
skipped_still = 0

for fname in file_list:
    fpath = os.path.join(DATA_FOLDER, fname)
    if not os.path.exists(fpath):
        skipped_short += 1
        continue

    df = pd.read_csv(fpath)
    if not {'meas_x', 'meas_y', 'meas_z', 'wall_time'}.issubset(df.columns):
        skipped_short += 1
        continue

    # Parse timestamps to compute actual DT per file
    df['wall_time'] = pd.to_datetime(df['wall_time'])
    timestamps = (df['wall_time'] - df['wall_time'].iloc[0]).dt.total_seconds().values

    coords = df[['meas_x', 'meas_y', 'meas_z']].values.copy()
    n = len(coords)
    if n < 10:
        skipped_short += 1
        continue

    # Smooth
    for i in range(3):
        coords[:, i] = gaussian_filter1d(coords[:, i], sigma=SIGMA)

    # Compute per-step DT from actual timestamps
    dt_array = np.diff(timestamps)
    if len(dt_array) == 0 or np.median(dt_array) < 1e-6:
        skipped_short += 1
        continue
    DT_file = np.median(dt_array)  # Median DT for this file

    # Trim stationary tail
    vels = np.linalg.norm(np.diff(coords, axis=0), axis=1) / DT_file
    if np.all(vels < V_THRESH):
        skipped_still += 1
        continue
    last_active = int(np.where(vels >= V_THRESH)[0][-1])
    n_keep = last_active + 2
    coords = coords[:n_keep]
    timestamps = timestamps[:n_keep]

    x_f = coords[-1]  # Single target = end point after trimming

    # Sample along trajectory
    n_samples_this = 0
    for t in range(0, n_keep - 1, STEP_EVERY):
        T_remaining = timestamps[n_keep - 1] - timestamps[t]  # Use real timestamps
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
        task_time = timestamps[n_keep - 1] - timestamps[0]
        info_list.append({
            'file':       fname,
            'n_keep':     n_keep,
            'DT':         DT_file,
            'task_time':  task_time,
            'n_samples':  n_samples_this,
        })

# -- Statistics ----------------------------------------------------------------
print(f"{'='*60}")
print(f"  Trajectories processed:  {len(file_list)}")
print(f"  Valid trajectories:      {len(info_list)}")
print(f"  Skipped (short/missing): {skipped_short}")
print(f"  Skipped (always still):  {skipped_still}")
print(f"  Total sample points:     {len(ID_list)}")
print(f"{'='*60}\n")

if len(ID_list) < 3:
    print("[!] Not enough samples to fit. Check data folder.")
    exit(1)

ID_arr = np.array(ID_list)
T_arr  = np.array(T_list)

print(f"ID range:    [{ID_arr.min():.3f}, {ID_arr.max():.3f}]  mean={ID_arr.mean():.3f}")
print(f"T_remaining: min={T_arr.min():.2f}s  max={T_arr.max():.2f}s  mean={T_arr.mean():.2f}s")

# Per-trajectory table
print(f"\n{'File':<55} {'n':>4} {'DT(ms)':>7} {'Task(s)':>7} {'Pts':>4}")
print('-' * 82)
for info in info_list:
    print(f"  {info['file']:<53} {info['n_keep']:>4} "
          f"{info['DT']*1000:>7.1f} {info['task_time']:>7.2f} {info['n_samples']:>4}")

# -- Linear Regression ---------------------------------------------------------
reg   = LinearRegression().fit(ID_arr.reshape(-1, 1), T_arr)
A_new = float(reg.intercept_)
B_new = float(reg.coef_[0])
R2    = reg.score(ID_arr.reshape(-1, 1), T_arr)

T_pred = A_new + B_new * ID_arr
rmse   = np.sqrt(np.mean((T_arr - T_pred) ** 2))
mae    = np.mean(np.abs(T_arr - T_pred))

print(f"\n{'='*60}")
print(f"  FITTS' LAW FIT RESULTS (Cocarry Training Set)")
print(f"{'='*60}")
print(f"  a = {A_new:.4f}")
print(f"  b = {B_new:.4f}")
print(f"  R2 = {R2:.4f}")
print(f"  RMSE = {rmse:.4f} s")
print(f"  MAE  = {mae:.4f} s")
print(f"{'='*60}")
print(f"\nSuggested config:")
print(f"  FITTS_A = {A_new:.4f}")
print(f"  FITTS_B = {B_new:.4f}")
print(f"  FITTS_W = {W}")

# Save parameters to JSON for test script
params = {'a': A_new, 'b': B_new, 'W': W, 'R2': R2, 'RMSE': rmse, 'MAE': mae}
params_path = os.path.join(DATA_FOLDER, 'fitts_params_cocarry.json')
with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\nParameters saved to: {params_path}")
