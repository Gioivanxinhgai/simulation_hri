"""
test_gp_hri.py
==============
Test 2 mô hình GP đã huấn luyện:
  - gp_hri_kmeans_ratio_1p0  (KMeans ratio = 1.0)
  - gp_hri_kmeans_ratio_0p1  (KMeans ratio = 0.1)

Dữ liệu test: 1 quỹ đạo cho mỗi ScenarioID 5, 9, 14.
Kết quả: hình dự đoán trên từng trục (X/Y/Z), quỹ đạo 3D, và file log .txt.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Headless – không cần màn hình
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter1d
import time

# ─── Font Times New Roman ────────────────────────────────────────────────
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['font.size']        = 12
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['axes.labelsize']   = 12
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 10

# ─── CONFIG ──────────────────────────────────────────────────────────────
DATA_DIR    = "Experiment_Train_Trajectory_HRI"
WINDOW_SIZE = 10
HORIZON     = 1

# File test: (ScenarioID, filename)
TEST_FILES = [
    (5,  "trajectory_018.csv"),
    (9,  "trajectory_000.csv"),
    (14, "trajectory_046.csv"),
]

# 2 mô hình cần test
MODELS = {
    "KMeans_1p0": "gp_hri_kmeans_ratio_1p0",
    "KMeans_0p1": "gp_hri_kmeans_ratio_0p1",
}

# Thư mục gốc để lưu output
OUTPUT_ROOT = "gp_hri_test_results"

# ─── HELPERS ─────────────────────────────────────────────────────────────

def read_and_smooth(filepath, sigma=1, apply_smoothing=True):
    df = pd.read_csv(filepath)[["X", "Y", "Z"]].dropna()
    if apply_smoothing:
        df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
        df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
        df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
    return df.reset_index(drop=True)


def create_sequences_with_padding(data_sequence, window_size, horizon, start_point):
    """Tạo cửa sổ sliding window với zero-padding bằng start_point."""
    X_sequences, Y_targets = [], []
    L = data_sequence.shape[0]

    for i in range(1, L):
        current_window = np.tile(start_point, (window_size, 1)).astype(data_sequence.dtype)
        actual_start = max(0, i - window_size)
        num_actual  = i - actual_start
        if num_actual > 0:
            current_window[window_size - num_actual:window_size] = data_sequence[actual_start:i]
        X_sequences.append(current_window.flatten())
        Y_targets.append(data_sequence[i : i + 1].flatten())

    return np.array(X_sequences), np.array(Y_targets)


def load_model(model_dir):
    with open(os.path.join(model_dir, "scaler_x.pkl"), "rb") as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(model_dir, "gp_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler_x, scaler_y, model


def predict_trajectory(model, scaler_x, scaler_y, X_test, Y_test, first_point):
    """Dự đoán toàn bộ quỹ đạo, trả về arrays gốc, dự đoán và khoảng tin cậy."""
    orig_list, pred_list = [first_point], [first_point.copy()]
    lb_x, ub_x = [first_point[0]], [first_point[0]]
    lb_y, ub_y = [first_point[1]], [first_point[1]]
    lb_z, ub_z = [first_point[2]], [first_point[2]]

    for i in range(X_test.shape[0]):
        X_win = X_test[i].reshape(WINDOW_SIZE, 3)
        X_s   = scaler_x.transform(X_win).flatten().reshape(1, -1)

        mean_s, var_s = model.predict_y(X_s)

        pred_unscaled = scaler_y.inverse_transform(mean_s.numpy()).flatten()  # (3,)

        # Unscale variance
        var_np  = var_s.numpy()                          # shape (1, 3)
        var_np  = var_np.reshape(-1, 3)
        scale2  = np.square(scaler_y.scale_.reshape(1, -1))  # (1, 3)
        std_dev = np.sqrt(var_np * scale2).flatten()    # (3,)

        lb_x.append(pred_unscaled[0] - 1.96 * std_dev[0])
        ub_x.append(pred_unscaled[0] + 1.96 * std_dev[0])
        lb_y.append(pred_unscaled[1] - 1.96 * std_dev[1])
        ub_y.append(pred_unscaled[1] + 1.96 * std_dev[1])
        lb_z.append(pred_unscaled[2] - 1.96 * std_dev[2])
        ub_z.append(pred_unscaled[2] + 1.96 * std_dev[2])

        orig_list.append(Y_test[i])
        pred_list.append(pred_unscaled)

    orig = np.array(orig_list)
    pred = np.array(pred_list)
    bounds = {
        "lb_x": np.array(lb_x), "ub_x": np.array(ub_x),
        "lb_y": np.array(lb_y), "ub_y": np.array(ub_y),
        "lb_z": np.array(lb_z), "ub_z": np.array(ub_z),
    }
    return orig, pred, bounds


def compute_metrics(orig, pred):
    err  = orig - pred
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(err)))
    # Per-axis
    mse_x  = float(np.mean(err[:, 0] ** 2))
    mse_y  = float(np.mean(err[:, 1] ** 2))
    mse_z  = float(np.mean(err[:, 2] ** 2))
    rmse_x = float(np.sqrt(mse_x))
    rmse_y = float(np.sqrt(mse_y))
    rmse_z = float(np.sqrt(mse_z))
    mae_x  = float(np.mean(np.abs(err[:, 0])))
    mae_y  = float(np.mean(np.abs(err[:, 1])))
    mae_z  = float(np.mean(np.abs(err[:, 2])))
    return {
        "mse": mse, "rmse": rmse, "mae": mae,
        "mse_x": mse_x, "mse_y": mse_y, "mse_z": mse_z,
        "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_z": rmse_z,
        "mae_x": mae_x, "mae_y": mae_y, "mae_z": mae_z,
    }


def coverage_percent(orig_col, lb, ub):
    return float(np.mean((orig_col >= lb) & (orig_col <= ub)) * 100)


def save_axis_plot(time_idx, orig, pred, lb, ub, axis_name, unit,
                   scenario_id, model_tag, out_dir):
    fig, ax = plt.subplots(figsize=(4.3, 4.3))
    ax.plot(time_idx, orig,  '-',  color='blue', linewidth=1.5, label='Observation')
    ax.plot(time_idx, pred,  '--', color='red',  linewidth=1.5, label='Prediction')
    ax.fill_between(time_idx, lb, ub, color='gray', alpha=0.2, label='95% CI')
    ax.set_xlabel('Timestamp (s)')
    ax.set_ylabel(f'{axis_name} (m)')
    ax.set_title(f'ScenarioID {scenario_id} — {axis_name} axis [{model_tag}]')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"scenario{scenario_id}_{axis_name}_{model_tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def save_3d_plot(orig, pred, scenario_id, model_tag, out_dir):
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(orig[:, 0], orig[:, 1], orig[:, 2],
            '-', color='blue', linewidth=1.5, label='Observation')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2],
            '--', color='red',  linewidth=1.5, label='Prediction')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'ScenarioID {scenario_id} — 3D Trajectory [{model_tag}]')
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"scenario{scenario_id}_3D_{model_tag}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ── Pre-load tất cả mô hình ──────────────────────────────────────────
    print("Loading models...")
    loaded_models = {}
    for tag, model_dir in MODELS.items():
        print(f"  [{tag}] from '{model_dir}' ...")
        scaler_x, scaler_y, gp_model = load_model(model_dir)
        loaded_models[tag] = (scaler_x, scaler_y, gp_model)
    print("All models loaded.\n")

    # Accumulate per-model global results
    global_results = {tag: {"all_orig": [], "all_pred": [], "total_time": 0.0}
                      for tag in MODELS}

    log_lines = []
    log_lines.append("=" * 72)
    log_lines.append("  GP HRI TEST RESULTS")
    log_lines.append(f"  Models : {list(MODELS.keys())}")
    log_lines.append(f"  Test   : ScenarioID 5 ({TEST_FILES[0][1]}), "
                     f"9 ({TEST_FILES[1][1]}), 14 ({TEST_FILES[2][1]})")
    log_lines.append("=" * 72)

    # ── Lặp qua từng file test ────────────────────────────────────────────
    for scenario_id, fname in TEST_FILES:
        full_path = os.path.join(DATA_DIR, fname)
        print(f"\n{'─'*60}")
        print(f"Test file : {fname}  (ScenarioID {scenario_id})")
        print(f"{'─'*60}")

        log_lines.append("")
        log_lines.append(f"{'─'*60}")
        log_lines.append(f"File: {fname}   ScenarioID: {scenario_id}")
        log_lines.append(f"{'─'*60}")

        if not os.path.isfile(full_path):
            msg = f"  [ERROR] File not found: {full_path}"
            print(msg); log_lines.append(msg)
            continue

        df            = read_and_smooth(full_path)
        full_sequence = df[["X", "Y", "Z"]].values
        L             = full_sequence.shape[0]
        start_point   = full_sequence[0]

        print(f"  Points      : {L}")
        print(f"  Start point : {start_point}")
        log_lines.append(f"  Points      : {L}")
        log_lines.append(f"  Start point : X={start_point[0]:.4f}  Y={start_point[1]:.4f}  Z={start_point[2]:.4f}")

        # Build windows once (shared across models)
        X_test, Y_test = create_sequences_with_padding(
            full_sequence, WINDOW_SIZE, HORIZON, start_point
        )
        # Time axis: L points, spanning 12 s
        time_idx = np.arange(L) / ((L - 1) / 12.0)

        # Per-scenario output dir
        out_dir = os.path.join(OUTPUT_ROOT, f"scenario_{scenario_id}")
        os.makedirs(out_dir, exist_ok=True)

        # ── Run each model ──────────────────────────────────────────────
        for tag, (scaler_x, scaler_y, gp_model) in loaded_models.items():
            print(f"\n  [{tag}] predicting...")
            log_lines.append(f"\n  Model: {tag}")

            t_start = time.time()
            orig, pred, bounds = predict_trajectory(
                gp_model, scaler_x, scaler_y, X_test, Y_test, start_point
            )
            t_end   = time.time()
            elapsed = t_end - t_start

            metrics = compute_metrics(orig, pred)
            cov_x   = coverage_percent(orig[:, 0], bounds["lb_x"], bounds["ub_x"])
            cov_y   = coverage_percent(orig[:, 1], bounds["lb_y"], bounds["ub_y"])
            cov_z   = coverage_percent(orig[:, 2], bounds["lb_z"], bounds["ub_z"])

            # Accumulate global
            global_results[tag]["all_orig"].append(orig)
            global_results[tag]["all_pred"].append(pred)
            global_results[tag]["total_time"] += elapsed

            # Console
            print(f"    Points   : {len(orig)}")
            print(f"    MSE      : {metrics['mse']:.8f}")
            print(f"    RMSE     : {metrics['rmse']:.8f}")
            print(f"    MAE      : {metrics['mae']:.8f}")
            print(f"    Time     : {elapsed:.4f} s")
            print(f"    CovX 95%: {cov_x:.1f}%  CovY 95%: {cov_y:.1f}%  CovZ 95%: {cov_z:.1f}%")

            # Log
            log_lines.append(f"    Points predicted : {len(orig)}")
            log_lines.append(f"    MSE (overall)    : {metrics['mse']:.8f}")
            log_lines.append(f"      MSE_X          : {metrics['mse_x']:.8f}")
            log_lines.append(f"      MSE_Y          : {metrics['mse_y']:.8f}")
            log_lines.append(f"      MSE_Z          : {metrics['mse_z']:.8f}")
            log_lines.append(f"    RMSE (overall)   : {metrics['rmse']:.8f}")
            log_lines.append(f"      RMSE_X         : {metrics['rmse_x']:.8f}")
            log_lines.append(f"      RMSE_Y         : {metrics['rmse_y']:.8f}")
            log_lines.append(f"      RMSE_Z         : {metrics['rmse_z']:.8f}")
            log_lines.append(f"    MAE (overall)    : {metrics['mae']:.8f}")
            log_lines.append(f"      MAE_X          : {metrics['mae_x']:.8f}")
            log_lines.append(f"      MAE_Y          : {metrics['mae_y']:.8f}")
            log_lines.append(f"      MAE_Z          : {metrics['mae_z']:.8f}")
            log_lines.append(f"    95% CI Coverage  : X={cov_x:.1f}%  Y={cov_y:.1f}%  Z={cov_z:.1f}%")
            log_lines.append(f"    Prediction time  : {elapsed:.4f} s")

            # Save plots
            print(f"    Saving plots to '{out_dir}' ...")
            save_axis_plot(time_idx, orig[:, 0], pred[:, 0],
                           bounds["lb_x"], bounds["ub_x"],
                           "X", "m", scenario_id, tag, out_dir)
            save_axis_plot(time_idx, orig[:, 1], pred[:, 1],
                           bounds["lb_y"], bounds["ub_y"],
                           "Y", "m", scenario_id, tag, out_dir)
            save_axis_plot(time_idx, orig[:, 2], pred[:, 2],
                           bounds["lb_z"], bounds["ub_z"],
                           "Z", "m", scenario_id, tag, out_dir)
            save_3d_plot(orig, pred, scenario_id, tag, out_dir)
            log_lines.append(f"    Plots saved to   : {out_dir}/scenario{scenario_id}_{{X,Y,Z,3D}}_{tag}.png")

    # ── Global summary ────────────────────────────────────────────────────
    log_lines.append("")
    log_lines.append("=" * 72)
    log_lines.append("  GLOBAL SUMMARY  (across all 3 test trajectories)")
    log_lines.append("=" * 72)

    print(f"\n{'='*60}")
    print("  GLOBAL SUMMARY")
    print(f"{'='*60}")

    for tag in MODELS:
        all_orig = np.vstack(global_results[tag]["all_orig"])
        all_pred = np.vstack(global_results[tag]["all_pred"])
        total_t  = global_results[tag]["total_time"]

        gm = compute_metrics(all_orig, all_pred)
        avg_t = total_t / len(TEST_FILES)

        print(f"\n  [{tag}]")
        print(f"    Total points : {len(all_orig)}")
        print(f"    MSE          : {gm['mse']:.8f}")
        print(f"    RMSE         : {gm['rmse']:.8f}")
        print(f"    MAE          : {gm['mae']:.8f}")
        print(f"    Total time   : {total_t:.4f} s  (avg {avg_t:.4f} s/traj)")

        log_lines.append("")
        log_lines.append(f"  Model: {tag}")
        log_lines.append(f"    Total points     : {len(all_orig)}")
        log_lines.append(f"    MSE (overall)    : {gm['mse']:.8f}")
        log_lines.append(f"      MSE_X          : {gm['mse_x']:.8f}")
        log_lines.append(f"      MSE_Y          : {gm['mse_y']:.8f}")
        log_lines.append(f"      MSE_Z          : {gm['mse_z']:.8f}")
        log_lines.append(f"    RMSE (overall)   : {gm['rmse']:.8f}")
        log_lines.append(f"      RMSE_X         : {gm['rmse_x']:.8f}")
        log_lines.append(f"      RMSE_Y         : {gm['rmse_y']:.8f}")
        log_lines.append(f"      RMSE_Z         : {gm['rmse_z']:.8f}")
        log_lines.append(f"    MAE (overall)    : {gm['mae']:.8f}")
        log_lines.append(f"      MAE_X          : {gm['mae_x']:.8f}")
        log_lines.append(f"      MAE_Y          : {gm['mae_y']:.8f}")
        log_lines.append(f"      MAE_Z          : {gm['mae_z']:.8f}")
        log_lines.append(f"    Total pred. time : {total_t:.4f} s")
        log_lines.append(f"    Avg per traj.    : {avg_t:.4f} s")

    log_lines.append("")
    log_lines.append("=" * 72)

    # ── Write log ─────────────────────────────────────────────────────────
    log_path = os.path.join(OUTPUT_ROOT, "test_results.txt")
    with open(log_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(log_lines) + "\n")
    print(f"\nLog saved to: {log_path}")
    print(f"All outputs in: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
