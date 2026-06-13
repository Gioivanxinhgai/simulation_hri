import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from config import SCENARIO_NAMES

# ─── Config ───────────────────────────────────────────────────────────────────
TEST_FOLDER = "Experiment_Test_Trajectory_HRI"
TEST_FILE_LIST = "test_file_list.json"
T_F = 12.0              # Tổng thời gian hành trình (s)
DT = T_F / 191          # Time-step giống config.py
SIGMA_SMOOTH = 1        # Gaussian smoothing sigma cho ground truth
SAVE_DIR = "compare_mjm_plots"
# Thư mục chứa CSV simulation (x_ref, x_r, mode ...)
SIM_CSV_DIR = r"MJM_CURRENT_10_10_60_10_2_0.75_3_150_1000000_1400.00\csv_logs"



# ─── MJM (PAPER) ─────────────────────────────────────────────────────────────

def mjm_progress(tau):
    """p(τ) = 10τ³ - 15τ⁴ + 6τ⁵"""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

def generate_mjm_trajectory(x_0, x_f, t_f, dt):
    """
    Sinh toàn bộ quỹ đạo MJM từ X_0 đến X_f trong thời gian t_f.
    X(τ) = X_0 + (X_f - X_0) * p(τ)  với τ = t / t_f
    """
    n_steps = int(round(t_f / dt)) + 1
    t_array = np.linspace(0, t_f, n_steps)
    tau_array = t_array / t_f  # τ ∈ [0, 1]

    p_array = mjm_progress(tau_array)  # (n_steps,)
    trajectory = x_0[np.newaxis, :] + np.outer(p_array, (x_f - x_0))  # (n_steps, 3)

    return t_array, trajectory


# ─── Read ground truth ────────────────────────────────────────────────────────

def read_ground_truth(filepath, sigma=SIGMA_SMOOTH):
    """Doc va lam min quy dao ground truth."""
    df = pd.read_csv(filepath)
    scenario_id = int(df['ScenarioId'].iloc[0]) if 'ScenarioId' in df.columns else None
    df = df.dropna()
    coords = df[['X', 'Y', 'Z']].values.copy()
    for i in range(3):
        coords[:, i] = gaussian_filter1d(coords[:, i], sigma=sigma)
    return coords, scenario_id


def read_sim_trajectory(trajectory_name, sim_csv_dir=SIM_CSV_DIR):
    """
    Đọc CSV simulation tương ứng với trajectory_name.
    Trả về DataFrame (hoặc None nếu không tìm thấy file).
    Các cột quan trọng: time_s, x_ref_X/Y/Z, x_r_X/Y/Z, mode
    """
    sim_path = os.path.join(sim_csv_dir, f"{trajectory_name}.csv")
    if not os.path.exists(sim_path):
        print(f"  [SIM] Không tìm thấy: {sim_path}")
        return None
    df = pd.read_csv(sim_path)
    print(f"  [SIM] Đọc: {sim_path}  ({len(df)} rows)")
    return df


# ─── Plotting ─────────────────────────────────────────────────────────────────

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def _find_target2_idx(gt_traj):
    """Tìm chỉ số của Target 2: điểm cực đại Y (điểm ngoặt quỹ đạo)."""
    return int(np.argmax(gt_traj[:, 1]))


def plot_3d_trajectory(gt_traj, trajectory_name, scenario_name, sim_df=None, save_dir=None):
    """Xuất hình 3D độc lập với X_0, Target 2, Target 3 được đánh dấu.
    sim_df: DataFrame CSV simulation (tùy chọn) để vẽ x_ref theo mode.
    """
    fig3d = plt.figure(figsize=(8, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')

    # ── Quỹ đạo Ground Truth – màu ĐEN ──────────────────────────────────────
    ax3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
              color='black', linewidth=1.8, label='Ground Truth', alpha=0.85)

    # ── x_ref từ simulation (tô màu theo mode) ──────────────────────────────
    if sim_df is not None and {'x_ref_X', 'x_ref_Y', 'x_ref_Z', 'mode'}.issubset(sim_df.columns):
        x_ref = sim_df[['x_ref_X', 'x_ref_Y', 'x_ref_Z']].values
        modes = sim_df['mode'].values
        # Tách segment liên tục theo mode
        _plot_xref_3d(ax3d, x_ref, modes)

    # ── Điểm đặc biệt ────────────────────────────────────────────────────────
    # X_0 – Start
    ax3d.scatter(*gt_traj[0], c='green', s=100, marker='o', zorder=5, label='$X_0$ (Start)')

    # Target 2 – điểm cực đại Y
    t2_idx = _find_target2_idx(gt_traj)
    t2 = gt_traj[t2_idx]
    ax3d.scatter(*t2, c='orange', s=120, marker='^', zorder=5,
                 label=f'Target 2 [{t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f}]')
    ax3d.text(t2[0], t2[1], t2[2] + 0.005, 'Target 2', color='orange',
              fontsize=9, fontweight='bold', ha='center')

    # Target 3 – điểm cuối X_f
    t3 = gt_traj[-1]
    ax3d.scatter(*t3, c='red', s=120, marker='X', zorder=5,
                 label=f'Target 3 [$X_f$: {t3[0]:.4f}, {t3[1]:.4f}, {t3[2]:.4f}]')
    ax3d.text(t3[0], t3[1], t3[2] + 0.005, 'Target 3', color='red',
              fontsize=9, fontweight='bold', ha='center')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'3D Trajectory\n{trajectory_name} | {scenario_name}', fontsize=11, fontweight='bold')
    ax3d.legend(fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'3d_{trajectory_name}.png'
        fig3d.savefig(os.path.join(save_dir, filename), dpi=150)
        print(f"  -> Saved (3D): {filename}")

    plt.show()


def _plot_xref_3d(ax, x_ref, modes):
    """Vẽ x_ref lên ax 3D, tô màu từng đoạn theo mode:
       FOLLOWER → xanh nước biển, LEADER → đỏ.
    """
    MODE_COLOR = {'FOLLOWER': '#1565C0', 'LEADER': '#C62828'}  # xanh đậm / đỏ đậm
    MODE_LABEL = {'FOLLOWER': '$x_{ref}$ (FOLLOWER)', 'LEADER': '$x_{ref}$ (LEADER)'}
    plotted = set()
    i = 0
    n = len(x_ref)
    while i < n:
        cur_mode = modes[i]
        j = i + 1
        while j < n and modes[j] == cur_mode:
            j += 1
        # Đoạn [i, j+1) để liền mạch với đoạn tiếp theo
        seg = x_ref[i:min(j+1, n)]
        color = MODE_COLOR.get(cur_mode, 'gray')
        lbl = MODE_LABEL.get(cur_mode, cur_mode) if cur_mode not in plotted else '_nolegend_'
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=color, linewidth=1.5, linestyle='-', label=lbl, alpha=0.9)
        plotted.add(cur_mode)
        i = j


def _plot_xref_2d(ax, t_sim, x_ref, modes, axis_idx):
    """Vẽ x_ref lên ax 2D (per-axis), tô màu từng đoạn theo mode."""
    MODE_COLOR = {'FOLLOWER': '#1565C0', 'LEADER': '#C62828'}
    MODE_LABEL = {'FOLLOWER': '$x_{ref}$ (FOLLOWER)', 'LEADER': '$x_{ref}$ (LEADER)'}
    plotted = set()
    i = 0
    n = len(x_ref)
    while i < n:
        cur_mode = modes[i]
        j = i + 1
        while j < n and modes[j] == cur_mode:
            j += 1
        color = MODE_COLOR.get(cur_mode, 'gray')
        lbl = MODE_LABEL.get(cur_mode, cur_mode) if cur_mode not in plotted else '_nolegend_'
        
        end_idx = min(j+1, n)
        ax.plot(t_sim[i:end_idx], x_ref[i:end_idx, axis_idx],
                color=color, linewidth=1.5, linestyle='-', label=lbl, alpha=0.9)
        plotted.add(cur_mode)
        i = j


def plot_comparison(t_gt, gt_traj, t_mjm, mjm_traj, trajectory_name, scenario_name,
                    sim_df=None, save_dir=None):
    """Ve so sanh Ground Truth vs MJM tren tung truc X, Y, Z + 3D.
    sim_df: DataFrame CSV simulation (tùy chọn) để vẽ x_ref theo mode.
    """

    fig = plt.figure(figsize=(16, 10))

    # ── 3D subplot ────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    # Ground Truth – màu ĐEN
    ax3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2],
              color='black', linewidth=1.5, label='Ground Truth')
    #ax3d.plot(mjm_traj[:, 0], mjm_traj[:, 1], mjm_traj[:, 2],'r--', linewidth=1.5, label='MJM (PAPER)')

    # x_ref từ simulation
    if sim_df is not None and {'x_ref_X', 'x_ref_Y', 'x_ref_Z', 'mode'}.issubset(sim_df.columns):
        _plot_xref_3d(ax3d, sim_df[['x_ref_X', 'x_ref_Y', 'x_ref_Z']].values, sim_df['mode'].values)

    # X_0 – Start
    ax3d.scatter(*gt_traj[0], c='green', s=80, marker='o', zorder=5, label='$X_0$ (Start)')

    # Target 2 – điểm cực đại Y
    t2_idx = _find_target2_idx(gt_traj)
    t2 = gt_traj[t2_idx]
    ax3d.scatter(*t2, c='orange', s=100, marker='^', zorder=5,
                 label=f'Target 2 [{t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f}]')
    ax3d.text(t2[0], t2[1], t2[2] + 0.005, 'Target 2', color='orange',
              fontsize=8, fontweight='bold', ha='center')

    # Target 3 – điểm cuối X_f
    t3 = gt_traj[-1]
    ax3d.scatter(*t3, c='red', s=100, marker='X', zorder=5,
                 label=f'Target 3 ($X_f$) [{t3[0]:.4f}, {t3[1]:.4f}, {t3[2]:.4f}]')
    ax3d.text(t3[0], t3[1], t3[2] + 0.005, 'Target 3', color='red',
              fontsize=8, fontweight='bold', ha='center')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'3D Trajectory')
    ax3d.legend(fontsize=8, loc='upper left')

    # ── Per-axis plots ────────────────────────────────────────────────────────
    # Chuẩn bị dữ liệu x_ref cho per-axis
    has_sim = (sim_df is not None and
               {'time_s', 'x_ref_X', 'x_ref_Y', 'x_ref_Z', 'mode'}.issubset(sim_df.columns))
    if has_sim:
        t_sim    = sim_df['time_s'].values
        x_ref    = sim_df[['x_ref_X', 'x_ref_Y', 'x_ref_Z']].values
        sim_modes = sim_df['mode'].values

    axis_names = ['X', 'Y', 'Z']
    for i, ax_name in enumerate(axis_names):
        ax = fig.add_subplot(2, 2, i + 2)
        # Ground Truth – màu ĐEN
        ax.plot(t_gt, gt_traj[:, i], color='black', linewidth=1.5, label='Ground Truth')
        #ax.plot(t_mjm, mjm_traj[:, i], 'g--', linewidth=1.5, label='MJM (PAPER)')
        if has_sim:
            _plot_xref_2d(ax, t_sim, x_ref, sim_modes, i)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{ax_name} (m)')
        ax.set_title(f'{ax_name}-axis')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    x0_str = f"[{gt_traj[0,0]:.3f}, {gt_traj[0,1]:.3f}, {gt_traj[0,2]:.3f}]"
    xf_str = f"[{gt_traj[-1,0]:.3f}, {gt_traj[-1,1]:.3f}, {gt_traj[-1,2]:.3f}]"
    plt.suptitle(f'{trajectory_name}\n{scenario_name}\n|  X_0={x0_str}  X_f={xf_str}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'compare_{trajectory_name}.png'
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
        print(f"  -> Saved: {filename}")

    #plt.close('all')
    plt.show()

def plot_error(t_gt, gt_traj, mjm_traj_resampled, trajectory_name, scenario_name, save_dir=None):
    """Vẽ sai số có dấu từng trục (GT - MJM) theo thời gian."""
    err_mm = (gt_traj - mjm_traj_resampled) * 1000

    fig, ax = plt.subplots(figsize=(7, 5))
    
    colors = ['red', 'green', 'blue']
    labels = ['x', 'y', 'z']
    
    for i, (color, lbl) in enumerate(zip(colors, labels)):
        ax.plot(t_gt, err_mm[:, i], color=color, linestyle='-', linewidth=1.5, label=f'Error_{lbl}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (mm)')
    ax.set_title(f'{trajectory_name}\n{scenario_name}\nPosition Error: GT vs MJM')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.8)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'error_{trajectory_name}.png'
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
        print(f"  -> Saved: {filename}")
    #plt.close('all')
    plt.show()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load danh sách file test
    test_list_path = os.path.join(TEST_FOLDER, TEST_FILE_LIST)
    with open(test_list_path, 'r') as f:
        test_files = json.load(f)

    print(f"=== So sanh Ground Truth vs MJM (PAPER) ===")
    print(f"  t_f = {T_F}s, dt = {DT:.6f}s")
    print(f"  Equation: p(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5")
    print(f"  Num trajectories: {len(test_files)}")
    print()

    summary = []

    for fname in test_files:
        filepath = os.path.join(TEST_FOLDER, fname)
        trajectory_name = os.path.splitext(fname)[0]

        if not os.path.exists(filepath):
            print(f"  [SKIP] {fname} not found")
            continue

        # 1. Doc ground truth
        gt_coords, scenario_id = read_ground_truth(filepath)
        
        # --- Cắt phần dư thừa dựa trên vận tốc (< 5 cm/s) ---
        velocities = np.linalg.norm(np.diff(gt_coords, axis=0), axis=1) / DT
        threshold = 0.05
        below_threshold = velocities < threshold
        if not np.all(below_threshold):
            last_active_frame = np.where(~below_threshold)[0][-1]
            n_keep = last_active_frame + 2
            gt_coords = gt_coords[:n_keep]

        n_gt = len(gt_coords)
        t_gt = np.arange(n_gt) * DT
        real_t_f = (n_gt - 1) * DT

        scenario_name = SCENARIO_NAMES.get(scenario_id, f"Unknown_Scenario_{scenario_id}")

        # 2. Diem dau va diem cuoi lay truc tiep tu quy dao (đã rút gọn)
        x_0  = gt_coords[0].copy()
        x_f  = gt_coords[-1].copy()
        dist = np.linalg.norm(x_f - x_0)

        print(f"---- {trajectory_name} ----")
        print(f"  X_0 = [{x_0[0]:.4f}, {x_0[1]:.4f}, {x_0[2]:.4f}]")
        print(f"  X_f = [{x_f[0]:.4f}, {x_f[1]:.4f}, {x_f[2]:.4f}]")
        print(f"  Distance = {dist*100:.2f} cm,  N_steps (GT) = {n_gt}, real t_f = {real_t_f:.2f}s")

        # 3. Sinh quỹ đạo MJM
        t_mjm, mjm_traj = generate_mjm_trajectory(x_0, x_f, real_t_f, DT)

        # 4. Resample MJM về cùng số điểm với ground truth để tính error
        #    Ground truth có n_gt điểm (mỗi điểm cách nhau DT)
        #    MJM có len(t_mjm) điểm cũng cách nhau DT
        #    → Lấy min(n_gt, len(mjm)) điểm để so sánh
        n_compare = min(n_gt, len(mjm_traj))
        gt_compare = gt_coords[:n_compare]
        mjm_compare = mjm_traj[:n_compare]
        t_compare = t_gt[:n_compare]

        # 5. Tính sai số
        errors = np.linalg.norm(gt_compare - mjm_compare, axis=1)
        mean_err = np.mean(errors) * 1000  # mm
        max_err = np.max(errors) * 1000
        final_err = np.linalg.norm(gt_coords[-1] - mjm_traj[-1]) * 1000

        print(f"  Error: mean={mean_err:.1f} mm, max={max_err:.1f} mm, final={final_err:.1f} mm")

        summary.append({
            'trajectory': trajectory_name,
            'distance_cm': dist * 100,
            'mean_error_mm': mean_err,
            'max_error_mm': max_err,
            'final_error_mm': final_err,
        })

        # 6. Đọc CSV simulation tương ứng (nếu có)
        sim_df = read_sim_trajectory(trajectory_name)

        # 7. Ve do thi
        plot_comparison(t_gt, gt_coords, t_mjm, mjm_traj, trajectory_name, scenario_name,
                        sim_df=sim_df, save_dir=SAVE_DIR)
        plot_error(t_compare, gt_compare, mjm_compare, trajectory_name, scenario_name,
                   save_dir=SAVE_DIR)
        # 7b. Xuất hình 3D riêng
        plot_3d_trajectory(gt_coords, trajectory_name, scenario_name, sim_df=sim_df, save_dir=SAVE_DIR)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=== SUMMARY ===")
    print(f"{'Trajectory':<20} {'Dist(cm)':>9} {'Mean(mm)':>9} {'Max(mm)':>8} {'Final(mm)':>10}")
    print("-" * 62)
    for s in summary:
        print(f"{s['trajectory']:<20} {s['distance_cm']:>9.2f} "
              f"{s['mean_error_mm']:>9.1f} {s['max_error_mm']:>8.1f} "
              f"{s['final_error_mm']:>10.1f}")

    if summary:
        mean_all = np.mean([s['mean_error_mm'] for s in summary])
        max_all  = np.max([s['max_error_mm'] for s in summary])
        print("-" * 62)
        print(f"{'AVERAGE':<20} {'':>9} {mean_all:>9.1f} {max_all:>8.1f}")

    print(f"\n-> Plots saved to: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
