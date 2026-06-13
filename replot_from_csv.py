

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ─── Cấu hình ────────────────────────────────────────────────────────────────

SIM_DIR   = r"D:\LAB\MJM_CURRENT_10_10_60_5_5_0.75_3_150_1000000_1400.00_GTCHEAT"

# Ground-truth trajectory folder
GT_FOLDER = "Experiment_Test_Trajectory_HRI"

# ─── Import hàm từ shared_control_lib ─────────────────────────────────────
from config import (
    SCENARIO_NAMES
)
from shared_control_lib import add_control_mode_background, control_mode_legend_patches

# ─── Tiện ích ─────────────────────────────────────────────────────────────────

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 12
plt.rcParams['savefig.dpi'] = 300  # Đảm bảo nút Save trong GUI lưu ảnh né

def _mode_list(df):
    """Chuyển cột 'mode' thành list chuỗi khớp với ControlMode."""
    return list(df['mode'].values)

def _find_target2_idx(gt_traj):
    """Tìm chỉ số của Target 2: điểm cực đại Y (điểm ngoặt quỹ đạo)."""
    return int(np.argmax(gt_traj[:, 1]))

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
        # Đoạn [i, j+1) để nối liền với điểm bắt đầu của mode tiếp theo
        seg = x_ref[i:min(j+1, n)]
        color = MODE_COLOR.get(cur_mode, 'gray')
        lbl = MODE_LABEL.get(cur_mode, cur_mode) if cur_mode not in plotted else '_nolegend_'
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                color=color, linewidth=1.5, linestyle='--', label=lbl, alpha=0.9)
        plotted.add(cur_mode)
        i = j

def plot_3d_trajectory_from_csv(df, trajectory_name, scenario_name, save_dir=None):
    """Xuất hình 3D độc lập với X_0, Target 2, Target 3 được đánh dấu."""
    
    x_h = df[['x_h_X', 'x_h_Y', 'x_h_Z']].values   # ground truth position
    x_ref = df[['x_ref_X', 'x_ref_Y', 'x_ref_Z']].values
    modes = _mode_list(df)

    if len(x_h) == 0:
        return

    fig3d = plt.figure(figsize=(8, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')

    # ── Quỹ đạo Ground Truth – màu ĐEN ──────────────────────────────────────
    ax3d.plot(x_h[:, 0], x_h[:, 1], x_h[:, 2],
              color='black', linewidth=1.8, label='Ground Truth', alpha=0.85)

    # ── x_ref từ simulation (tô màu theo mode) ──────────────────────────────
    _plot_xref_3d(ax3d, x_ref, modes)

    # ── Điểm đặc biệt ────────────────────────────────────────────────────────
    # X_0 – Start
    ax3d.scatter(*x_h[0], c='green', s = 80, marker='o', zorder = 3, label='$X_0$ (Start)')

    # Target 2 – điểm cực đại Y
    t2_idx = _find_target2_idx(x_h)
    t2 = x_h[t2_idx]
    ax3d.scatter(*t2, c='orange', s=120, marker='o', zorder=3,
                 label=f'Target 2 [{t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f}]')
    ax3d.text(t2[0], t2[1], t2[2] - 0.006, 'Target 2', color='orange',
              fontsize=9, fontweight='bold', ha='center', va='top')

    # Target 3 – điểm cuối X_f
    t3 = x_h[-1]
    ax3d.scatter(*t3, c='red', s=120, marker='X', zorder = 3,
                 label=f'Target 3 [$X_f$: {t3[0]:.4f}, {t3[1]:.4f}, {t3[2]:.4f}]')
    ax3d.text(t3[0], t3[1], t3[2] - 0.006, 'Target 3', color='red',
              fontsize=9, fontweight='bold', ha='center', va='top')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.grid(False)
    if scenario_name:
        ax3d.set_title(f'3D Trajectory\n{trajectory_name} | {scenario_name}', fontsize=11, fontweight='bold')
    else:
        ax3d.set_title(f'3D Trajectory\n{trajectory_name}', fontsize=11, fontweight='bold')
    ax3d.legend(fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig3d.savefig(os.path.join(save_dir, f'3d_traj_{trajectory_name}.png'), dpi=150)
        plt.close(fig3d)
    else:
        plt.show()

def plot_xref_vs_gt_3axes_from_csv(df, trajectory_name, scenario_name, save_dir=None):
    """Vẽ đồ thị so sánh x_ref và Ground Truth trên 3 trục X, Y, Z."""
    time_s = df['time_s'].values
    x_h = df[['x_h_X', 'x_h_Y', 'x_h_Z']].values
    x_ref = df[['x_ref_X', 'x_ref_Y', 'x_ref_Z']].values
    modes = _mode_list(df)

    if len(x_h) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    coords = ['X', 'Y', 'Z']

    # Lấy tọa độ Target 2 để chú thích
    t2_idx = _find_target2_idx(x_h)
    t2 = x_h[t2_idx]
    t2_time = time_s[t2_idx]

    for i, ax in enumerate(axes):
        # 1. Vẽ nền FOLLOWER/LEADER
        add_control_mode_background(ax, modes, time_s, zorder=0)

        # 2. Vẽ Ground truth (màu đen)
        ax.plot(time_s, x_h[:, i], 'k-', linewidth=1.5, label='Ground Truth ($x_h$)', zorder=2)
        
        # 3. Vẽ x_ref (chỉ vẽ 1 đường màu đỏ nét đứt)
        ax.plot(time_s, x_ref[:, i], color='#C62828', linewidth=1.5, linestyle='--', label='$x_{ref}$', zorder=2)

        # 4. Đánh dấu điểm Target 2 (cùng thời điểm đạt max Y)
        ax.scatter(t2_time, t2[i], color='orange', marker='o', s=80, zorder=3, 
                   label=f'Target 2')
        
        # 5. Đánh dấu điểm Target 3 (cuối quỹ đạo)
        ax.scatter(time_s[-1], x_h[-1, i], color='red', marker='X', s=80, zorder=3,
                   label=f'Target 3')
        
        ax.set_ylabel(f'{coords[i]} (m)')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
            if scenario_name:
                ax.set_title(f'$x_{{ref}}$ vs Ground Truth (3 Axes)\n{trajectory_name}  |  {scenario_name}', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'$x_{{ref}}$ vs Ground Truth (3 Axes) - {trajectory_name}', fontsize=12, fontweight='bold')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'3axes_{trajectory_name}.png'), dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    csv_dir = os.path.join(SIM_DIR, "csv_logs")
    if not os.path.isdir(csv_dir):
        print(f"[ERROR] Không tìm thấy thư mục: {csv_dir}")
        return

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        print(f"[ERROR] Không có file CSV nào trong: {csv_dir}")
        return

    print(f"=== Replot from CSV ===")
    print(f"  Source : {os.path.abspath(SIM_DIR)}")
    print(f"  Files  : {len(csv_files)} trajectories\n")

    for csv_path in csv_files:
        trajectory_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"--- {trajectory_name} ---")

        df = pd.read_csv(csv_path)
        if df.empty:
            print("  [SKIP] File rỗng")
            continue

        # Lấy scenario_name nếu có cột ScenarioId trong CSV
        scenario_name = None
        if 'ScenarioId' in df.columns:
            sid = int(df['ScenarioId'].iloc[0])
            scenario_name = SCENARIO_NAMES.get(sid, f"Scenario_{sid}")
        else:
            # Thử map từ GT folder nếu có
            gt_path = os.path.join(GT_FOLDER, f"{trajectory_name}.csv")
            if os.path.exists(gt_path):
                try:
                    gt_df = pd.read_csv(gt_path)
                    if 'ScenarioId' in gt_df.columns:
                        sid = int(gt_df['ScenarioId'].iloc[0])
                        scenario_name = SCENARIO_NAMES.get(sid, f"Scenario_{sid}")
                except Exception:
                    pass

        # Thư mục lưu cấu hình
        save_dir = r"d:\LAB\compare_mjm_plots"

        # Vẽ đồ thị 3D
        plot_3d_trajectory_from_csv(df, trajectory_name, scenario_name)
        
        # Vẽ đồ thị 3 trục
        plot_xref_vs_gt_3axes_from_csv(df, trajectory_name, scenario_name)

    print("=== Hoàn thành ===")


if __name__ == "__main__":
    main()
