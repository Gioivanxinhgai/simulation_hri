import os
import json

import numpy as np
import pandas as pd

from config import get_save_dir

SAVE_DIR = get_save_dir()

from shared_control_lib import (
    load_gmm_system, load_svgp_system, read_and_smooth_3d,
    plot_simulation_results,
    GMM_MODEL_PATH, GMM_SCALER_PATH, SVGP_CHECKPOINT_DIR,
    TEST_FOLDER, GROUND_TRUTH, GOALS, SIGMA,
    ControlMode, DT, calculate_and_plot_hri_metrics, plot_phi, plot_xd_vs_xr
)
from config import SCENARIO_NAMES

from outer_loop import LocalPythonBridge, run_control_loop

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 70)
    print(" pHRI SHARED CONTROL")
    print(f"  Output directory: {SAVE_DIR}")
    print("=" * 70)

    # ── 1. Load Models ──────────────────────────────────────────────────
    print("\n[*] Loading models...")
    gmm_models, gmm_scaler = load_gmm_system(GMM_MODEL_PATH, GMM_SCALER_PATH)
    svgp_scaler_x, svgp_scaler_y, svgp_model = load_svgp_system(SVGP_CHECKPOINT_DIR)

    if svgp_model is None:
        print("[!] SVGP model not found. Exiting.")
        return

    print(f"   [OK] GMM: {len(gmm_models)} targets")
    print(f"   [OK] SVGP: loaded")

    # ── 2. Khởi tạo Local Python Dynamics ──────────────────────────────────
    bridge = LocalPythonBridge()
    bridge.connect()

    # ── 3. Load test files ──────────────────────────────────────────────
    file_list_path = os.path.join(TEST_FOLDER, "test_file_list.json")
    if not os.path.exists(file_list_path):
        print(f"[!] Cannot find {file_list_path}")
        return
        
    with open(file_list_path, 'r') as f:
        file_names = json.load(f)

    print(f"\n[*] Found {len(file_names)} test files")
    print("-" * 70)

    all_metrics = []

    for i, test_file in enumerate(file_names):
        print(f"\n[{i+1}/{len(file_names)}] {test_file}")

        test_path = os.path.join(TEST_FOLDER, test_file)
        df, trajectory_name = read_and_smooth_3d(test_path, SIGMA)

        if df is None:
            continue
        test_data  = df[['X', 'Y', 'Z']].values

        # Tính n_keep: cắt phần đuôi đứng im (vận tốc < 5 cm/s) — giống outer_loop
        _vel = np.linalg.norm(np.diff(test_data, axis=0), axis=1) / DT
        _below = _vel < 0.0
        if not np.all(_below):
            _last_active = int(np.where(~_below)[0][-1])
            n_keep = _last_active + 2
        else:
            n_keep = len(test_data)

        # Lấy ScenarioId từ cột CSV (không phải từ tên file)
        if 'ScenarioId' in df.columns:
            scenario_id = int(df['ScenarioId'].iloc[0])
        else:
            try:
                scenario_id = int(trajectory_name.split('_')[-1])
            except:
                scenario_id = i

        true_target = GROUND_TRUTH.get(scenario_id)
        scenario_name = SCENARIO_NAMES.get(scenario_id, f"Scenario_{scenario_id}")

        if hasattr(bridge, 'reset'):
            bridge.reset(test_data[0])

        results = run_control_loop(
            test_data,
            gmm_models, gmm_scaler,
            svgp_model, svgp_scaler_x, svgp_scaler_y,
            bridge=bridge,
            goals=GOALS,
            true_target_id=true_target
        )

        # ── Metrics ─────────────────────────────────────────────────────
        x_ref_traj = results['x_ref_history']
        gt         = test_data[:len(x_ref_traj)]
        errors     = x_ref_traj - gt
        mae        = np.mean(np.sqrt(np.sum(errors**2, axis=1)))
        rmse       = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

        phi_arr    = np.asarray(results['phi_history'])

        print(f"   MAE: {mae:.4f} m | RMSE: {rmse:.4f} m")

        # Vẽ đồ thị quỹ đạo (Robot vs Human)
        plot_simulation_results(test_data, results, trajectory_name, true_target, save_dir=SAVE_DIR)
        
        # Plot 1: x_ref (outer loop) vs Ground Truth (Human CSV)
        plot_xd_vs_xr(
            results['x_ref_history'], test_data, results['modes'], trajectory_name,
            save_dir=SAVE_DIR,
            label_d='$x_{ref}$ (Outer Loop)',
            label_r='Ground Truth ($x_H$)',
            title_prefix='Outer Loop: $x_{ref}$ vs Human Ground Truth',
            filename_prefix='xref_vs_gt',
            n_keep=n_keep,
            scenario_name=scenario_name,
        )

        # Plot 2: x_d (PD desired) vs x_r (actual robot) — đánh giá PD tracking
        if 'x_d_history' in results:
            plot_xd_vs_xr(
                results['x_d_history'], results['robot_trajectory'], results['modes'], trajectory_name,
                save_dir=SAVE_DIR,
                label_d='$x_d$ (PD Desired = $x_{ref}+x_{adm}$)',
                label_r='$x_r$ (Actual Robot)',
                title_prefix='Inner Loop PD Tracking: $x_d$ vs $x_r$',
                filename_prefix='xd_vs_xr'
            )
        
        # Vẽ đồ thị Disagreement Phi
        plot_phi(results['phi_history'], trajectory_name, save_dir=SAVE_DIR)

        # ── TÍNH TOÁN VÀ VẼ HRI METRICS ─────────────────────────────────
        mean_theta, mean_assist = calculate_and_plot_hri_metrics(
            results['f_h_history'],
            results['f_r_history'],
            DT,
            trajectory_name,
            modes=results.get('modes'),
            save_dir=SAVE_DIR,
            assist_energy_history=results.get('assist_energy_history'),
        )

        # ── XUẤT DỮ LIỆU RA CSV ─────────────────────────────────────────
        csv_dir = os.path.join(SAVE_DIR, "csv_logs")
        os.makedirs(csv_dir, exist_ok=True)
        
        n_len = len(results['modes'])
        time_s = np.arange(n_len) * DT
        
        # Chuyển đổi Phi (cos) sang Theta (độ)
        theta_degrees = np.arccos(np.clip(results['phi_history'], -1.0, 1.0)) * (180.0 / np.pi)
        
        # Sắp xếp thứ tự các cột theo logic luồng dữ liệu (Data Flow)
        df_log = pd.DataFrame({
            'time_s': time_s,
            't_fitts': results['t_fitts_remaining_history'][:n_len],
            'mode': results['modes'],
            'max_probs': results['max_probs'],
            
            # 1. Ý định của người (Ground truth)
            'x_h_X': test_data[:n_len, 0],
            'x_h_Y': test_data[:n_len, 1],
            'x_h_Z': test_data[:n_len, 2],
            
            # 2. Ý định của hệ thống AI
            'x_ref_X': results['x_ref_history'][:, 0],
            'x_ref_Y': results['x_ref_history'][:, 1],
            'x_ref_Z': results['x_ref_history'][:, 2],
            
            # 3. Đích đến đã qua Admittance
            'x_d_X': results['x_d_history'][:, 0],
            'x_d_Y': results['x_d_history'][:, 1],
            'x_d_Z': results['x_d_history'][:, 2],
            
            # 4. Vị trí thực tế của Robot
            'x_r_X': results['robot_trajectory'][:, 0],
            'x_r_Y': results['robot_trajectory'][:, 1],
            'x_r_Z': results['robot_trajectory'][:, 2],
            
            # 5. Lực tương tác sinh ra
            'F_h_X': results['f_h_history'][:, 0],
            'F_h_Y': results['f_h_history'][:, 1],
            'F_h_Z': results['f_h_history'][:, 2],
            'F_r_X': results['f_r_history'][:, 0],
            'F_r_Y': results['f_r_history'][:, 1],
            'F_r_Z': results['f_r_history'][:, 2],
            
            # 6. Đánh giá xung đột
            'theta_deg': theta_degrees,
        })
        df_log.to_csv(os.path.join(csv_dir, f"{trajectory_name}.csv"), index=False)

        # Lưu lại metrics cho bảng summary cuối cùng
        all_metrics.append({
            'file': test_file, 'trajectory_name': trajectory_name,
            'true_target': true_target,
            'mae': mae, 'rmse': rmse,
            'mean_theta': mean_theta,
            'mean_assist': mean_assist
        })

    # ── Kết thúc ────────────────────────────────────────────────────────
    bridge.disconnect()

    summary_lines = []
    summary_lines.append("\n" + "=" * 90)
    summary_lines.append("  OVERALL SUMMARY")
    summary_lines.append("=" * 90)
    summary_lines.append(f"  {'File':<25} {'MAE (m)':<10} {'Theta (deg)':<15} {'Assist Index (N)':<15}")
    summary_lines.append("-" * 90)
    for m in all_metrics:
        summary_lines.append(f"  {m['trajectory_name']:<25} {m['mae']:<10.4f} {m['mean_theta']:<15.2f} {m['mean_assist']:<15.4f}")

    overall_mae = np.mean([m['mae'] for m in all_metrics]) if all_metrics else 0
    summary_lines.append(f"\n  Overall Average MAE: {overall_mae:.4f} m")
    summary_lines.append("[OK] All simulations completed.")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(SAVE_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()