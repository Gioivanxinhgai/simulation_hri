"""
Sweep K_p để tìm điểm cân bằng giữa tracking error (x_d vs x_r) và số conflict.
Chỉ chạy FIRST_N scenario đầu để nhanh.
"""
import os
import re
import subprocess

FIRST_N = 6  # Chỉ chạy 6 scenario đầu để nhanh

KP_VALUES = [200, 400, 800, 1200, 1600, 2400, 3200]

run_sim   = r"run_simulation.py"
inner_loop = r"inner_loop.py"

def replace_in_file(filepath, pattern, replacement):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(pattern, replacement, content)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def patch_run_sim_first_n(n):
    """Giới hạn run_simulation chỉ chạy n file đầu."""
    with open(run_sim, "r", encoding="utf-8") as f:
        content = f.read()
    # Patch vòng lặp file_names → file_names[:n]
    content = re.sub(
        r"for i, test_file in enumerate\(file_names\):",
        f"for i, test_file in enumerate(file_names[:{n}]):",
        content
    )
    with open(run_sim, "w", encoding="utf-8") as f:
        f.write(content)

def unpatch_run_sim():
    with open(run_sim, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(
        r"for i, test_file in enumerate\(file_names\[:\d+\]\):",
        "for i, test_file in enumerate(file_names):",
        content
    )
    with open(run_sim, "w", encoding="utf-8") as f:
        f.write(content)

results = []
patch_run_sim_first_n(FIRST_N)

try:
    for kp in KP_VALUES:
        dir_name = f"sweep_kp_{kp}"
        print(f"\n[Sweep] K_p = {kp} -> {dir_name}")

        # Cập nhật K_p
        replace_in_file(inner_loop,
                        r"self\.k_p\s*=\s*[\d\.]+",
                        f"self.k_p = {kp}.0")

        # Cập nhật SAVE_DIR
        replace_in_file(run_sim,
                        r'SAVE_DIR\s*=\s*os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\),\s*".*?"\)',
                        f'SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{dir_name}")')

        log_name = f"log_{dir_name}.txt"
        with open(log_name, 'w', encoding='utf-8') as log_file:
            subprocess.run(["python", run_sim], stdout=log_file, stderr=subprocess.STDOUT)

        # Parse
        with open(log_name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        maes = [float(x) for x in re.findall(r"MAE:\s*([\d\.]+)\s*m", content)]
        conflicts = [int(x) for x in re.findall(r"Conflict steps.*?:\s*(\d+)\s*/", content)]
        thetas = [float(x) for x in re.findall(r"Mean Theta:\s*([\d\.]+)\s*degrees", content)]

        avg_mae      = sum(maes) / len(maes) if maes else 0
        total_conf   = sum(conflicts)
        avg_theta    = sum(thetas) / len(thetas) if thetas else 0

        results.append({
            "K_p": kp,
            "avg_MAE": avg_mae,
            "total_conflicts": total_conf,
            "avg_theta": avg_theta,
        })
        print(f"  MAE: {avg_mae:.4f} m | Conflicts: {total_conf} | Theta: {avg_theta:.2f}°")

finally:
    unpatch_run_sim()
    # Restore K_p về 800 (giá trị user đang set)
    replace_in_file(inner_loop,
                    r"self\.k_p\s*=\s*[\d\.]+",
                    "self.k_p = 800.0")
    replace_in_file(run_sim,
                    r'SAVE_DIR\s*=\s*os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\),\s*".*?"\)',
                    'SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_output")')

print("\n" + "="*65)
print(f"  K_P SWEEP SUMMARY (first {FIRST_N} scenarios)")
print("="*65)
print(f"{'K_p':<8} | {'Avg MAE (m)':<14} | {'Total Conflicts':<18} | {'Avg Theta (°)'}")
print("-"*65)
for r in results:
    print(f"{r['K_p']:<8} | {r['avg_MAE']:<14.4f} | {r['total_conflicts']:<18} | {r['avg_theta']:.2f}")
