import os
import subprocess
import re

run_sim = r"run_simulation.py"
outer_loop = r"outer_loop.py"
inner_loop = r"inner_loop.py"

configs = [
    # (Phi Mode, Kp_F, Kp_L)
    ("sync", 400, 400),
    ("sync", 1600, 1600),
    ("sync", 400, 1600),
    ("unsync", 400, 400),
    ("unsync", 1600, 1600),
    ("unsync", 400, 1600),
]

def replace_in_file(filepath, pattern, replacement):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(pattern, replacement, content)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

summary_results = []

for idx, (phi_mode, kp_f, kp_l) in enumerate(configs):
    dir_name = f"output_{phi_mode}_Kp_{kp_f}_{kp_l}"
    print(f"\n[{idx+1}/6] Running Config: Phi={phi_mode}, KP_F={kp_f}, KP_L={kp_l}")
    
    # Update SAVE_DIR
    replace_in_file(run_sim, 
                    r'SAVE_DIR\s*=\s*os\.path\.join\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\),\s*".*?"\)', 
                    f'SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{dir_name}")')

    # Update Phi
    if phi_mode == "sync":
        replace_in_file(outer_loop,
                        r"phi\s*=\s*MatlabBridge\.compute_disagreement\(.*?, f_r_raw\)",
                        r"phi = MatlabBridge.compute_disagreement(f_h_raw, f_r_raw)")
    else:
        replace_in_file(outer_loop,
                        r"phi\s*=\s*MatlabBridge\.compute_disagreement\(.*?, f_r_raw\)",
                        r"phi = MatlabBridge.compute_disagreement(f_h, f_r_raw)")
                        
    # Update K_p
    replace_in_file(inner_loop, r"self\.k_p_follower\s*=\s*\d+\.\d+", f"self.k_p_follower = {kp_f}.0")
    replace_in_file(inner_loop, r"self\.k_p_leader\s*=\s*\d+\.\d+", f"self.k_p_leader   = {kp_l}.0")
    
    # Run the simulation
    log_name = f"log_{dir_name}.txt"
    with open(log_name, 'w', encoding='utf-8') as log_file:
        subprocess.run(["python", run_sim], stdout=log_file, stderr=subprocess.STDOUT)
    
    # Parse results from log
    MAE = 0.0
    total_conflicts = 0
    with open(log_name, 'r', encoding='utf-8', errors='ignore') as log_file:
        content = log_file.read()
        
        # Find conflicts
        import re
        conflicts = re.findall(r"Conflict steps \(Phi < 0\.5\):\s*(\d+)", content)
        total_conflicts = sum(int(c) for c in conflicts)
        
        # Find Overall Average MAE
        match_mae = re.search(r"Overall Average MAE:\s*([\d\.]+)", content)
        if match_mae:
            MAE = float(match_mae.group(1))

    summary = {
        "Config": f"{phi_mode.upper()} | Kp_F: {kp_f} | Kp_L: {kp_l}",
        "MAE": MAE,
        "Total Conflicts": total_conflicts
    }
    summary_results.append(summary)
    print(f"      -> MAE: {MAE:.4f} m | Total Conflicts: {total_conflicts}")

# Restore variables
replace_in_file(outer_loop, r"phi = MatlabBridge\.compute_disagreement\(.*?, f_r_raw\)", r"phi = MatlabBridge.compute_disagreement(f_h_raw, f_r_raw)")
replace_in_file(inner_loop, r"self\.k_p_follower\s*=\s*\d+\.\d+", "self.k_p_follower = 400.0")
replace_in_file(inner_loop, r"self\.k_p_leader\s*=\s*\d+\.\d+", "self.k_p_leader   = 1600.0")

print("\n" + "="*60)
print("  FINAL BATCH SUMMARY")
print("="*60)
print(f"{'Configuration':<35} | {'Overall MAE':<12} | {'Total Conflicts'}")
print("-" * 60)
for s in summary_results:
    print(f"{s['Config']:<35} | {s['MAE']:<12.4f} | {s['Total Conflicts']}")
