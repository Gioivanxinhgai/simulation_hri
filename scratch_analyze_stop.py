import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

folder = r'd:\LAB\Experiment_Train_Trajectory_HRI'
sigma = 0.2

# Phân tích tất cả 306 file
all_files = sorted([f for f in os.listdir(folder) if f.startswith('trajectory_') and f.endswith('.csv')])

results = []
for fname in all_files:
    df = pd.read_csv(os.path.join(folder, fname))
    x = gaussian_filter1d(df['X'].values, sigma=sigma)
    y = gaussian_filter1d(df['Y'].values, sigma=sigma)
    z = gaussian_filter1d(df['Z'].values, sigma=sigma)
    
    dt = 12.0 / 190  # 191 points, 190 intervals
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    vz = np.diff(z) / dt
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Tìm step đầu tiên mà "đứng im liên tiếp >= 10 bước"
    threshold = 0.01  # m/s
    stopped = speed < threshold
    stop_start = None
    for j in range(len(stopped) - 9):
        if np.all(stopped[j:j+10]):
            stop_start = j
            break
    
    t_stop = stop_start * dt if stop_start is not None else -1
    pct_active = stop_start / 190 * 100 if stop_start is not None else 100
    
    results.append({
        'file': fname, 
        'stop_step': stop_start,
        't_stop': t_stop,
        'pct_active': pct_active,
        'speed_last_10': speed[-10:].mean()
    })

# Thống kê
n_total = len(results)
n_stopped = sum(1 for r in results if r['stop_step'] is not None)
n_active_full = n_total - n_stopped

print(f"=== THONG KE QUY DAO (n={n_total}) ===\n")
print(f"Quy dao co giai doan 'dung im' (speed<0.01 m/s lien tiep 10 buoc): {n_stopped}/{n_total}")
print(f"Quy dao KHONG dung im trong 12s:                                   {n_active_full}/{n_total}")
print()

stopped_results = [r for r in results if r['stop_step'] is not None]
if stopped_results:
    t_stops = [r['t_stop'] for r in stopped_results]
    pct_actives = [r['pct_active'] for r in stopped_results]
    
    print(f"Trong so nhung quy dao co 'dung im':")
    print(f"  Thoi diem bat dau dung im (trung binh): {np.mean(t_stops):.2f}s")
    print(f"  Thoi diem bat dau dung im (min):        {np.min(t_stops):.2f}s")
    print(f"  Thoi diem bat dau dung im (max):        {np.max(t_stops):.2f}s")
    print(f"  % quy dao 'hoat dong' (trung binh):     {np.mean(pct_actives):.1f}%")
    print()
    
    # Histogram
    bins = [0, 2, 4, 6, 8, 10, 12]
    hist, _ = np.histogram(t_stops, bins=bins)
    print("  Phan bo thoi diem dung im:")
    for i in range(len(bins)-1):
        bar = '#' * hist[i]
        print(f"    {bins[i]:2.0f}s - {bins[i+1]:2.0f}s: {hist[i]:3d}  {bar}")

# In 10 quy dao dien hinh
print(f"\n--- 10 quy dao dung im SOM nhat ---")
stopped_results.sort(key=lambda r: r['t_stop'])
for r in stopped_results[:10]:
    print(f"  {r['file']}: dung im tu t={r['t_stop']:.1f}s ({r['pct_active']:.0f}% active)")

# Van toc trung binh 10 buoc cuoi cua TAT CA quy dao
mean_v_last = np.mean([r['speed_last_10'] for r in results])
print(f"\nVan toc trung binh 10 buoc cuoi (tat ca): {mean_v_last:.4f} m/s")
