import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── CẤU HÌNH VÀ DỮ LIỆU ĐẦU VÀO ─────────────────────────────────────────
FOLDER = "Test_Results_GMM_switch"
# Tên file phải khớp chính xác với tên file được upload
FILE_MAP = {
    "GMM + SVGP (unweight)": ("gmm_svgp_unweight.csv", 'blue'),
    "GMM + SVGP (weight)": ("gmm_svgp_weight.csv", 'orange')
}

# Thiết lập font Times New Roman cho đồ thị
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# ─── HÀM XỬ LÝ ──────────────────────────────────────────────────────────
def calculate_mean_and_ci(df):
    N = df.shape[1]
    
    # Tính Mean và Std Dev theo hàng (qua các quỹ đạo tại mỗi mốc thời gian)
    mean_accuracy = df.mean(axis=1)
    std_accuracy = df.std(axis=1)
    
    # Tính Standard Error (SE): SE = StdDev / sqrt(N)
    se = std_accuracy / np.sqrt(np.maximum(N, 1))
    
    # Tính 95% CI (Z=1.96 cho mẫu lớn)
    ci_margin = 1.96 * se
    
    lower_bound = mean_accuracy - ci_margin
    upper_bound = mean_accuracy - ci_margin
    
    return mean_accuracy, lower_bound, upper_bound, N

# ─── MAIN ───────────────────────────────────────────────────────────────

data_to_plot = {}

print(f"Bắt đầu phân tích và tính toán Mean Accuracy và 95% CI từ thư mục: {FOLDER}")

for label, (filename, color) in FILE_MAP.items():
    file_path = filename 

    try:
        # Load CSV, giả định cột đầu tiên là index 't(s)'
        df = pd.read_csv(file_path, index_col=0)
        
        # Xử lý các giá trị NaN (Nếu quỹ đạo ngắn hơn, ta loại bỏ các cột đó)
        df.dropna(axis=1, how='all', inplace=True)
        
        if df.empty or df.shape[1] == 0:
            print(f"Cảnh báo: File {filename} trống hoặc không có quỹ đạo hợp lệ sau khi loại bỏ NaN.")
            continue
            
        mean_acc, lower_ci, upper_ci, N = calculate_mean_and_ci(df)
        
        data_to_plot[label] = {
            'mean': mean_acc,
            'lower': lower_ci,
            'upper': upper_ci,
            'color': color,
            'N': N
        }
        print(f"Đã xử lý file {filename}: {N} quỹ đạo, {df.shape[0]} mốc thời gian.")
        
    except FileNotFoundError:
        # Nếu file không tìm thấy ở thư mục gốc, thử kết hợp với FOLDER
        try:
            file_path_in_folder = os.path.join(FOLDER, filename)
            df = pd.read_csv(file_path_in_folder, index_col=0)
            df.dropna(axis=1, how='all', inplace=True)
            
            if df.empty or df.shape[1] == 0:
                print(f"Cảnh báo: File {filename} trống hoặc không có quỹ đạo hợp lệ sau khi loại bỏ NaN.")
                continue

            mean_acc, lower_ci, upper_ci, N = calculate_mean_and_ci(df)
            
            data_to_plot[label] = {
                'mean': mean_acc,
                'lower': lower_ci,
                'upper': upper_ci,
                'color': color,
                'N': N
            }
            print(f"Đã xử lý file {file_path_in_folder}: {N} quỹ đạo, {df.shape[0]} mốc thời gian.")

        except FileNotFoundError:
             print(f"Lỗi: Không tìm thấy file {filename} cả trong thư mục gốc lẫn {FOLDER}/.")
        except Exception as e:
            print(f"Lỗi khi đọc hoặc xử lý file {filename} trong {FOLDER}: {e}")
    except Exception as e:
        print(f"Lỗi khi đọc hoặc xử lý file {filename}: {e}")

# ─── VẼ ĐỒ THỊ ──────────────────────────────────────────────────────────
if data_to_plot:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for label, data in data_to_plot.items():
        time_steps = data['mean'].index
        color = data['color']
        
        # Vẽ đường Mean Accuracy
        ax.plot(time_steps, data['mean'], color=color, linewidth=1.5, marker = 'o', 
                label=f"{label} (N={data['N']})")
        
        # Vẽ 95% CI (Confidence Interval)
        # ax.fill_between(time_steps, data['lower'], data['upper'], color=color, alpha=0.15, label='_nolegend_')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.1)
    
    # Thiết lập mốc x chẵn
    try:
        if len(time_steps) > 0:
             # Lấy mốc thời gian từ dữ liệu đầu tiên (giả sử tất cả đều giống nhau)
             time_steps_list = [float(t) for t in time_steps]
             if all(t.is_integer() for t in time_steps_list):
                 ax.set_xticks(time_steps_list)
    except:
        pass 

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.6)
    
    # CÁC CHỈNH SỬA MỚI ĐỂ HÌNH VẼ ÔM SÁT HƠN
    ax.autoscale(enable=True, axis='x', tight=True) # Điều chỉnh trục X sát dữ liệu
    ax.margins(x=0.0) # Loại bỏ margin trục X (thời gian)
    
    fig.tight_layout() # Loại bỏ khoảng trắng thừa quanh viền hình
    plt.show()
    
else:
    print("\nKhông có dữ liệu hợp lệ để vẽ đồ thị.")

print("\nHoàn tất xử lý.")