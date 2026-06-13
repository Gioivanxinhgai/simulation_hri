import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Thiết lập font Times New Roman cho đồ thị
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12 #title
plt.rcParams['axes.labelsize'] = 12 #2 trục x và y
plt.rcParams['xtick.labelsize'] = 12 #các số trên trục x
plt.rcParams['ytick.labelsize'] = 12 #các số trên trục y
plt.rcParams['legend.fontsize'] = 10 #chú thích

def plot_trajectories_3d_from_json(json_file_path):
    try:
        # 1. Mở và đọc tệp JSON
        with open(json_file_path, 'r') as f:
            trajectory_files = json.load(f)

        # Khởi tạo biểu đồ 3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Tất cả các quỹ đạo 3D')
        ax.set_xlabel('Tọa độ X')
        ax.set_ylabel('Tọa độ Y')
        ax.set_zlabel('Tọa độ Z')

        # 2. Vòng lặp qua từng tên tệp trong danh sách
        for file_name in trajectory_files:
            try:
                # Tạo đường dẫn đầy đủ đến file CSV
                json_dir = os.path.dirname(json_file_path)
                csv_path = os.path.join(json_dir, file_name)
                
                # 3. Đọc dữ liệu từ tệp CSV bằng pandas
                df = pd.read_csv(csv_path)

                # Kiểm tra xem các cột 'X', 'Y', 'Z' có tồn tại không
                if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
                    # 4. Vẽ quỹ đạo 3D
                    ax.plot(df['X'], df['Y'], df['Z'])
                else:
                    print(f"Lỗi: Tệp {csv_path} không chứa các cột 'X', 'Y' hoặc 'Z'.")

            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy tệp {csv_path}. Vui lòng kiểm tra đường dẫn.")
            except Exception as e:
                print(f"Lỗi khi xử lý tệp {csv_path}: {e}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {json_file_path}. Vui lòng kiểm tra đường dẫn.")
    except Exception as e:
        print(f"Lỗi khi xử lý tệp JSON: {e}")

def plot_trajectories_components_from_json(json_file_path):
    try:
        # 1. Mở và đọc tệp JSON
        with open(json_file_path, 'r') as f:
            trajectory_files = json.load(f)

        # Khởi tạo biểu đồ subplots (3 hàng, 1 cột)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('Quỹ đạo theo từng trục (X, Y, Z) theo thời gian')

        axs[0].set_ylabel('Tọa độ X')
        axs[0].grid(True)
        
        axs[1].set_ylabel('Tọa độ Y')
        axs[1].grid(True)
        
        axs[2].set_ylabel('Tọa độ Z')
        axs[2].set_xlabel('Timestamp')
        axs[2].grid(True)

        # 2. Vòng lặp qua từng tên tệp trong danh sách
        for file_name in trajectory_files:
            try:
                json_dir = os.path.dirname(json_file_path)
                csv_path = os.path.join(json_dir, file_name)
                df = pd.read_csv(csv_path)

                # Kiểm tra cột
                if all(col in df.columns for col in ['Timestamp', 'X', 'Y', 'Z']):
                    axs[0].plot(df['Timestamp'], df['X']) # Không dùng label để tránh rối nếu quá nhiều file
                    axs[1].plot(df['Timestamp'], df['Y'])
                    axs[2].plot(df['Timestamp'], df['Z'])
                else:
                    # Fallback nếu không có Timestamp
                    if all(col in df.columns for col in ['X', 'Y', 'Z']):
                         axs[0].plot(df['X'])
                         axs[1].plot(df['Y'])
                         axs[2].plot(df['Z'])

            except Exception as e:
                print(f"Lỗi khi xử lý tệp {csv_path}: {e}")
                
    except Exception as e:
        print(f"Lỗi khi xử lý tệp JSON: {e}")

# Triển khai mã
if __name__ == '__main__':
    DATA_DIR = "Experiment_Trajectory_HRI"
    TRAIN_LIST = os.path.join(DATA_DIR, "train_file_list.json")
    
    print(f"Đang vẽ dữ liệu từ danh sách: {TRAIN_LIST}")
    
    # Vẽ 3D
    plot_trajectories_3d_from_json(TRAIN_LIST)
    
    # Vẽ Components
    plot_trajectories_components_from_json(TRAIN_LIST)
    
    plt.show()