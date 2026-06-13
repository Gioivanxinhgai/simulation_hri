import pandas as pd
import os  # Thư viện giúp thao tác với thư mục và đường dẫn

def split_and_save_to_folder(input_file, folder_name):
    # 1. Đọc dữ liệu
    try:
        df = pd.read_csv(input_file)
        print(f"-> Đã đọc file gốc: '{input_file}'")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'.")
        return

    # 2. Tạo thư mục chứa file kết quả
    # exist_ok=True nghĩa là: nếu thư mục đã tồn tại thì không báo lỗi, cứ dùng tiếp.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"-> Đã tạo thư mục mới: {folder_name}")
    else:
        print(f"-> Đã tìm thấy thư mục: {folder_name}")

    # 3. Tạo ID nhóm (Dựa trên logic Timestamp = 0 là bắt đầu nhóm mới)
    df['group_id'] = (df['Timestamp'] == 0).cumsum()

    # 4. Tách và lưu file vào thư mục
    grouped = df.groupby('group_id')
    count = 0

    print("-> Đang tiến hành tách và lưu file...")
    
    for _, group_data in grouped:
        # Tạo tên file: trajectory_000.csv, trajectory_001.csv...
        filename = f"trajectory_{count:03d}.csv"
        
        # Tạo đường dẫn đầy đủ: Tên_Thư_Mục/Tên_File
        # os.path.join giúp nối đường dẫn chuẩn xác trên cả Windows và Mac/Linux
        full_path = os.path.join(folder_name, filename)
        
        # Xóa cột group_id tạm trước khi lưu
        save_data = group_data.drop(columns=['group_id'])
        
        # Lưu file
        save_data.to_csv(full_path, index=False)
        count += 1

    print(f"\nHOÀN THÀNH! Đã lưu {count} file vào trong thư mục '{folder_name}'")

# --- CẤU HÌNH ---
# Tên file dữ liệu gốc của bạn
ten_file_goc = 'test_trajectories.csv'

# Tên thư mục bạn muốn lưu các file con vào
ten_thu_muc_luu = 'Experiment_Test_Trajectory_HRI'

# Chạy chương trình
if __name__ == "__main__":
    split_and_save_to_folder(ten_file_goc, ten_thu_muc_luu)