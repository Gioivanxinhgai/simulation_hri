import os
import json
import glob

def create_csv_list_json(folder_path, output_json_path):
    try:
        # Tìm tất cả file CSV trong thư mục
        csv_pattern = os.path.join(folder_path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Không tìm thấy file CSV nào trong thư mục {folder_path}")
            return
        
        # Lấy tên file (không bao gồm đường dẫn)
        csv_file_names = [os.path.basename(file) for file in csv_files]
        
        # Sắp xếp theo tên file để đảm bảo thứ tự nhất quán
        csv_file_names.sort()
        
        # Tạo dictionary với thông tin
        result = {
            "folder_path": folder_path,
            "total_files": len(csv_file_names),
            "csv_files": csv_file_names
        }
        
        # Tạo đường dẫn đầy đủ cho file JSON trong thư mục gốc
        full_output_path = os.path.join(folder_path, output_json_path)
        
        # Lưu vào file JSON
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(csv_file_names, f, indent=4, ensure_ascii=False)
        
        print(f"Đã tạo file JSON thành công: {full_output_path}")
        print(f"Số lượng file CSV tìm thấy: {len(csv_file_names)}")
        print("Danh sách file CSV:")
        for i, file_name in enumerate(csv_file_names, 1):
            print(f"  {i}. {file_name}")
            
    except Exception as e:
        print(f"Lỗi khi tạo file JSON: {e}")

if __name__ == "__main__":
    folder_path = "Filtered_Cocarry_Logs"  # Thay đổi đường dẫn thư mục tại đây
    output_json_path = "train_file_list.json"  # Tên file JSON output
    create_csv_list_json(folder_path, output_json_path)
