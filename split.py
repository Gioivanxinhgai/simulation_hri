import os
import json
from sklearn.model_selection import train_test_split

folder_path = 'Filtered_Cocarry_Logs'
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

print(f"Tổng số file CSV tìm thấy: {len(file_list)}")

train_files, val_files = train_test_split(
    file_list,
    train_size=0.9,
    random_state=36,
    shuffle=True
)

print(f"Số lượng file huấn luyện (train): {len(train_files)}")
print(f"Số lượng file kiểm định (validation): {len(val_files)}")
# Lưu danh sách file
with open(os.path.join(folder_path, 'test_file_list.json'), 'w') as f:
    json.dump(train_files, f, indent=4)
with open(os.path.join(folder_path, 'val_file_list.json'), 'w') as f:
    json.dump(val_files, f, indent=4)
print("Đã lưu danh sách file huấn luyện, kiểm định và kiểm thử.")