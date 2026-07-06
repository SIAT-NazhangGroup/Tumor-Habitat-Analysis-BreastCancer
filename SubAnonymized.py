import os
import shutil

# 源文件夹路径
source_dir = r'E:\DCESummary_2019-202004\benign-DCE-221subs\part1-175subs'
# 目标文件夹路径
target_dir = r'E:\liuzhou_breastcancer\benign'

# 遍历源文件夹中的每个文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    # 如果是文件夹且包含 Anonymized 子文件夹
    if os.path.isdir(folder_path) and 'Anonymized' in os.listdir(folder_path):
        # 创建对应的目标文件夹
        target_folder_path = os.path.join(target_dir, folder_name)
        # 检查目标文件夹是否已经存在，如果存在则跳过
        if os.path.exists(target_folder_path):
            print(f"目标文件夹 '{target_folder_path}' 已存在，跳过复制。")
            continue
        # 复制整个 Anonymized 文件夹到目标文件夹中
        anonymized_folder = os.path.join(folder_path, 'Anonymized')
        shutil.copytree(anonymized_folder, target_folder_path)

print("复制完成！")