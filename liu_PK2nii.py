import os
import SimpleITK as sitk

def find_dicom_series_directory(directory):
    """
    递归地搜索含有DICOM序列的最深层目录。
    """
    reader = sitk.ImageSeriesReader()
    for root, dirs, files in os.walk(directory):
        # 检查当前目录是否有DICOM文件
        dicom_names = reader.GetGDCMSeriesFileNames(root)
        if dicom_names:
            return root  # 返回包含DICOM文件的目录
        # 如果当前目录下没有文件，继续检查子目录
    return None  # 如果没有找到任何DICOM文件，返回None

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历input_dir下的所有子文件夹
    for patient in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient)
        if os.path.isdir(patient_dir):
            for session in os.listdir(patient_dir):
                session_dir = os.path.join(patient_dir, session)
                if os.path.isdir(session_dir):
                    dicom_dir = find_dicom_series_directory(session_dir)
                    if dicom_dir:
                        process_session(dicom_dir, output_dir, patient, session)

def process_session(session_dir, output_dir, patient, session):
    # 读取所有DICOM文件
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(session_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # 获取输出路径
    patient_session_dir = os.path.join(output_dir, patient, session)
    if not os.path.exists(patient_session_dir):
        os.makedirs(patient_session_dir)

    # 保存为NII格式
    output_file = os.path.join(patient_session_dir, f"{session}.nii")
    sitk.WriteImage(image, output_file)
    print(f"Saved NII file at {output_file}")

# 指定Benign和Malignant文件夹的路径
# benign_dir = '<REDACTED_PATH>'"
# malignant_dir = '<REDACTED_PATH>'"
malignant_dir = r"<NEW_ROOT>\P00019610\HGTFHIKV"
# benign_nii_dir = '<REDACTED_PATH>'"
malignant_nii_dir = '<REDACTED_PATH>'"

# 处理Benign和Malignant目录
# process_directory(benign_dir, benign_nii_dir)
process_directory(malignant_dir, malignant_nii_dir)
