import argparse
import os
import shutil
import re
from tkinter import filedialog
from Disease_Rating.Disease_AJ import DiseaseAJ
from Disease_Rating.Disease_BX import DiseaseBX
from Disease_Rating.Disease_CJ import DiseaseCJ
from Disease_Rating.Disease_CK import DiseaseCK
from Disease_Rating.Disease_FZ import DiseaseFZ
from Disease_Rating.Disease_JG import DiseaseJG
from Disease_Rating.Disease_QF import DiseaseQF

def extract_confidence(filename):
    """
    从文件名中提取置信度值，例如 'conf0.65' 返回 0.65
    """
    match = re.search(r'conf(\d\.\d+)', filename)
    if match:
        return float(match.group(1))
    return 0.0


def process_defect_folder(defect_folder, confidence_threshold=0.50):
    """
    处理单个缺陷分类文件夹（如CJ、CK等），将置信度高于threshold的图片复制到新文件夹
    """
    # 获取缺陷类型（文件夹名，如CJ、CK等）
    defect_type = os.path.basename(defect_folder)
    # 新文件夹名，例如 CJ_conf0.50
    new_folder_name = f"{defect_type}_conf{confidence_threshold:.2f}"
    new_folder_path = os.path.join(os.path.dirname(defect_folder), new_folder_name)

    # 如果新文件夹不存在，则创建
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 遍历缺陷文件夹中的所有文件
    for filename in os.listdir(defect_folder):
        if filename.endswith('.jpg'):  # 只处理jpg文件
            confidence = extract_confidence(filename)
            if confidence > confidence_threshold:
                # 源文件路径
                src_path = os.path.join(defect_folder, filename)
                # 目标文件路径
                dst_path = os.path.join(new_folder_path, filename)
                # 复制文件
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {filename} to {new_folder_path}")

def process_classified_results(root_dir, confidence_threshold):
    """
    递归遍历root_dir，找到所有classified_results文件夹，处理其中的缺陷分类文件夹
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'classified_results' in dirpath:
            # 遍历classified_results中的子文件夹（缺陷分类文件夹）
            for defect_folder in dirnames:
                # 排除已经是_conf0.50的文件夹
                if '_conf' not in defect_folder:
                    defect_folder_path = os.path.join(dirpath, defect_folder)
                    process_defect_folder(defect_folder_path, confidence_threshold)



# 处理病害图像
def process_images(defect_folder_path, disease_classified):
    """
    处理单个缺陷分类文件夹（如CJ、CK等），将置信度高于threshold的图片复制到新文件夹
    """
    # 获取缺陷类型（文件夹名，如CJ、CK等）
    defect_type = os.path.basename(defect_folder_path)
    # 新文件夹名，例如 CJ_disease
    new_folder_name = f"{defect_type}_disease{disease_classified}"
    new_folder_path = os.path.join(os.path.dirname(defect_folder_path), new_folder_name)

    # 如果新文件夹不存在，则创建
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    print(f"[DEBUG] 创建的新文件夹名称: {new_folder_name}")  # 调试信息

    return new_folder_path


# 病害处理图像程序查找程序
def disease_results(image_path, disease_classified):
    """
    递归遍历image_path，找到所有CJ，CR。。。 等文件夹，处理其中的缺陷分类文件夹
    """
    processed_folders = []  # 用于存储所有处理后的文件夹路径

    for dirpath, disease_classified_names, _ in os.walk(image_path):
        if 'classified_results' in dirpath:
            # 遍历classified_results中的子文件夹（缺陷分类文件夹）
            for defect_folder in disease_classified_names:
                if (disease_classified + "_conf0.80") in defect_folder and "_disease" not in defect_folder:
                    defect_folder_path = os.path.join(dirpath, defect_folder)
                    return process_images(defect_folder_path, disease_classified)
    return None





# 暗接
def disease_aj(root_dir):

    image_path = root_dir#传入路径

    disease_classified = "AJ"
    result_paths =  disease_results(image_path, disease_classified)

    # 拆分路径和文件名
    dir_path, folder_name = os.path.split(result_paths)
    # 去掉后缀
    clean_folder = folder_name.replace("_diseaseAJ", "")
    # 重新组合路径
    clean_path = os.path.join(dir_path, clean_folder)

    # print(f"图像路径是：{clean_path}")
    # print(f"处理完成的图像路径是：{result_paths}")
    #
    # if not os.path.exists(clean_path):
    #     print(f"❌ 输入目录不存在: {clean_path}")
    #     return

    detector = DiseaseAJ(clean_path, result_paths)
    detector.process_directory()

# 变形
def disease_bx(root_dir):
    """Example usage of PipeDeformationDetector."""
    image_path = root_dir  # 传入路径

    disease_classified = "BX"
    result_paths = disease_results(image_path,disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseBX", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseBX(clean_path, result_paths)
    detector.process_directory()

    # detector = DiseaseBX()
    # result = detector.detect_deformation("TEST_PHOTO/BX_1.png")
    #
    # print("\n📊 Detection Report:")
    # for k, v in result.items():
    #     if k != 'debug_images':
    #         print(f"{k:>18}: {v}")
    #
    # detector.show_debug_images(result)

# 沉积
def disease_cj(root_dir):
    """Example usage of PipeSedimentDetector."""
    image_path = root_dir  # 传入路径

    disease_classified = "CJ"
    result_paths = disease_results(image_path, disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseCJ", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseCJ(clean_path, result_paths)
    detector.process_directory()

# 错口
def disease_ck(root_dir):
    """Example usage of PipeSedimentDetector."""
    image_path = root_dir  # 传入路径

    disease_classified = "CK"
    result_paths = disease_results(image_path, disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseCK", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseCK(clean_path, result_paths)
    detector.process_directory()
    # detector = DiseaseCK()
    # # 替换为您的图像路径
    # image_path = "TEST_PHOTO/CK1.png"
    # img_name = os.path.splitext(os.path.basename(image_path))[0]
    #
    # try:
    #     result = detector.detect_deformation(image_path)
    #     print("\n📊 最终结果:")
    #     for k, v in result.items():
    #         if k != "debug_images":
    #             print(f"{k}: {v}")
    #
    #     # 显示最终结果
    #     detector.show_debug_images(img_name)
    #
    # except Exception as e:
    #     print(f"❌ 处理过程中发生错误: {str(e)}")

# 浮渣
def disease_fz(root_dir):
    """Example usage of PipeSedimentDetector."""
    image_path = root_dir  # 传入路径

    disease_classified = "FZ"
    result_paths = disease_results(image_path, disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseFZ", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseFZ(clean_path, result_paths)
    detector.process_directory()

    # """Example usage of PipeSedimentDetector."""
    # parser = argparse.ArgumentParser(description="Pipe Floating Debris Detection")
    # parser.add_argument("--image", default="Test_PHOTO/FZ.png", help="Path to the input image")
    # parser.add_argument("--output", default="result_with_diameter.jpg", help="Path to save the output image")
    # args = parser.parse_args()
    #
    # detector = DiseaseFZ()
    #
    # # Automatically estimate pipe diameter
    # try:
    #     detector.auto_estimate_diameter(args.image, method="auto")
    #     if detector.pipe_diameter is None:
    #         detector.pipe_diameter = 342  # Fallback to manual setting
    #         print("Automatic estimation failed! Using manual diameter: 342 px")
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     exit(1)
    #
    # # Process the image
    # try:
    #     result, max_thickness = detector.process_image(args.image)
    #     print("\nDetection Result:")
    #     for k, v in result.items():
    #         print(f"{k}: {v}")
    #     print(f"Max Thickness: {max_thickness} px")
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     exit(1)
    #
    # # Visualize results
    # try:
    #     detector.visualize(args.image, args.output)
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     exit(1)
    #
    # # Calculate debris area
    # try:
    #     area = detector.calculate_sediment_area(args.image)
    #     print(f"Estimated debris area: {area} px²")
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     exit(1)

# 结垢
def disease_jg(root_dir):

    image_path = root_dir  # 传入路径

    disease_classified = "JG"
    result_paths = disease_results(image_path, disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseJG", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseJG(clean_path, result_paths)
    detector.process_directory()
    # detector = DiseaseJG()
    # result = detector.calculate_blockage_ratio("TEST_PHOTO/JG.png")  # 替换为你的图像路径
    # print("\n最终结果:")
    # for k, v in result.items():
    #     print(f"{k}: {v}")
    #
    # #PipeScaleDetector.read_photo("JG_debug_output/JG_05_debug.jpg") # 替换为你的图像路径

def disease_qf(root_dir):
    image_path = root_dir  # 传入路径

    disease_classified = "QF"
    result_paths = disease_results(image_path, disease_classified)

    dir_path, folder_name = os.path.split(result_paths)
    clean_folder = folder_name.replace("_diseaseQF", "")
    clean_path = os.path.join(dir_path, clean_folder)

    print(f"图像路径是：{clean_path}")
    print(f"处理完成的图像路径是：{result_paths}")

    if not os.path.exists(clean_path):
        print(f"❌ 输入目录不存在: {clean_path}")
        return

    detector = DiseaseQF(clean_path, result_paths)
    detector.process_directory()
    # # Initialize the detector
    # detector = DiseaseQF()
    #
    # # Automatically estimate pipe diameter
    # detector.auto_estimate_diameter("Test_PHOTO/QF.png", method="auto")
    # if detector.pipe_diameter is None:
    #     detector.pipe_diameter = 342  # Fallback to manual setting if auto-estimation fails
    #     print("自动估算失败！！！使用手动设置直径: 342 px")
    #
    # # Process the image for water depth analysis
    # result, water_depth_pixels = detector.detect_water_depth("Test_PHOTO/QF.png")
    # print("检测结果:")
    # for k, v in result.items():
    #     print(f"{k}: {v}")
    # print(f"检测到的水深 (像素): {water_depth_pixels} px")
    #
    # # Visualize the results
    # detector.visualize("Test_PHOTO/QF.png", "QF_debug_output/result_water_depth_diameter_ratio.jpg")
    #
    # # Test water depth pixels calculation
    # depth_pixels = detector.calculate_water_depth_pixels("Test_PHOTO/QF.png")
    # print(f"估算水深: {depth_pixels} px")


def main():
    # 设置根目录
    # Open a directory chooser dialog
    # print("Please select the root directory (e.g., D:/安澜路（康平路-迎宾大道）) in the pop-up window...")
    root_directory = filedialog.askdirectory(title="选择项目目录")
    confidence_threshold = 0.80  # 置信度阈值

    # 确保根目录存在
    if not os.path.exists(root_directory):
        print(f"Error: Directory {root_directory} does not exist.")
        return

    print(f"Processing directory: {root_directory}") #打印处理目录
    process_classified_results(root_directory, confidence_threshold)#处理张照片
    print("Processing completed.")  #处理完成

    #调用病害处理方法
    disease_aj(root_directory)
    disease_bx(root_directory)
    disease_cj(root_directory)
    disease_ck(root_directory)
    disease_fz(root_directory)
    disease_jg(root_directory)
    disease_qf(root_directory)

if __name__ == "__main__":
    main()