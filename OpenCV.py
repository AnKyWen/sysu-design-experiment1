import cv2
import numpy as np
import os
from glob import glob

# 输入和输出目录
input_dir = "image"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
# 创建原图和畸变图的子目录
raw_dir = os.path.join(output_dir, "raw")
distorted_dir = os.path.join(output_dir, "distorted")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(distorted_dir, exist_ok=True)

# 输出图像尺寸
target_size = (256, 256)

# 保持比例缩放并 padding 到指定大小
def resize_with_padding(img, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 缩放比例（不超过目标尺寸）
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(img, (new_w, new_h))

    # 计算 padding
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # 添加边框（黑色 padding）
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

# 获取所有图像路径
image_paths = (
    glob(os.path.join(input_dir, "*.jpg")) +
    glob(os.path.join(input_dir, "*.jpeg")) +  # ✅ 加入 jpeg
    glob(os.path.join(input_dir, "*.png"))
)

if not image_paths:
    print("❌ 没有找到图像文件")
    exit()

print(f"共发现图像文件: {len(image_paths)} 张")

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ 加载失败: {path}")
        continue

    h, w = img.shape[:2]

    # 改进的相机内参设置
    focal_length = max(w, h) * 0.8  # 减小焦距，避免过度畸变
    cx = w / 2
    cy = h / 2
    camera_matrix = np.array([[focal_length, 0, cx],
                              [0, focal_length, cy],
                              [0, 0, 1]], dtype=np.float32)
  
    
    # 修改畸变参数生成部分（保持轻微畸变）
    distortion_type = np.random.choice(['barrel', 'pincushion']) 
  
    if distortion_type == 'barrel':  # 轻微桶形畸变（广角端）
        k1 = np.random.uniform(-0.2, -0.08)  # 缩小范围
        k2 = np.random.uniform(-0.05, 0.05)
    elif distortion_type == 'pincushion':  # 轻微枕形畸变（长焦端）
        k1 = np.random.uniform(0.08, 0.2)
        k2 = np.random.uniform(-0.05, 0.05)

    # 切向畸变保持极小值（真实相机中通常很小）
    p1 = np.random.uniform(0, 0)
    p2 = np.random.uniform(0, 0)

    k3=0.0  # 保持k3为0，避免过度畸变
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    # 打印当前畸变参数，方便调试
    print(f"Image {idx+1}/{len(image_paths)}: "
          f"Distortion type={distortion_type}, "
          f"k1={k1:.4f}, k2={k2:.4f}, p1={p1:.4f}, p2={p2:.4f}")

    # ✅ 修正：正确使用相机矩阵和畸变系数
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0.8  # 减小alpha值，保留更多原始区域
    )

    # ✅ 使用正确的畸变系数生成映射
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )
    
    # ✅ 添加边界处理，避免孔洞
    distorted = cv2.remap(img, map1, map2, 
                        interpolation=cv2.INTER_LANCZOS4,  # 高阶插值，效果更好
                        borderMode=cv2.BORDER_REFLECT101)   # 自然反射边界，减少放射模糊


    # 统一尺寸（原图 & 畸变图都缩放+padding）
    img_resized = resize_with_padding(img, target_size)
    distorted_resized = resize_with_padding(distorted, target_size)

    # 保存图像（保持原文件名）
    filename = os.path.basename(path)
    raw_output_path = os.path.join(raw_dir, filename)
    distorted_output_path = os.path.join(distorted_dir, filename)

    cv2.imwrite(raw_output_path, img_resized)
    cv2.imwrite(distorted_output_path, distorted_resized)

    print(f"✅ 已处理: {filename} (畸变类型: {distortion_type})")

print("🎉 所有图像已成功处理和保存！")