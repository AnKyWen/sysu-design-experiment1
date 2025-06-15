import cv2
import numpy as np
import os
import random

# 设置路径
chessboard_path = 'chessboard.png'
output_dir = 'output_chess'
raw_dir = os.path.join(output_dir, 'raw')
distorted_dir = os.path.join(output_dir, 'distorted')
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(distorted_dir, exist_ok=True)

# 输出统一尺寸
target_size = (256, 256)

# 保持比例缩放并 padding 到指定大小
def resize_with_padding(img, target_size=(256, 256), pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded

# 加载并统一处理棋盘格图像
img = cv2.imread(chessboard_path)
if img is None:
    raise FileNotFoundError(f"未找到图像: {chessboard_path}")

img = resize_with_padding(img, target_size=target_size)
h, w = img.shape[:2]

# 相机内参
focal_length = max(w, h) * 0.8
cx, cy = w / 2, h / 2
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=np.float32)

# 生成随机畸变参数
def generate_random_distortion():
    distortion_type = random.choice(['barrel', 'pincushion'])
    if distortion_type == 'barrel':
        k1 = random.uniform(-0.23, -0.1)
        k2 = random.uniform(0, 0)
    else:
        k1 = random.uniform(0.1, 0.23)
        k2 = random.uniform(0, 0)
    p1 = random.uniform(0, 0)
    p2 = random.uniform(0, 0)
    k3 = 0.0
    return distortion_type, np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 生成图像数量
num_images = 1000

for i in range(num_images):
    distortion_type, dist_coeffs = generate_random_distortion()
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0.8)

    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1
    )
    distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 再次 resize（确保 remap 后仍为标准尺寸）
    img_resized = resize_with_padding(img, target_size)
    distorted_resized = resize_with_padding(distorted, target_size)

    # 保存
    cv2.imwrite(os.path.join(raw_dir, f"{i:04d}.png"), img_resized)
    cv2.imwrite(os.path.join(distorted_dir, f"{i:04d}.png"), distorted_resized)

    if i % 100 == 0:
        print(f"[生成] 第 {i} 张图像完成，畸变类型: {distortion_type}")

print(f"\n✅ 共生成 {num_images} 对图像，保存于：{output_dir}")
