import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class DistortionDataset(Dataset):
    def __init__(self, raw_dir, distorted_dir, image_size=(256, 256)):
        self.raw_paths = sorted(glob(os.path.join(raw_dir, "*.jpg")) + 
                               glob(os.path.join(raw_dir, "*.png")) +
                               glob(os.path.join(raw_dir, "*.jpeg")))
        self.distorted_paths = sorted(glob(os.path.join(distorted_dir, "*.jpg")) + 
                                    glob(os.path.join(distorted_dir, "*.png")) +
                                    glob(os.path.join(distorted_dir, "*.jpeg")))
        self.image_size = image_size
        assert len(self.raw_paths) == len(self.distorted_paths), "原图与畸变图数量不一致"

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        try:
            raw_img = cv2.imread(self.raw_paths[idx])
            distorted_img = cv2.imread(self.distorted_paths[idx])
            if raw_img is None or distorted_img is None:
                raise ValueError("图像加载失败")
                
            # 数据增强
            if np.random.rand() > 0.5:
                raw_img = cv2.flip(raw_img, 1)
                distorted_img = cv2.flip(distorted_img, 1)
                
            # 颜色空间转换和归一化
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) / 255.0
            distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB) / 255.0
            
            # 转换为张量
            return {
                'distorted': torch.from_numpy(distorted_img).permute(2, 0, 1).float(),
                'target': torch.from_numpy(raw_img).permute(2, 0, 1).float()
            }
        except Exception as e:
            print(f"Error loading {self.raw_paths[idx]}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))