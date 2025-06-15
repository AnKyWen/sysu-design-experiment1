from Dataset import DistortionDataset
from UNet import DistortionCorrectionNet
from utils import Evaluate, visualize_result
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import os

def flow_smoothness_loss(flow):
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    return dx.mean() + dy.mean()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistortionCorrectionNet().to(device)

    # 只加载棋盘格预训练模型
    pretrained_path = "pretrained_chess.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("✅ 已加载预训练模型权重 pretrained_chess.pth")
    else:
        print("❌ 未找到预训练模型，请先完成棋盘格预训练！")
        exit()

    smoothness_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 只加载真实图片数据集
    real_dataset = DistortionDataset(raw_dir="output_images/raw", distorted_dir="output_images/distorted")
    real_loader = DataLoader(real_dataset, batch_size=8, shuffle=True)

    finetune_epochs = 30  # 可调整

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in real_loader:
            distorted = batch['distorted'].to(device)
            target = batch['target'].to(device)

            output, flow = model(distorted)
            recon_loss = torch.nn.functional.l1_loss(output, target)
            smooth_loss = flow_smoothness_loss(flow)
            loss = recon_loss + smoothness_weight * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Finetune Epoch {epoch}] Loss: {total_loss / len(real_loader):.4f}")

        if epoch % 10 == 0:
            try:
                Evaluate(model, real_loader, device)
                visualize_result(model, real_dataset, device, idx=0)
            except Exception as e:
                print(f"⚠️ [微调评估错误] Epoch {epoch}: {e}")

    torch.save(model.state_dict(), "distortion_correction.pth")
    print("✅ 微调完成，模型已保存为 distortion_correction.pth")