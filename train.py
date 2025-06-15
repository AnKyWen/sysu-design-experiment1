from Dataset import DistortionDataset
from UNet import DistortionCorrectionNet
from utils import Evaluate, visualize_result
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import os

# ✅ 平滑约束函数（加在 loss 上）
def flow_smoothness_loss(flow):
    dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
    dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    return dx.mean() + dy.mean()

if __name__ == "__main__":
    # ✅ 设置随机种子以增强复现性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型并移动到设备
    model = DistortionCorrectionNet().to(device)

    # ✅ 如果存在预训练模型，则加载它（避免重复训练）
    pretrained_path = "pretrained_chess.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("✅ 已加载预训练模型权重 pretrained_chess.pth")

    # 你还需要定义优化器和损失函数等
    smoothness_weight = 0.1  # 可根据需要调整
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 加载数据集（请根据你的数据集路径和参数进行调整）
    chess_dataset = DistortionDataset(raw_dir="output_chess/raw", distorted_dir="output_chess/distorted")
    real_dataset = DistortionDataset(raw_dir="output_images/raw", distorted_dir="output_images/distorted")

    chess_loader = DataLoader(chess_dataset, batch_size=8, shuffle=True)
    real_loader = DataLoader(real_dataset, batch_size=8, shuffle=True)

    pretrain_success = True  # ✅ 用于判断是否继续 finetune

    # 预训练和微调的 epoch 数量
    pretrain_epochs = 50  # 可以根据需要调整
    finetune_epochs = 30  # 可以根据需要调整

    # ---------------------
    # ✅ 第一阶段：棋盘格预训练
    # ---------------------
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total_loss = 0.0

        try:
            for batch in chess_loader:
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

        except Exception as e:
            print(f"❌ [预训练阶段错误] Epoch {epoch}: {e}")
            pretrain_success = False
            break

        avg_loss = total_loss / len(chess_loader)
        print(f"[预训练 Epoch {epoch}] Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            try:
                Evaluate(model, chess_loader, device)
                visualize_result(model, chess_dataset, device, idx=0)
            except Exception as e:
                print(f"⚠️ [预训练评估错误] Epoch {epoch}: {e}")
                pretrain_success = False
                break
        
        # ✅ 第一阶段训练完成后（确认 success）保存一次
        if pretrain_success:
            torch.save(model.state_dict(), "pretrained_chess.pth")
            print("✅ 已保存预训练模型为 pretrained_chess.pth")

    # ---------------------
    # ✅ 第二阶段：真实图像微调（仅在成功预训练后）
    # ---------------------
    # ---------------------
    # ✅ 第二阶段：真实图像微调（仅在成功预训练后）
    # ---------------------
    finetuned_path = "distortion_correction.pth"

    if pretrain_success:
        if os.path.exists(finetuned_path):
            print("🚫 检测到已有微调模型 distortion_correction.pth，跳过微调阶段")
        else:
            for epoch in range(1, finetune_epochs + 1):
                model.train()
                total_loss = 0.0
                ...
            torch.save(model.state_dict(), finetuned_path)
            print("✅ 模型已保存为 distortion_correction.pth")

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
                    # break  # Removed because break is invalid here; optionally use 'return' or 'continue' if inside a function or loop

        # ✅ 保存模型
        torch.save(model.state_dict(), "distortion_correction.pth")
        print("✅ 模型已保存为 distortion_correction.pth")
    else:
        print("🚫 跳过微调阶段，因为预训练未成功完成")