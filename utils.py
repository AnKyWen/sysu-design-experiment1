from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import matplotlib.pyplot as plt
import numpy as np

def Evaluate(model, dataloader, device):
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            distorted = batch['distorted'].to(device)
            gt = batch['target'].to(device)
            output, _ = model(distorted)  # 若模型输出为元组，正确处理

            if output.shape != gt.shape:
                print(f"[警告] 第{i+1}批次输出尺寸与目标不匹配: output={output.shape}, gt={gt.shape}")
                continue

            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gt.cpu().numpy().transpose(0, 2, 3, 1)

            batch_size = output_np.shape[0]
            batch_psnr = 0.0
            batch_ssim = 0.0

            for b in range(batch_size):
                pred_img = np.clip(output_np[b], 0, 1)
                gt_img = np.clip(gt_np[b], 0, 1)

                # ✅ 加在这！做归一化判断（防止错误范围）
                if pred_img.max() > 1.0:
                    pred_img = pred_img / 255.0
                if gt_img.max() > 1.0:
                    gt_img = gt_img / 255.0

                h, w = pred_img.shape[:2]
                win_size = min(h, w, 7)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size < 3:
                    print(f"[跳过] 第{i+1}批第{b+1}张图太小: {h}x{w}")
                    continue

                try:
                    psnr_val = psnr(pred_img, gt_img, data_range=1.0)
                    ssim_val = ssim(pred_img, gt_img, channel_axis=-1, win_size=win_size, data_range=1.0)
                except Exception as e:
                    print(f"[错误] 第{i+1}批第{b+1}张计算失败: {e}")
                    continue

                batch_psnr += psnr_val
                batch_ssim += ssim_val
                count += 1

            # 这一批次所有图像平均值累加到总指标
            if batch_size > 0:
                total_psnr += batch_psnr / batch_size
                total_ssim += batch_ssim / batch_size

    if count == 0:
        print("❌ 无有效数据用于评估")
        return

    avg_psnr = total_psnr / (count / batch_size)  # 按批次数算平均
    avg_ssim = total_ssim / (count / batch_size)

    print(f"📈 平均 PSNR: {avg_psnr:.2f}, 平均 SSIM: {avg_ssim:.4f}")


def visualize_result(model, dataset, device, idx=0): 
    model.eval()
    with torch.no_grad():
        sample = dataset[idx]
        distorted = sample['distorted'].unsqueeze(0).to(device)
        target = sample['target'].unsqueeze(0).to(device)
        output, _ = model(distorted)  # ✅ 只取 output

        def tensor_to_image(tensor):
            img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            return np.clip(img, 0, 1)

        distorted_img = tensor_to_image(distorted)
        target_img = tensor_to_image(target)
        output_img = tensor_to_image(output)

        plt.figure(figsize=(15, 5))
        titles = ['Distorted', 'Corrected', 'Ground Truth']
        images = [distorted_img, output_img, target_img]

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

