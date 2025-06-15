import torch
from UNet import DistortionCorrectionNet
from torchvision import transforms
from PIL import Image
import sys
import os

def resize_with_padding(img, target_size=(256, 256), pad_color=(0, 0, 0)):
    w, h = img.size
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new("RGB", target_size, pad_color)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

def load_image(img_path, device):
    img = Image.open(img_path).convert('RGB')
    img = resize_with_padding(img, (256, 256))  # 保持比例缩放并补黑边
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
    return img_tensor

def save_image(tensor, out_path):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(out_path)

if __name__ == "__main__":
    # 用法: python inference.py 输入文件夹路径 输出文件夹路径
    if len(sys.argv) != 3:
        print("用法: python inference.py 输入文件夹路径 输出文件夹路径")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = DistortionCorrectionNet().to(device)
        model.load_state_dict(torch.load("distortion_correction.pth", map_location=device))
        model.eval()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ CUDA显存不足，自动切换到CPU。")
            device = torch.device("cpu")
            model = DistortionCorrectionNet().to(device)
            model.load_state_dict(torch.load("distortion_correction.pth", map_location=device))
            model.eval()
        else:
            raise e

    # 支持的图片格式
    exts = ['.jpg', '.jpeg', '.png', '.bmp']

    for fname in os.listdir(input_dir):
        if os.path.splitext(fname)[1].lower() in exts:
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            img_tensor = load_image(input_path, device)
            with torch.no_grad():
                output, _ = model(img_tensor)
            save_image(output, output_path)
            print(f"已保存矫正结果到 {output_path}")