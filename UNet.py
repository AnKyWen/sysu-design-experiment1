import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation 注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight


# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(0.2)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        res = self.residual(x)
        return self.relu(out + res)


# 主模型：输出 flow（形变坐标）
class DistortionCorrectionNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()

        self.enc1 = ResidualBlock(3, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 4),
            SEBlock(base_channels * 4)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 2),
            SEBlock(base_channels * 2)
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels),
            SEBlock(base_channels)
        )

        # 输出偏移场（flow），值范围控制在 [-1, 1]
        self.flow = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码-解码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        flow = self.flow(d1)  # 输出形变场 [B, 2, H, W]
        flow = torch.tanh(flow) * 0.3  # 控制偏移范围，可调系数（0.3 表示最多偏移 30% 图像尺寸）

        # 构造 [-1, 1] 范围的标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # flow 是 [B, 2, H, W] -> 转为 [B, H, W, 2]，与 base_grid 相加
        sampling_grid = base_grid + flow.permute(0, 2, 3, 1)

        # 使用 grid_sample 做坐标变换采样
        corrected = F.grid_sample(x, sampling_grid, align_corners=True, mode='bilinear', padding_mode='border')
        return corrected, flow  # 可以返回 flow 方便后续计算可视化或 smooth loss
