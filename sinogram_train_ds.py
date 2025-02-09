import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class RecurrentSinogramNet(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        # U-Net模块输入包含原始通道+隐藏通道
        self.unet = UNet(1 + hidden_channels, hidden_channels + 1)
        self.hidden_channels = hidden_channels

    def forward(self, x_in):
        """处理分块后的sinogram数据
        Args:
            x_in: [B, C=1, H, W, N_blocks] 这里假设已分块并调整维度
        Returns:
            output_blocks: [B, C, H, W, N_blocks] 重建的各分块
        """
        B, C, H, W, N_blocks = x_in.shape
        device = x_in.device
        
        # 初始化为4维张量 [B, hidden_ch, H, W]
        hidden = torch.zeros(B, self.hidden_channels, H, W).to(device)
        
        output_list = []
        for t in range(N_blocks):
            # 调整维度顺序
            current_input = x_in[:, :, :, :, t]  # [B,1,H,W]
            
            # 拼接时应保持4维结构
            unet_input = torch.cat([
                current_input,  # [B,1,H,W]
                hidden  # [B,hidden_channels,H,W]
            ], dim=1)  # →[B, (1+hidden_channels), H, W]
            
            # U-Net前向传播
            unet_output = self.unet(unet_input)  # [B, hidden_channels+1, H, W]
            
            # 解构输出：最后一层是预测值，其余是新的隐藏状态
            current_output = unet_output[:, :1]  # [B, 1, H, W]
            new_hidden = unet_output[:, 1:]     # [B, hidden_channels, H, W]
            
            # 保存当前输出
            output_list.append(current_output.permute(0,2,3,1).unsqueeze(-1))  # [B, H, W, 1, 1]
            
            # 更新隐藏状态
            hidden = new_hidden
        
        # 整合所有块的结果
        reconstructed = torch.cat(output_list, dim=-1)  # [B, H, W, 1, N_blocks]
        return reconstructed.squeeze(-2)  # [B, H, W, N_blocks]

class SinogramDataset(torch.utils.data.Dataset):
    """自定义数据集处理分块sinogram"""
    def __init__(self, full_sinograms, block_size=64):
        self.block_size = block_size
        self.data = []
        # 对每个完整sinogram进行分块处理
        for sino in full_sinograms:  # sino shape: [H, W, D]
            n_blocks = sino.shape[-1] // block_size
            for idx in range(n_blocks):
                block = sino[..., idx*block_size:(idx+1)*block_size]
                self.data.append((block, sino))  # (输入块, 完整sinogram)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_block, target = self.data[idx]
        return torch.FloatTensor(input_block), torch.FloatTensor(target)

# 测试验证
if __name__ == "__main__":
    model = RecurrentSinogramNet(hidden_channels=32)
    test_input = torch.randn(1, 1, 224, 448, 64)  # BCHWN
    output = model(test_input)
    print("输入形状:", test_input.shape)  # [1,1,224,449,8] 
    print("输出形状:", output.shape)      # [1,224,449,8]