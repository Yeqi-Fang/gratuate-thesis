import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """Modified UNet with hidden state support"""
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        # Encoder
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Hidden state processor
        self.hidden_conv = nn.Conv2d(512 + hidden_channels, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.up1 = Up(768, 256)  # 512+256
        self.up2 = Up(384, 128)  # 256+128
        self.up3 = Up(192, 64)   # 128+64
        self.outc = nn.Conv2d(64, output_channels + hidden_channels, kernel_size=1)

    def forward(self, x, hidden):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Combine with hidden state
        if hidden is None:
            hidden = torch.zeros(x4.size(0), self.hidden_channels, 
                               x4.size(2), x4.size(3)).to(x.device)
        
        combined = torch.cat([x4, hidden], dim=1)
        processed = self.hidden_conv(combined)
        
        # Decoder
        x = self.up1(processed, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        
        # Split output into prediction and new hidden state
        pred = output[:, :self.output_channels]
        new_hidden = output[:, self.output_channels:]
        
        return pred, new_hidden

class RecurrentSinogramModel(nn.Module):
    """Main reconstruction model with recurrent processing"""
    def __init__(self, input_channels, output_channels, num_blocks, hidden_channels=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        self.unet = UNet(input_channels, output_channels, hidden_channels)
        
    def forward(self, x_blocks):
        batch_size, _, h, w = x_blocks[0].shape
        hidden = None
        outputs = []
        
        # Process each block sequentially
        for i in range(self.num_blocks):
            block = x_blocks[:, i]
            output, hidden = self.unet(block, hidden)
            outputs.append(output)
        
        # Combine all outputs along the depth dimension
        reconstructed = torch.stack(outputs, dim=2)  # [B, C, D, H, W]
        reconstructed = reconstructed.view(batch_size, -1, h, w)
        return reconstructed

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    N1 = 448    # 探测器数量（根据示例中的449调整）
    N2 = 64     # 环数量
    num_blocks = N2
    input_channels = 64  # 每个块的通道数（N2 per block）
    output_channels = 64
    
    # 模拟输入数据：[batch_size, num_blocks, channels, height, width]
    # 假设每个块是 (224, 449, 64) -> reshape为 (64, 224, 449) 作为输入
    batch_size = 2
    dummy_input = [torch.randn(batch_size, input_channels, 224, 449) for _ in range(num_blocks)]
    dummy_input = torch.stack(dummy_input, dim=1)  # [2, 64, 64, 224, 449]
    
    # 初始化模型
    model = RecurrentSinogramModel(
        input_channels=input_channels,
        output_channels=output_channels,
        num_blocks=num_blocks,
        hidden_channels=64
    )
    
    # 前向传播
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # 应该为 [2, 4096, 224, 449]