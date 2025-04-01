import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# 1. Define a ConvLSTM Cell
#############################################

class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Parameters:
            input_dim: Number of channels of input tensor.
            hidden_dim: Number of channels of hidden state.
            kernel_size: Size of the convolutional kernel.
            bias: Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding    = kernel_size // 2
        self.bias       = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Parameters:
            input_tensor: (batch, input_dim, height, width)
            cur_state: tuple (h_cur, c_cur) of current hidden and cell state
        Returns:
            h_next, c_next: Next hidden and cell state
        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (batch, input_dim+hidden_dim, height, width)
        conv_output = self.conv(combined)
        # split the convolution output into four parts for gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # cell candidate

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden state and cell state with zeros.
        """
        height, width = image_size
        device = self.conv.weight.device
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)


#############################################
# 2. Define a simple U-Net for image refinement
#############################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # if bilinear, use the normal upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # learnable upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x1 if necessary in case the in/out dimensions differ slightly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to produce the desired number of output channels."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Parameters:
            n_channels: number of input channels (e.g., feature maps from the ConvLSTM)
            n_classes: number of output channels (for sinogram reconstruction, typically 1)
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#############################################
# 3. Combine ConvLSTM and U-Net into a Recurrent Model
#############################################

class SinogramReconstructionNet(nn.Module):
    """
    This model processes a sequence of sinogram chunks.
    
    For each time step (e.g., each ring difference or sinogram sub-image),
    the input is first projected into a feature space, then a ConvLSTM cell 
    updates its hidden state based on that feature, and finally a U-Net refines 
    the hidden state to output a completed sinogram chunk.
    
    The final output is a sequence of reconstructed sinogram chunks that 
    can be merged into the complete sinogram.
    """
    def __init__(self, 
                 input_channels=1, 
                 lstm_hidden_dim=16, 
                 unet_out_channels=1,
                 lstm_kernel_size=3,
                 bilinear=True):
        super(SinogramReconstructionNet, self).__init__()
        # First, map the input (e.g., a single-channel sinogram chunk) to the hidden_dim size.
        self.initial_conv = nn.Conv2d(input_channels, lstm_hidden_dim, kernel_size=3, padding=1)
        # Define the ConvLSTM cell.
        self.conv_lstm = ConvLSTMCell(input_dim=lstm_hidden_dim,
                                      hidden_dim=lstm_hidden_dim,
                                      kernel_size=lstm_kernel_size)
        # Define a U-Net that takes the ConvLSTM hidden state as input.
        self.unet = UNet(n_channels=lstm_hidden_dim, n_classes=unet_out_channels, bilinear=bilinear)

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (batch, seq_len, channels, height, width)
               For example, if you have 64 sinogram chunks, seq_len=64.
        Returns:
            outputs: Tensor of shape (batch, seq_len, unet_out_channels, height, width)
        """
        batch_size, seq_len, channels, H, W = x.shape
        # Initialize ConvLSTM hidden and cell states.
        h, c = self.conv_lstm.init_hidden(batch_size, (H, W))
        outputs = []
        for t in range(seq_len):
            # Get the t-th sinogram chunk.
            x_t = x[:, t]  # shape: (batch, channels, H, W)
            # Project the input to match the ConvLSTM input channels.
            x_t_proj = self.initial_conv(x_t)  # shape: (batch, lstm_hidden_dim, H, W)
            # Update the ConvLSTM state.
            h, c = self.conv_lstm(x_t_proj, (h, c))
            # Use the updated hidden state as input to the U-Net.
            y_t = self.unet(h)
            outputs.append(y_t.unsqueeze(1))
        # Concatenate outputs along the time dimension.
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, seq_len, unet_out_channels, H, W)
        return outputs


#############################################
# 4. Example usage with dummy data
#############################################

if __name__ == '__main__':
    # Assume each sinogram chunk has shape (N1/2, N1+1) for a single channel.
    # For example, let H = 224, W = 449, and we have 64 such chunks.
    batch_size = 1
    seq_len = 64
    H, W = 112, 225
    dummy_input = torch.randn(batch_size, seq_len, seq_len, H, W, dtype=torch.float32)
    
    # Create the model instance.
    model = SinogramReconstructionNet(input_channels=64,
                                      lstm_hidden_dim=16,
                                      unet_out_channels=64,
                                      lstm_kernel_size=3,
                                      bilinear=True)
    # Forward pass.
    reconstructed_chunks = model(dummy_input)
    print("Output shape:", reconstructed_chunks.shape)
    # Expected output shape: (2, 64, 1, 224, 449)
