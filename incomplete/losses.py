# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_window(window_size, sigma, channel):
    x = torch.arange(window_size).float() - window_size//2
    g = torch.exp(-0.5 * (x / sigma).pow(2))
    g /= g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.T
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 1  # assuming grayscale
        self.window = gaussian_window(window_size, sigma, self.channel)

    def forward(self, img1, img2):
        device = img1.device
        window = self.window.to(device)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1,2,3])


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = SSIM(window_size=11, sigma=1.5)

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
