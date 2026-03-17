import torch
import torch.nn as nn


# ---------------- Residual Block ----------------
class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):

        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)

        return x + res


# ---------------- ERestormer ----------------
class ERestormer(nn.Module):

    def __init__(self, inp_channels=6, out_channels=3, dim=64, num_blocks=8, **kwargs):
        super().__init__()

        # RGB image (3) + event channels
        in_ch = inp_channels + 3

        self.embed = nn.Conv2d(in_ch, dim, 3, padding=1)

        body = []
        for _ in range(num_blocks):
            body.append(ResBlock(dim))

        self.body = nn.Sequential(*body)

        self.out = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x, event):
        """
        x: blurry RGB image  (B,3,H,W)
        event: event voxel tensor (B,C,H,W)
        """

        inp = torch.cat([x, event], dim=1)

        feat = self.embed(inp)

        res = self.body(feat)

        feat = feat + res

        out = self.out(feat)

        return out