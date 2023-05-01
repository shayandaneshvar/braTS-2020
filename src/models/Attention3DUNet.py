import torch
import torch.nn as nn

try:
    from .Base3DUNet import Base3DUNet, DoubleConv3D
except ImportError:
    from Base3DUNet import Base3DUNet, DoubleConv3D


class AttentionBlock(nn.Module):
    # Xs come from the encoder and Gs come from previous lower layer and a point-wise conv is applied to both and
    # their sum will be calculated and fed into ReLU Then another point-wise conv (Psi) is applied with a sigmoid after
    # that, which is supposed to be the probability map of each data point (pixel) hence it is multiplied with the X

    # Why we don't multiply it with G?
    # My Intuition: because more information will be available in X and its size is closer to the current module
    # if G is used then we need rescaling or padding, which makes it useless, on the other hand we already have the
    # effect of both
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x_o = self.W_x(x)
        relu_sum = self.relu(g + x_o)
        psi = self.psi(relu_sum)
        psi = torch.sigmoid(psi)
        return x * psi


class Attention3UNet(Base3DUNet):  # Generalization of 2D attention UNet
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], up_sample=True):
        super().__init__(in_channels, out_channels, features, up_sample=up_sample)
        self.attention = nn.ModuleList()

        for feature in reversed(features):
            if up_sample:
                self.attention.append(AttentionBlock(feature * 2, feature, feature))
            else:
                self.attention.append(AttentionBlock(feature, feature, feature))
                # when convT is used, then the G is normal and the same as skip

    def forward(self, inputs):
        skips = []

        x = inputs
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        # same as original Unet till here

        for idx in range(0, len(self.ups), 2):  # going up 2 steps, as each step has convTranspose and DoubleConv
            x = self.ups[idx](x)  # up sampling w/ the convTranspose or Upsampler

            # ---> DIFF START: the skip_connection acquired is the X_l in attention UNet paper, and x is G
            skip_connection = skips.pop()  # give me the last skip I added, to add it first on the ups
            skip_connection = self.attention[idx // 2](x, skip_connection)  # (G,X) = (x, skip_connection) in here
            # <---- end of difference between attention and base 3D UNet

            x = torch.cat((skip_connection, x), dim=1)  # dim 0 is batch, dim 1 is the channels
            x = self.ups[idx + 1](x)  # double conv

        return self.output_conv(x)


def _test_Att3dUNet():
    x = torch.randn((1, 3, 128, 128, 128))
    print(x.shape)
    model = Attention3UNet(in_channels=3, out_channels=3, up_sample=True)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    _test_Att3dUNet()
