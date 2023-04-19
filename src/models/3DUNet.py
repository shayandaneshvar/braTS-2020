import torch
import torch.nn as nn


class SimpleLogger:

    def __init__(self, debug=True):
        self.debug = debug

    def enable_debug(self):
        self.debug = True

    def disable_debug(self):
        self.debug = False

    def log(self, message, condition=True):
        if self.debug and condition:
            print(message)


logger = SimpleLogger(debug=True)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1 + (L - l + 2P)/s
        self.conv = nn.Sequential(
            # 1 + out - 3 + 2 = out
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)


class Base3DUNet(nn.Module):
    # the default dataset has 3 channels of data ->  T1CE, T2, FLAIR
    # The output has background, NCR/NET, ED, ET
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512]):
        super().__init__()
        # 1 + (L - l + 2P)/s
        # 1 + (L - 2)/2 = L
        self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Each Layer - number of filters , see UNet architecture
        input_channels = in_channels

        for feature in features:
            self.downs.append(DoubleConv3D(input_channels, feature))
            input_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)  # this connects downs to ups

        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)  # last layer - feature compression

    def forward(self, inputs):
        skips = []

        x = inputs
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):  # going up 2 steps, as each step has convTranspose and DoubleConv
            x = self.ups[idx](x)  # up sampling w/ the convTranspose
            skip_connection = skips.pop()  # give me the last skip I added, to add it first on the ups
            x = torch.cat((skip_connection, x), dim=1)  # dim 0 is batch, dim 1 is the channels
            x = self.ups[idx + 1](x)  # double conv

        return self.output_conv(x)


def _test_3dUNet():
    x = torch.randn((1, 3, 128, 128, 128))
    print(x.shape)
    model = Base3DUNet(in_channels=3)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    _test_3dUNet()
