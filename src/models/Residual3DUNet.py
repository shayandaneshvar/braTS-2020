# Residual 3DUNet model

import torch
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # the output image will be (n + 2p — f + 1) * (n + 2p — f + 1) where p =1 in this case.
        # Convolutional Layer
        self.conv = nn.Sequential(
          nn.BatchNorm3d(in_channels),
          nn.ReLU(inplace=True),
          nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
          nn.BatchNorm3d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))

        # Identity Mapping
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.conv(inputs) 
        s = self.shortcut(inputs)       
        skip = x + s
        return skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True) #  mode="trilinear"
        self.residual = ResidualBlock(in_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.residual(x)
        return x

class Res3DUNet(nn.Module):
    # the default dataset has 3 channels of data ->  T1CE, T2, FLAIR
    # The output has background, NCR/NET, ED, ET 

    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()

        # Encoder 1 
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        #Identity Mapping
        self.conv3 = nn.Conv3d(in_channels, 64, kernel_size=1, padding=0)
        
        # Encoder 2 
        self.r2 = ResidualBlock(64, 128, stride=2)
        # Encoder 3 
        self.r3 = ResidualBlock(128, 256, stride=2)
        # Encoder 4 
        self.r4 = ResidualBlock(256, 512, stride=2)
        # Bridge
        self.r5 = ResidualBlock(512, 1024, stride=2)
        # Decoder 1
        self.d1 = DecoderBlock(1024, 512)
        # Decoder 2
        self.d2 = DecoderBlock(512, 256)
        # Decoder 3
        self.d3 = DecoderBlock(256, 128)
        # Decoder 4
        self.d4 = DecoderBlock(128, 64)

        # Output 
        self.output = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)


    def forward(self, inputs):
        # Encoder 1 
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        s = self.conv3(inputs)
        skip1 = x + s
        # Encoder 2 
        skip2 = self.r2(skip1)
        # Encoder 3 
        skip3 = self.r3(skip2)
        # Encoder 4 
        skip4 = self.r4(skip3)
        # Bridge 
        b = self.r5(skip4)
        # Decoder 1
        d1 = self.d1(b, skip4)
        # Decoder 2
        d2 = self.d2(d1, skip3)
        # Decoder 3
        d3 = self.d3(d2, skip2)
        # Decoder 4
        d4 = self.d4(d3, skip1)
        # output 
        output = self.output(d4)

        return output
