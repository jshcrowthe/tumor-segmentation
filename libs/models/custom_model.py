from collections import OrderedDict

import torch
import torch.nn as nn

def conv_block(in_features, out_features, name):
    return nn.Sequential(
        nn.Conv3d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
        nn.Conv3d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
        nn.Conv3d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm3d(out_features),
        nn.ReLU(),
    )

class CustomModel(nn.Module):

    def __init__(self, init_features=32):
        super(CustomModel, self).__init__()
        in_features = 1
        out_features = 1

        features = init_features
        self.encoder1 = conv_block(in_features, features, name="enc1") # Input channels is always 1
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        
        self.transition = conv_block(features, features * 2, name="transition")

        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = conv_block(features*2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_features, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)

        transition = self.transition(self.pool1(enc1))

        dec1 = self.upconv1(transition)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
