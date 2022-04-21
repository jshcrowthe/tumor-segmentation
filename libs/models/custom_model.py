import torch
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(CustomModel, self).__init__()

        features = init_features

        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(features, features*2, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv3d(features*2, features*4, kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(
            nn.Conv3d(features*4, features*8, kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = self.decoder3 = nn.Sequential(
            nn.Conv3d(features*8, features*16, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = nn.Sequential(
            nn.Conv3d(features*16, features*8, kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(features*8, features*4, kernel_size=3, padding=1, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.decoder3 = nn.Sequential(
            nn.Conv3d(features*4, features*2, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.decoder3 = nn.Sequential(
            nn.Conv3d(features*2, features, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))