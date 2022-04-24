import torch
import torch.nn as nn

def conv_block(in_features, out_features):
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
    def __init__(self, init_features=32, num_classes=5):
        super(CustomModel, self).__init__()
        in_channels = 1

        features = init_features

        # Encoding layers
        self.encoder1 = conv_block(in_channels, features) # Input channels is always 1
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d()
        self.encoder3 = conv_block(features * 2, features * 4)
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(features * 4, features * 8)
        self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout3d()

        # Intermediate transition layer
        self.transition = conv_block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = conv_block(features * 2, features)

        # Final output layer
        self.conv = nn.Conv3d(
            in_channels=features,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        
        enc3 = self.encoder3(self.dropout1(self.pool2(enc2)))
        enc4 = self.encoder4(self.pool3(enc3))

        transition = self.transition(self.dropout2(self.pool4(enc4)))

        dec4 = self.upconv4(transition)
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
