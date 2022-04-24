from textwrap import dedent
from turtle import down
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

"""
Unet implemenation for biomedical image segmentation:
https://arxiv.org/abs/1505.04597

Other references:
https://amaarora.github.io/2020/09/13/unet.html
"""


class Unet(nn.Module):
    def __init__(self, depth=4, num_classes=5) -> None:
        super(Unet, self).__init__()

        self.depth = depth

        # Maximum depth 4, otherwise dimensions are too small. Channels must be passed as integers to torch.nn
        self.channel_dims = [int(64 * 2.0 ** (i)) for i in range(self.depth)]
        # print("Unet channels", self.channel_dims)

        self.up_channel = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=1)
        self.down = UnetDown(self.channel_dims)
        self.bottom = ConvBlock(self.channel_dims[-1], self.channel_dims[-1])
        self.up = UnetUp(self.channel_dims)


        # Final conv to retain the input volume dimensions
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(self.channel_dims[1], self.channel_dims[1]),

        )
        self.final_conv = nn.Conv3d(self.channel_dims[1], num_classes, 1)

    def forward(self, input):
        _, _, D, H, W = input.shape

        # print("initial input shape", input.shape)
        input = self.up_channel(input)
        out, features = self.down(input)
        out = self.bottom(out)
        out = self.up(out, features)
        out = self.upscale(out)
        out = self.final_conv(out)

        return out


class UnetUp(nn.Module):
    def __init__(self, channel_dims) -> None:
        super(UnetUp, self).__init__()

        # Large -> small
        self.channel_dims = copy.deepcopy(channel_dims)
        self.channel_dims.reverse()
        # print("up dims", self.channel_dims)

        self.up = nn.ModuleList(
            [
                ConvBlock(self.channel_dims[i], self.channel_dims[i + 1])
                for i in range(len(self.channel_dims) - 1)
            ]
        )

        self.conv = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    self.channel_dims[i],
                    self.channel_dims[i + 1],
                    kernel_size=2,
                    stride=2,
                )
                for i in range(len(self.channel_dims) - 1)
            ]
        )

    def center_crop_3d(self, input, features):
        # Reference:
        # https://stackoverflow.com/questions/57517121/how-can-i-do-the-centercrop-of-3d-volumes-inside-the-network-model-with-pytorch
        N_i, C_i, D_i, H_i, W_i = input.shape
        N_f, C_f, D_f, H_f, W_f = features.shape

        # print("input shape", input.shape)
        # print("features shape", features.shape)

        # The feature dimensions are always bigger because they come from the input
        d_delta = D_f - D_i
        h_delta = H_f - H_i
        w_delta = W_f - W_i

        # print("D, H, W", d_delta, h_delta, w_delta)

        # Return the center cropped volume
        res = features[
            :,
            :,
            int(math.floor(d_delta / 2)) : int(math.floor(-d_delta / 2)),
            int(math.floor(h_delta / 2)) : int(math.floor(-h_delta / 2)),
            int(math.floor(w_delta / 2)) : int(math.floor(-w_delta / 2)),
        ]

        return res

    def forward(self, input, features):
        features.reverse()
        # print("feature channels", [f.shape for f in features])
        # print("up iter step", len(self.up))
        for i in range(len(self.up) - 1):
            # print("iteration #", i)
            out = self.conv[i](input)
            down_features = self.center_crop_3d(out, features[i + 1])
            # print("out shape", out.shape)
            # print("features shape", down_features.shape)
            out = torch.cat([out, down_features], dim=1)  # Concatenate on the channels
            # print("out shape", out.shape)
            # print("expected for shape for upconv", self.channel_dims[i])
            input = self.up[i](out)

        return input


class UnetDown(nn.Module):
    def __init__(self, channel_dims) -> None:
        super(UnetDown, self).__init__()

        # Small -> large
        self.channel_dims = copy.deepcopy(channel_dims)

        # Construct the downward convolution path with the desired dimensions
        self.down = nn.ModuleList(
            [
                ConvBlock(self.channel_dims[i], self.channel_dims[i + 1])
                for i in range(len(self.channel_dims) - 1)
            ]
        )
        # Maxpool and halve all dimensions of the volume
        self.max_pool = nn.MaxPool3d(2)

    def forward(self, input):
        # Store and return these features to be fed on the expanding path of Unet
        features = []

        # print("Down iters", len(self.down))
        for f in self.down:
            out = f(input)
            features.append(out)
            # Use the max pooled output for the next layer
            input = self.max_pool(out)

        # print("Down features", [f.shape for f in features])
        return input, features


"""
Defines the generic convolution block used in both the contracting and expansive paths
"""


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            # Convolutions are unpadded
            nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv_block(input)
