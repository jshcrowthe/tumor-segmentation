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
    def __init__(self, depth=5, num_classes=5) -> None:
        super(Unet, self).__init__()

        self.depth = depth

        # Channels must be passed as integers to torch.nn
        self.channel_dims = [int(16 * 2.0 ** (i)) for i in range(self.depth)]

        self.up_channel = nn.Conv3d(in_channels=1, out_channels=self.channel_dims[0], kernel_size=1)
        self.down = UnetDown(self.channel_dims)
        self.bottom = ConvBlock(self.channel_dims[-1], self.channel_dims[-1])
        self.up = UnetUp(self.channel_dims)

        # Upscale 
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(self.channel_dims[1], self.channel_dims[1]),

        )

        # Final conv to retain the input volume dimensions
        self.final_conv = nn.Conv3d(self.channel_dims[1], num_classes, 1)

    def forward(self, input):
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

        # The feature dimensions are always bigger because they come from the input
        d_delta = D_f - D_i
        h_delta = H_f - H_i
        w_delta = W_f - W_i


        # Return the center cropped volume
        res = features[
            :,
            :,
            int(math.floor(d_delta / 2)) : int(math.floor(-d_delta / 2)),
            int(math.floor(h_delta / 2)) : int(math.floor(-h_delta / 2)),
            int(math.floor(w_delta / 2)) : int(math.floor(-w_delta / 2)),
        ].cuda()

        return res

    def forward(self, input, features):
        features.reverse()

        for i in range(len(self.up) - 1):
            # Perform a transposed convolution to double the volum space
            out = self.conv[i](input)
            down_features = self.center_crop_3d(out, features[i + 1])
            out = torch.cat([out, down_features], dim=1)  # Concatenate on the channels
            input = self.up[i](out) # Perform a forward convolution for the up-tep

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

        for f in self.down:
            out = f(input)
            features.append(out)
            # Use the max pooled output for the next layer
            input = self.max_pool(out)

        return input, features


"""
Defines the generic convolution block used in both the contracting and expansive paths
"""


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            # Convolutions are padded so we don't lose dimensionality with increasing depth
            nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv_block(input)
