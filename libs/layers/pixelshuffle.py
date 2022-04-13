import torch.nn as nn


class PixelShuffle(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class UnPixelShuffle(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, out_depth, out_height, out_width = input.size()
        nOut = channels* self.scale ** 3

        in_depth = out_depth // self.scale
        in_height = out_height // self.scale
        in_width = out_width // self.scale

        input_view = input.contiguous().view(batch_size, channels, self.scale, self.scale, self.scale,  in_depth, in_height, in_width)
        input_view = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        return input_view.view(batch_size, nOut, in_depth, in_height, in_width)