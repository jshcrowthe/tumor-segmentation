import torch
import torch.nn as nn
from layers.pixelshuffle import PixelShuffle
import torch.nn.functional as F

class OneLayer(nn.Module):

    def __init__(self,n_classes = 5):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1,n_classes,kernel_size=1,stride=1)
        )

    def forward(self,x):
        return self.f(x)


class Residual(torch.nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        self.down_sample = None
        self.up_sample  = None

        self.up_sample = PixelShuffle(2)

        n = in_channels//2**3
  
        self.f = nn.Sequential(
            nn.BatchNorm3d(n),
            nn.GELU(),
            nn.Conv3d(n,n,1,1),
            nn.BatchNorm3d(n),
            nn.GELU(),
            nn.Conv3d(n,n,3,1,1),
            nn.BatchNorm3d(n),
            nn.GELU(),
            nn.Conv3d(n,n,1,1),            
        )

        self.final = nn.Sequential(
            nn.BatchNorm3d(n),
            nn.GELU()
        )

    def forward(self,X):
        x = self.up_sample(X)

        res = x

        x = self.f(x)

        x = x+res

        return self.final(x)

class Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        )

        self.layer1 = ResLayer(64,128,3)
        self.layer2 = ResLayer(128,256,4,stride = 2)
        self.layer3 = ResLayer(256,512,4,stride = 2)
        self.layer4 = ResLayer(512,1024,4,stride = 2)

    def forward(self,X):

        l0 = self.pre(X)

        l1 = self.layer1(l0)

        l2 = self.layer2(l1)

        l3 = self.layer3(l2)

        l4 = self.layer4(l3) 

        return l4

class ResBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,downsample = False):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size = 3,stride= stride,padding=1,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels,out_channels,kernel_size = 1,stride= 1,padding=0,bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels,out_channels,kernel_size=1,stride =stride,bias= False),
                nn.BatchNorm3d(out_channels)
                )

    def forward(self,X):

        x = self.f(X)

        res = X
        if self.downsample is not None:
            res = self.downsample(res)

        x = x+res

        return F.gelu(x)

class ResLayer(nn.Module):
    def __init__(self,in_channels,out_channels,n_blocks,stride =1):
        super().__init__()
        self.f = self.make_layer(in_channels,out_channels,n_blocks,stride)
    def make_layer(self,in_channels,out_channels,n_blocks,stride):
        layers = []
        layers.append(ResBlock(in_channels,out_channels,stride=stride,downsample=True))
        
        for i in range(1,n_blocks):

            layers.append(ResBlock(out_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,X):
        x = self.f(X)
        return x


class PixelModel(nn.Module):
    def __init__(self,n_classes=5,shape = (48,64,48)):
        super().__init__()
        self.f = nn.Sequential(
            Backbone(),
            Residual(1024),
            ResLayer(128,512,1,stride = 1),
            Residual(512),
            ResLayer(64,256,1,stride = 1),
            Residual(256),
            ResLayer(32,128,1,stride = 1),
            Residual(128),
            ResLayer(16,64,1,stride = 1),
            Residual(64),
            ResLayer(8,32,1,stride = 1),
            nn.Conv3d(32,n_classes,3,1,1)
        )
    def forward(self,X):
        x = self.f(X)
        x  = nn.functional.interpolate(x,size = (48,64,48))
        return x