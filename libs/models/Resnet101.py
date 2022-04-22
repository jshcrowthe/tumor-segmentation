
import torch.nn as nn

import torch.nn.functional as F


#https://arxiv.org/pdf/1512.03385.pdf
class ResBlock101(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,downsample = True):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size = 1,stride= 1,padding=0,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),            
            nn.Conv3d(out_channels,out_channels,kernel_size = 3,stride= stride,padding=1,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),            
            nn.Conv3d(out_channels,out_channels*4,kernel_size = 1,stride= 1,padding=0,bias=False),
            nn.BatchNorm3d(out_channels*4)     
        )
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels,out_channels*4,kernel_size=1,stride =stride,bias= False),
                nn.BatchNorm3d(out_channels*4)
                )

    def forward(self,X):

        x = self.f(X)
        #torch.Size([4, 256, 16, 20, 16])
        res = X.clone()
        if self.downsample is not None:
            res = self.downsample(res)

        x = x+res

        return F.gelu(x)
  
class Resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        )               
        self.layer1 = ResLayer(64,64,3)
        self.layer2 = ResLayer(256,128,4,stride = 2)
        self.layer3 = ResLayer(512,256,23,stride = 2)
        self.layer4 = ResLayer(1024,512,3,stride = 2)
 
    def forward(self,X):

        l0 = self.pre(X)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)        
        l3 = self.layer3(l2)        
        l4 = self.layer4(l3) 
        
        return l1, l4 #only features


class ResLayer(nn.Module):
    def __init__(self,in_channels,out_channels,n_blocks,stride =1):
        super().__init__()
        self.f = self.make_layer(in_channels,out_channels,n_blocks,stride)
        
    def make_layer(self,in_channels,out_channels,n_blocks,stride):
        layers = []
        
        layers.append(ResBlock101(in_channels,out_channels,stride=stride,downsample=True))
        
        for i in range(1,n_blocks):
            layers.append(ResBlock101( out_channels*4,out_channels,downsample=False))

        return nn.Sequential(*layers)

    def forward(self,X):
        x = self.f(X)
        return x
        
