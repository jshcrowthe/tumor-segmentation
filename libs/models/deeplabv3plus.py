import torch
from torch import nn
from typing import  List
from torch.nn import functional as F
from .models import Backbone
from .Resnet101 import Resnet101
 
#Paper https://arxiv.org/pdf/1802.02611.pdf

class AtrousConvEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        
        
        self.conv1x1 = torch.nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.conv3x3_rate_6 = torch.nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.conv3x3_rate_12 = torch.nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.conv3x3_rate_18 = torch.nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        self.image_pooling = ImagePooling(in_channels,out_channels)
        
        num_convs = 4
        self.last_conv = nn.Sequential(
                                        nn.Conv3d(num_convs * out_channels, out_channels, 1, bias=False),
                                        nn.BatchNorm3d(out_channels),
                                        nn.ReLU(),
                                        nn.Dropout(),
        )
    def forward(self, x):
        concat = torch.cat((self.conv1x1(x),
                            self.conv3x3_rate_6(x),
                            self.conv3x3_rate_12(x),
                            self.conv3x3_rate_18(x)
                            )  
                           ,dim=1)
        
        return self.last_conv(concat)
        
class ImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.Conv3d =  nn.Conv3d(in_channels, out_channels, 1, bias=False),
        self.batchNorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()        

    def forward(self, x):
        _, _, H, W = x.shape
        output = self.pool2d(x)
        output = self.Conv3d(output)
        output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)
        return output
      
class Decoder (nn.Sequential):
    def __init__(self,num_classes, num_low_level_filters, num_filters) -> None:
        modules = [
            nn.Conv3d(num_low_level_filters+num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(True),
            nn.Conv3d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(True),
            nn.Conv3d(num_filters, num_classes, 1)
        ]
        super().__init__(*modules)
        
class LowLevelFeatures (nn.Sequential):
    def __init__(self,in_channels, num_low_level_filters) -> None:
        modules = [
            nn.Conv3d(in_channels, num_low_level_filters, 1, bias=False),
            nn.BatchNorm3d(num_low_level_filters),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)
        
class DeepLabV3Plus(nn.Module):
    #Encoder-Decoder with Atrous Convolution
    def __init__(self, backbone_name = 'resnetv2_101',
                       num_classes=5, 
                       output_stride=16, 
                       num_filters = 256, 
                       num_low_level_filters=48 #or 32
                ) -> None:
        
        super().__init__()
 
        #ResNet-101           
        self.backbone = Resnet101()
        
        channels = [256, 2048]
        self.lowlevelfeatures = LowLevelFeatures(channels[0],num_low_level_filters)
        self.encoder = AtrousConvEncoder(channels[1], num_filters) 
        self.decoder = Decoder(num_classes=num_classes, num_low_level_filters=num_low_level_filters,num_filters=num_filters)
        
    def forward(self, x):
        
        features = self.backbone(x)        
        x=self.encoder(features[1])
        
        x2=self.lowlevelfeatures(features[0])        
        _, _, x2H, x2W , x2Z= x2.shape      
        x = F.interpolate(x, size=(x2H,x2W,x2Z), 
                          mode='nearest' )
        
        x=torch.cat((x,x2),dim=1)        
        x=self.decoder(x)        
        
        return F.interpolate(x, 
                             #size=(xH,xW), 
                             mode='nearest',
                              
                             scale_factor=4
                             )
        
