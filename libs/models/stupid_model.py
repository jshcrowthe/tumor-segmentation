import torch
import torch.nn as nn

class OneLayer(nn.Module):

    def __init__(self,n_classes = 5):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1,n_classes,kernel_size=1,stride=1)
        )

    def forward(self,x):
        return self.f(x)
