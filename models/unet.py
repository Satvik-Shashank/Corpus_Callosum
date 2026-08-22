"""Reusable 2-D U-Net for a future validated CC segmentation dataset."""
import torch
from torch import nn

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(nn.Conv2d(in_channels,out_channels,3,padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,out_channels,3,padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """Four-level 2-D U-Net. `out_channels=2` is suitable for binary CC labels."""
    def __init__(self, in_channels: int = 1, out_channels: int = 2, features: tuple[int,...] = (32,64,128,256)) -> None:
        super().__init__(); self.down=nn.ModuleList(); previous=in_channels
        for feature in features: self.down.append(DoubleConv(previous,feature)); previous=feature
        self.pool=nn.MaxPool2d(2); self.bottleneck=DoubleConv(features[-1],features[-1]*2); self.up=nn.ModuleList(); previous=features[-1]*2
        for feature in reversed(features): self.up.extend([nn.ConvTranspose2d(previous,feature,2,2),DoubleConv(feature*2,feature)]); previous=feature
        self.head=nn.Conv2d(features[0],out_channels,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips=[]
        for block in self.down: x=block(x); skips.append(x); x=self.pool(x)
        x=self.bottleneck(x)
        for index in range(0,len(self.up),2):
            x=self.up[index](x); skip=skips[-(index//2+1)]
            if x.shape[-2:] != skip.shape[-2:]: x=nn.functional.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)
            x=self.up[index+1](torch.cat((skip,x),dim=1))
        return self.head(x)
