import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self,channels_img,features) -> None:
        super(Discriminator,self).__init__()
        self.discNet=nn.Sequential( 
            nn.Conv2d(channels_img,features,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._disc_block(features,features*2,kernel_size=4,stride=2,padding=1),
            self._disc_block(features*2,features*4,kernel_size=4,stride=2,padding=1),
            self._disc_block(features*4,features*8,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(features*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
            )
        
    def _disc_block(self,in_channels,out_channels,kernel_size
              ,stride,padding):
        """
        This is the building block of Discriminator model
        """
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self,image):
        return self.discNet(image)