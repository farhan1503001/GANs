import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self,noise_dim,channels_img,features) -> None:
        super(Generator,self).__init__()
        self.gen_net=nn.Sequential(
            self._generator_block(noise_dim,features*16,4,1,0),
            self._generator_block(features*16,features*8,4,2,1),
            self._generator_block(features*8,features*4,4,2,1),
            self._generator_block(features*4,features*2,4,2,1),
            nn.ConvTranspose2d(features*2,channels_img,4,2,1),
            nn.Tanh()
        )
    def _generator_block(self,in_channels,out_channels,kernel_size,strides,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,strides,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
            )
    def forward(self,image):
        """
        Forward propagation function
        """
        return self.gen_net(image)