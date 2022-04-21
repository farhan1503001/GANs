import torch 
torch.manual_seed(42)
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
from tqdm.notebook import tqdm

from torch.nn.modules.activation import LeakyReLU
#creating a common dicriminator block
def disciriminator_block(in_channels,out_channels,kernel_size,stride):
  return nn.Sequential(
     nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride),
     nn.BatchNorm2d(num_features=out_channels),
     nn.LeakyReLU(negative_slope=0.2)
  )
  
#Now creating our discreminator model
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()

    self.conv_block_1=disciriminator_block(1,16,(3,3),2)
    self.conv_block_2=disciriminator_block(16,32,(5,5),2)
    self.conv_block_3=disciriminator_block(32,64,(5,5),2)
    #now flatten layer
    self.flatten=nn.Flatten()
    self.linear=nn.Linear(64,1)

  def forward(self,image):
    x1=self.conv_block_1(image)
    x2=self.conv_block_2(x1)
    x3=self.conv_block_3(x2)

    x4=self.flatten(x3)
    x5=self.linear(x4)

    return x5