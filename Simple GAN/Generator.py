import torch 
torch.manual_seed(42)
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
from tqdm.notebook import tqdm

from torch.nn.modules.activation import LeakyReLU


z_dim=64
def generator_block(in_channels,out_channels,kernel_size,stride,final_layer=False):
  if final_layer==True:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride),
        nn.Tanh()
    )
  return nn.Sequential(
      nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )
  
#Now creating generator model class
class Generator(nn.Module):
  def __init__(self,noise_dim) -> None:
      super(Generator,self).__init__()

      self.noise_dim=noise_dim
      self.transpose_block1=generator_block(noise_dim,256,(3,3),2)
      self.transpose_block2=generator_block(256,128,(4,4),1)
      self.transpose_block3=generator_block(128,64,(3,3),2)
      #final_block
      self.final_block=generator_block(64,1,(4,4),2,final_layer=True)

  def forward(self,noise_vector):
    #Now first we have to change input vector dimension
    #from (bs,noise_dim)__> It will become (bs,noise_dim,1,1)
    x=noise_vector.view(-1,self.noise_dim,1,1)

    x1=self.transpose_block1(x)
    x2=self.transpose_block2(x1)
    x3=self.transpose_block3(x2)

    x4=self.final_block(x3)
    return x4