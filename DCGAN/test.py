from Generator import *
from Discriminator import *
from weight_initializer import *

def test():
    N,channels,H,W=8,3,64,64
    noise_dim=100
    x=torch.randn((N,channels,H,W))
    disc=Discriminator(channels_img=channels,features=8)
    weight_initializer(disc)
    
    assert disc(x).shape==(N,1,1,1)
    
    noise=torch.randn((N,noise_dim,1,1))
    gen=Generator(noise_dim,channels,features=8)
    weight_initializer(gen)
    assert gen(noise).shape==(N,channels,H,W)
    
test()