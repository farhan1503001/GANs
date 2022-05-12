# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:59:40 2022

@author: Shoumik
"""

import torch
import torch.nn as nn

def gradient_penalty(critic,real_img,fake_img,device='cpu'):
    #Finding all the required parameters
    #The thesis of using gradient penalty is to interpolate fake and real
    #image using a random number for image called epsilon
    batch_size,channel,Height,Width=real.shape
    epsilon=torch.randn(batch_size,1,1,1).repeat(batch_size,channel,Height,Width).to(device)
    #Now interpolated image
    interpolated_img=epsilon*real_img+(1-epsilon)*fake_img 
    #Now predict scores
    mixed_penalty=critic(interpolated_img)
    #Find out gradient
    gradient=torch.autograd.grad(
        inputs=interpolated_img,
        outputs=mixed_penalty,
        grad_outputs=torch.ones_like(mixed_penalty),
        create_graph=True,
        retain_graph=True
        )[0]
    
    #Now flatten gradient for normalizing
    gradient=gradient.view(gradient.shape[0],-1)
    #Now applying normalization
    gradient_norm=gradient.norm(2,dim=1)
    #Now our final gradient penalty
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return grgradient_penalty
    