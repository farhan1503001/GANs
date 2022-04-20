import torch
import torch.nn as nn
import numpy as np

def weight_initializer(model):
    """
    We will initialize the model's weight with normalized form
    not the random weights which are used while creating model instances
    """
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
        if isinstance(m,nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
            
        if isinstance(m,nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
            
    