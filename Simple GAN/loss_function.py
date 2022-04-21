import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
#==================Loss Functions=============
#Now writing real and fake loss functions for our discriminator and generator
def real_loss(disc_pred):
  #Just defining our loss function
  loss_criteria=nn.BCEWithLogitsLoss()
  #gaining loss value
  ground_truth=torch.ones_like(disc_pred)
  loss=loss_criteria(disc_pred,ground_truth)
  return loss
#Now writing function for fake loss
def fake_loss(disc_pred):
  loss_criterion=nn.BCEWithLogitsLoss()
  ground_truth=torch.zeros_like(disc_pred)
  loss=loss_criterion(disc_pred,ground_truth)
  return loss

