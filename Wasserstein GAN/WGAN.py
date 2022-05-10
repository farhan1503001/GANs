# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:59:28 2022

@author: Shoumik
"""

import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets 
import torchvision.transforms as transformers 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
#All the important libraries are imported here
#Now we will define all the constansts
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=2e-4
batch_size=128 
image_size=64
channels=1 #as our current image is grayscale
Z_dimension=100
Num_epochs=5
Discriminator_features=64
Generator_features=64 #Same as image size

#Now we will load the necessary data
#First defining transformer in our image transformations
transformer=transformers.Compose(
    [
     transformers.Resize(image_size),
     transformers.ToTensor(),
     transformers.Normalize(
         [0.5 for _ in range(channels)],[0.5 for _ in range(channels)]
         )
     ]
    )

#Now loading the dataset
dataset=datasets.MNIST(root='dataset/',transform=transformer,download=True)
#Now create insert these dataset into dataloader
mnist_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
#---------------------------$$$$$$------------------------------
#Now loading models
generator=Generator(channels_noise=z_dimension, channels_img=image_size, features_g=Generator_features).to(device)
critic=Discriminator(channels, features_d=Discriminator_features).to(device)
#Model loading complete now initiliazing the weight
initialize_weights(generator)
initialize_weights(critic)
#Now we will start our training section
#Now initializing our  optimizers
gen_optim=optim.RMSprop(generator.parameters(),lr=learning_rate)
critic_optim=optim.RMSprop(critic.parameters(),lr=learning_rate)

#Now just additional stuffs for our tensorboard plotting
fixed_noise=torch.randn(32,z_dimension,1,1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
critic_iteration=5
weight_clip=0.1
generator.train()
critic.train()
for i in range(Num_epochs):
    #Now we will train eatch batch 
    #No label is needed as it is unsupervised
    for index,(images,_) in enumerate(mnist_loader):
        images=images.to(device)
        current_batch_size=images.shape[0]
        #Remember our loss function here is Total_loss=E(critic(real))-E(critic(fake))
        for _ in range(critic_iteration):
            noise=torch.randn(current_batch_size,z_dimension,1,1).to(device)
            #Now generate fake image using noise
            fake_img=generator(noise)
            #Now find out critics result on this fake and real image
            critic_real=critic(images).reshape(-1)
            critic_fake=critic(fake).reshape(-1)
            #Now loss function is here
            #Succesful critic must be able to differentiate between fake and real
            loss_critic=-(tf.mean(critic_real)-tf.mean(critic_fake))
            #Now backproping critic
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_optim.step()   
            
            for p in critic.parameters():
                p.data.clamp_(-weight_clip,weight_clip)
        #Now for generator our loss should be minimum means -loss maximum
        output=critic(fake_img).reshape(-1)
        gen_loss=-tf.mean(output)
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        #Now storing and display
        for index%100==0 and index>0:
            generator.eval()
            critic.eval()
            print(
               f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                 Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
           )
                
            #Now storing the image
            with torch.no_grad():
                fake=generator(noise)
                
                img_real=torchvision.utils.make_grid(images[:32], normalize=True)
                img_grid_fake=torchvision.utils.make_grid(fake[:32],normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
        step+=1
        generator.train()
        critic.train()

