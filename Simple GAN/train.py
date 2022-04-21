import torch 
torch.manual_seed(42)
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm.notebook import tqdm
from Discriminator import *
from Generator import *
from weight_initializer import *
from loss_function import *

#Defining hyperparameters
#Now defining all important constants and parameters
device='cuda'
batch_size=16
noise_dim=64 #will be used for our data
#Now setting optimizer parameters here we will use Adam optimizer
learning_rate=0.0002
beta_1=0.5
beta_2=0.99
#Training parameters
Epochs=20
#Now we will be importing dataset and augmentations
from torchvision import datasets,transforms as Transformer
#Now training augmentation will be created
train_transformer=Transformer.Compose(
    [
      Transformer.RandomRotation((-20,+20)),
      Transformer.ToTensor() #it will convert our data from h,w,c to c,h,w format
    ]
)
#Now we will load the dataset
training_dataset=datasets.MNIST('MNIST/',download=True,transform=train_transformer)
#Now we will try to visualize the image
image,label=training_dataset[5]
plt.imshow(image.squeeze(),cmap='gray')
print(label)
print("Image size is :",image.shape)
#Now checking the dataset size
print("The total number of image present in this dataset: ",len(training_dataset))
print("Each image is of size: ",image.shape)
#Now we will import dataloaders for making batch loading possible
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
#Now batch load the data
train_loader=DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
print("Finding out Total Number of Batches in Loader: ",len(train_loader))
#Now visualizing each batch_shape
dataiter=iter(train_loader)
image,_=dataiter.next()
print("Shape of a batch image: ",image.shape)
# 'show_tensor_images' : function is used to plot some of images from the batch

def show_tensor_images(tensor_img, num_images = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()
#Visualizing tensor images
show_tensor_images(image,num_images=16)

import torch.nn as nn



D=Discriminator()
D.to(device)

#Now calling the generator model
G=Generator(noise_dim=64)
G.to(device)
#Now we will be using normal weights instead of random ones so have 
#to just apply to each of the models
G.apply(weights_init)
D.apply(weights_init)


#Now defining our optimizer
Discriminator_optim=torch.optim.Adam(D.parameters(),lr=learning_rate,betas=(beta_1,beta_2))
Generator_optim=torch.optim.Adam(G.parameters(),lr=learning_rate,betas=(beta_1,beta_2))
for i in range(Epochs):
  total_d_loss=0.0
  total_g_loss=0.0
  for img,label in train_loader:
    #sending image to device
    img=img.to(device)
    #creating noise vector
    noise=torch.randn(batch_size,noise_dim,device=device)
    #Now finding loss and weight update for discriminator
    Discriminator_optim.zero_grad()
    fake_img=G(noise)
    pred_img=D(fake_img)
    D_fake_loss=fake_loss(pred_img)
    #for real loss
    pred_img=D(img)#using real image
    D_real_loss=real_loss(pred_img)
    #finding d loss
    d_loss=(D_fake_loss+D_real_loss)/2.0
    #add d_loss to it's total
    total_d_loss+=d_loss.item()
    d_loss.backward()
    Discriminator_optim.step()


    #Now for generative loss
    Generator_optim.zero_grad()
    noise=torch.randn(batch_size,noise_dim,device=device)
    #generate a fake image
    fake_img=G(noise)
    pred_img=D(fake_img)
    g_loss=real_loss(pred_img)
    #Now add to total loss
    total_g_loss+=g_loss.item()

    g_loss.backward()
    Generator_optim.step()
  avg_d_loss=total_d_loss/len(train_loader)
  avg_g_loss=total_g_loss/len(train_loader)

  print(f"For Epoch {i}_-->G_loss={avg_g_loss}--->D_loss={avg_d_loss}")
  show_tensor_images(fake_img)



# Run after training is completed.
# Now you can use Generator Network to generate handwritten images

noise = torch.randn(batch_size, noise_dim, device = device)
generated_image = G(noise)

show_tensor_images(generated_image)







