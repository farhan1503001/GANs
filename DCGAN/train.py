from sklearn import datasets
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Discriminator import *
from Generator import *
from weight_initializer import *
#Now defining the hyperparameters
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=2e-4
batch_size=128
img_height=64
img_width=64
channels_img=1
noise_dimension=100
Epochs=10
Discriminator_feature=64
Generator_feature=64

#Now image_transformer or augmentar
transformer=transforms.Compose(
    [
        transforms.Resize(img_height),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]
        )
    ]
)

#Now we will import the dataset
dataset=torchvision.datasets.MNIST(root='dataset/',transform=transformer,download=True)
loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
generator=Generator(noise_dim=noise_dimension,
                    channels_img=channels_img,
                    features=Generator_feature).to(device=device)

discriminator=Discriminator(channels_img,Discriminator_feature).to(device)
#Now initlializing weights
weight_initializer(generator)
weight_initializer(discriminator)

#Now defining our optimizers
optimizer_gen=optim.Adam(generator.parameters(),lr=learning_rate,betas=(0.5,0.999))
optimizer_disc=optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(0.5,0.999))
#Now defining our loss function
criterion=nn.BCELoss()
step=0
#Now creating logg writers
writer_real=SummaryWriter(f'logs/real')
writer_fake=SummaryWriter(f'logs/fake')
#Now creating noise step and starting training
fixed_noise=torch.randn((32,noise_dimension,1,1)).to(device)
#Now starting generator and discriminator training
generator.train()
discriminator.train()

for i in range(Epochs):
    #Now we will iterate through the dataloader
    for index,(real_img,_) in enumerate(loader):
        #Taking real_img to cuda
        real_img=real_img.to(device)
        #Now creating random noise
        noise=torch.randn((batch_size,noise_dimension,1,1)).to(device=device)
        #--------Discriminator portion------
        #Our discriminator loss function is log(D(real))+log(1-D(G(real)))
        #creating fake image using generatord
        fake=generator(noise)
        #finding out log(d(real)) part
        disc_real=discriminator(real_img).reshape(-1)
        disc_real_loss=criterion(disc_real,torch.ones_like(disc_real))
        #now going for log(1-D(G(z))) part
        disc_fake=discriminator(fake.detach()).reshape(-1)
        disc_fake_loss=criterion(disc_fake,torch.zeros_like(disc_fake))
        
        disc_loss=(disc_real_loss+disc_fake_loss)/2.0
        #Now we will backprop discriminator
        discriminator.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()
        
        #----------Generator---loss section----part
        #it is logG(z)
        output=discriminator(fake).reshape(-1)
        gen_loss=criterion(output,torch.ones_like(output))
        generator.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()
        #Now evaluation and visualize
        # Print losses occasionally and print to tensorboard
        if index % 100 == 0:
            print(
                f"Epoch [{i}/{Epochs}] Batch {index}/{len(loader)} \
                  Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real_img[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1