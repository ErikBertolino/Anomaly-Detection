import pickle, os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from datamanager import ScenarioDatasetMNIST, ToTensor


## Variational Autoencoder model and utils functions
#
# Sources: 
#   https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
#   https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/


class Encoder(nn.Module):
    def __init__(self, latent_dims=2, capacity=64):
        super(Encoder, self).__init__()
        self.capacity = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.capacity, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=self.capacity, out_channels=self.capacity*2, kernel_size=4, stride=2, padding=1) # out: 2*c x 7 x 7
        self.fc_mu = nn.Linear(in_features=self.capacity*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.capacity*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims=2, capacity=64):
        super(Decoder, self).__init__()
        self.capacity = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=self.capacity*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.capacity*2, out_channels=self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.capacity, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=2, capacity=64):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims=latent_dims, capacity=capacity)
        self.decoder = Decoder(latent_dims=latent_dims, capacity=capacity)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
def vae_loss(recon_x, x, mu, logvar, variational_beta=1.0):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + variational_beta * kl_divergence


def train( 
    training_dir, 
    validation_dir, 
    model_dir, 
    output_dir, 
    scenario=3,
    latent_dims=2, 
    capacity=64, 
    learning_rate=1e-3, 
    weight_decay=1e-5, 
    batch_size=64, 
    epochs=100, 
    use_gpu=False):
    
    # Training Dataset
    train_set = ScenarioDatasetMNIST(root_dir=training_dir, partition="train", scenario=scenario, transform=ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Validation Dataset
    val_set = ScenarioDatasetMNIST(root_dir=validation_dir, partition="validation", scenario=scenario, transform=ToTensor())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Define the model
    vae = VariationalAutoencoder(latent_dims, capacity)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    vae = vae.to(device)

    # Print model information   
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    # Define optimizer
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    train_loss_avg = []
    val_loss_avg = []

    print('Training ...')
    for epoch in range(epochs):

        # set to training mode
        vae.train()

        train_loss_avg.append(0)
        num_train_batches = 0

        for train_batch in train_loader:
            
            image_batch = train_batch["images"].to(device)
            labels_batch = train_batch["labels"].to(device)
            
            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
            
            # reconstruction error
            train_loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
            
            # backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += train_loss.item()
            num_train_batches += 1
            
        train_loss_avg[-1] /= num_train_batches

        # set to evaluation mode
        vae.eval()
        
        val_loss_avg.append(0) 
        num_val_batches = 0

        for val_batch in val_loader:
            
            with torch.no_grad():
            
                image_batch = val_batch["images"].to(device)
                labels_batch = val_batch["labels"].to(device)
                
                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

                # reconstruction error
                val_loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

                val_loss_avg[-1] += val_loss.item()
                num_val_batches += 1
            
        val_loss_avg[-1] /= num_val_batches

        # output epoch metrics
        print('Epoch [%d / %d] Train Loss: %f; Validation Loss: %f;' % (epoch+1, epochs, train_loss_avg[-1], val_loss_avg[-1]))


    # save losses
    losses = {"train_loss": train_loss_avg, "validation_loss": val_loss_avg}
    with open(os.path.join(output_dir, "losses.pkl"), "wb") as f:
        pickle.dump(losses, f)

    # save model
    torch.save(vae.state_dict(), os.path.join(model_dir, 'model.pt'))

    return vae, losses



def input_fn():
    pass

def model_fn():
    pass

def predict_fn():
    pass






