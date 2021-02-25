import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from torch.utils.data import DataLoader
from datamanager import ScenarioDatasetMNIST, ToTensor



# Install packages usage
# from utils import install
# install('dgl==0.2')
# import dgl
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


def plot_losses(losses, output_dir):

    train_loss = losses["train_loss"]
    val_loss = losses["validation_loss"]

    fig, ax = plt.subplots()
    ax.plot(train_loss, c="tab:blue", label="Train")
    ax.plot(val_loss, c="tab:orange", label="Validation")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Average Losses")
    ax.legend()

    fig.savefig(os.path.join(output_dir, "losses.png"))


def to_img(x):
    x = x.clamp(0, 1)
    return x


def show_original(ax, img):
    img = to_img(img)
    npimg = img.numpy()
    
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title("Original Images")
    ax.axis("off")


def show_reconstruction(ax, images, model, device):

    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
    
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = utils.make_grid(images[0:64], 8, 8).numpy()

        ax.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        ax.set_title("Reconstructed Images")
        ax.axis("off")
        

def predict_test_data(model, device, scenario, test_dir, output_dir):
    
    # Test Dataset
    test_set = ScenarioDatasetMNIST(root_dir=test_dir, partition="test", scenario=scenario, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0)

    test_data = iter(test_loader).next()
    images = test_data["images"]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # First visualise the original images
    show_original(ax[0], utils.make_grid(images[0:64],8, 8))

    # Reconstruct and visualise the images using the vae
    show_reconstruction(ax[1], images, model, device)

    fig.savefig(os.path.join(output_dir, "test_data_prediction.png"))








