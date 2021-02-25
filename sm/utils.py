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


def to_img(x):
    x = x.clamp(0, 1)
    return x


def show_image(img, title, output_dir):
    img = to_img(img)
    npimg = img.numpy()
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    
    plt.savefig(os.path.join(output_dir, "test_original.png"))


def visualise_output(images, model, device, title, output_dir):

    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
    
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = utils.make_grid(images[0:64], 8, 8).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        
        plt.title(title)
        
        plt.savefig(os.path.join(output_dir, "test_reconstruction.png"))


def plot_losses(losses, output_dir):

    train_loss = losses["train_loss"]
    val_loss = losses["validation_loss"]

    plt.plot(train_loss, c="tab:blue", label="Train")
    plt.plot(val_loss, c="tab:orange", label="Validation")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Losses")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "losses.png"))


def predict_test_data(model, device, scenario, test_dir, output_dir):
    
    # Test Dataset
    test_set = ScenarioDatasetMNIST(root_dir=test_dir, partition="test", scenario=scenario, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0)

    test_data = iter(test_loader).next()
    images = test_data["images"]

    # First visualise the original images
    show_image(utils.make_grid(images[0:64],8, 8), "Original Images", output_dir)

    # Reconstruct and visualise the images using the vae
    visualise_output(images, model, device, "Reconstructed Images", output_dir)











