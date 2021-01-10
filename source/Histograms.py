# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:50:36 2021

@author: 
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# set seed for reproducing
np.random.seed(42)
n = 5000
mean_mu1 = 60
sd_sigma1 = 15
data1 = np.random.normal(mean_mu1, sd_sigma1, n)
mean_mu2 = 80
sd_sigma2 = 15
data2 = np.random.normal(mean_mu2, sd_sigma2, n)

plt.figure(figsize=(8,6))
plt.hist(data1, bins=100, alpha=0.5, label="data1")
plt.hist(data2, bins=100, alpha=0.5, label="data2")

plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)
plt.title("Multiple Histograms with Matplotlib")
plt.legend(loc='upper right')
plt.savefig("overlapping_histograms_with_matplotlib_Python.png")


#This code prints overlapping histograms given a data matrix,
#where rows are observations and labels is the associated class

def Histograms(Data, labels):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.where(labels == labelclass)
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=100, alpha=0.5, label=label_hist)
    plt.title("Several classes of Histograms")
    plt.legend(loc='upper right')
    plt.savefig("Multiple Histograms.png")
    
    


