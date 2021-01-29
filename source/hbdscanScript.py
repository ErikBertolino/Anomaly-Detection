# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:21:21 2021

@author: ÄGARE
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score



sns.set(style='white', rc={'figure.figsize':(10,8)})


#W is a N x F matrix where N is the number of observations, and F is the number of features
#Each observation is the set of neural weights for a (common) neural network.

#LayerIndexes is the indexes that seperates layers. So one can do clustering for each layer
#It is a L x 2 matrix. Each row designated when a layer begins and ends.

#labels is a N x 1 matrix which designates membership in a class. 

#partitioning is a boolean

#Evaluation is done by using rand score and mutual information
def clustering(W, layerIndexes, labels, partitioning):
    
    #Clustering without partitioning with respect to layers
    if(partitioning == False):
    #Doing a hypertoptimization wrt to parameters.
    #
    
        
        #Applying UMAP
        
        clusterable_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42,).fit_transform(W)
    
        #Applying HDBSCAN on UMAP data
        
        assignedLabels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
        ).fit_predict(clusterable_embedding)
    
        #Evaluation of clustering
        
        clustered = (labels >= 0)
    (
        adjusted_rand_score(W[clustered], assignedLabels[clustered]),
        adjusted_mutual_info_score(W[clustered], assignedLabels[clustered])
    )
       

    if(partitioning == True):
        
        

    
    
    
    #Clustering with partitioning up on layers
    
    
    
    
    