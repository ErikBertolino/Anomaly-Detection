# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 01:33:10 2021

@author: ÄGARE
"""


import os
from datetime import datetime

    
def folders(folderpath):
    folderpathHist = os.path.join(folderpath, 'histograms')
    folderpathBoxplots = os.path.join(folderpath, 'boxplots')
    folderpathPCA = os.path.join(folderpath, 'PCA_UMAP')
    folderpathClustering = os.path.join(folderpath, 'clustering')
    folderpathWeights = os.path.join(folderpath, 'Weights')
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        os.makedirs(folderpathHist)
        os.makedirs(folderpathBoxplots)
        os.makedirs(folderpathPCA)
        os.makedirs(folderpathClustering)
        os.makedirs(folderpathWeights)