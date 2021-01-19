# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 01:33:10 2021

@author: ÄGARE
"""


import os
from datetime import datetime


def folders():
    timenow = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    currentpath = os.getcwd()
    folderpath = os.path.join(currentpath, str(timenow))
    folderpathHist = os.path.join(folderpath, "histograms")
    folderpathBoxplots = os.path.join(folderpath, "boxplots")
    folderpathPCA = os.path.join(folderpath, "PCA_UMAP")
    folderpathClustering = os.path.join(folderpath, "clustering")
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        os.makedirs(folderpathHist)
        os.makedirs(folderpathBoxplots)
        os.makedirs(folderpathPCA)
        os.makedirs(folderpathClustering)
