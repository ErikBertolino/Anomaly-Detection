# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:27:11 2021

@author: ÄGARE
"""

import umap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

digits = load_digits()

fig, ax_array = plt.subplots(20, 20)
axes = ax_array.flatten()
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray_r')
plt.setp(axes, xticks=[], yticks=[], frame_on=False)
plt.tight_layout(h_pad=0.5, w_pad=0.01)

digits_df = pd.DataFrame(digits.data[:,1:11])
digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
sns.pairplot(digits_df, hue='digit', palette='Spectral')




#The hyperparameters is important for the UMAP. 
def UMAPplot(Data, Labels, path):
    
        
    reducer = umap.UMAP()
    reducer = umap.UMAP(random_state=42)
    reducer.fit(Data)
    
    embedding = reducer.transform(digits.data)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    embedding.shape
    
    plt.scatter(embedding[:, 0], embedding[:, 1], c=Labels, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Latent Space', fontsize=24);
    plt.savefig(path+"/UMAP/umapplot.png")

    return None


















































