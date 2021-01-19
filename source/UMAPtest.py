# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:33:50 2021

@author: B
"""
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist.data)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)

plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP")
plt.savefig("MNIST - UMAP.png")
plt.show()


def UMAPplot(data, labels):

    typesOfLabels = np.unique(labels)

    colors = cm.rainbow(np.linspace(0, 1, len(typesOfLabels)))
    for label in typesOfLabels:
        indexes = np.where(np.array(labels) == label)
        length = len(indexes[0])
        plt.scatter(
            embedding[indexes, 0], embedding[indexes, 1], c=np.repeat(label, length)
        )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set(context="paper", style="white")
    labels = labels.astype(int)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
    plt.setp(ax, xticks=[], yticks=[])

    plt.title("Latent space embedded into two dimensions by UMAP")
    plt.savefig("Latent Space - UMAP.png")
