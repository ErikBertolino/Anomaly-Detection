
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)





def image2vec(image):
    return image.flatten()



#This function produces three plots: A cumulative-variance plot, a 2D dimensional plot
#and a 3D dimensional plot.

def PCAPlots(Data, labels):
    
    # Standardizing the features
    
    #2D dimensional plot
    
    X = np.array([image2vec(im) for im in Data])
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1) 
    scatter = ax.scatter(principalComponents[:,0],principalComponents[:,1], c = labels, label = labels)
    
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Two dimensional PCA plot of Latent Space', fontsize = 20)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
    ax.add_artist(legend1)
    plt.savefig("2D PCA PLOT - Latent Space")
    plt.show()
    plt.close()

    #3D dimensional plot
    
    X = np.array([image2vec(im) for im in Data])
    X= StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    fig = plt.figure()
    axes = plt.axes(projection='3d')
    scatter = axes.scatter3D(principalComponents[:,0], principalComponents[:,1], principalComponents[:,2], c=labels)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
    axes.set_xlabel('Principal Component 1')
    axes.set_ylabel('Principal Component 2')
    axes.set_zlabel('Principal Component 3')
    axes.set_title('3-dimesional PCA Plot of Latent Space')
    fig.add_axes(axes)
    
    plt.savefig("3D PCA PLOT of Latent Space final.png")
    plt.show()
    plt.close()
    
    #Cumulative-variance plot
    
    
    X_3 = np.array([image2vec(im) for im in Data])
    
    X_3 = StandardScaler().fit_transform(X_3)
    pca = PCA(n_components=X.shape[1])
    principalComponents = pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance');
    plt.title('Cumulative variance explained plot')
   
    plt.savefig('Cumulative variance explained plot - latent space.png')
    plt.show()
    plt.close()

    return None




def UMAPplot(Data, labels):
    
    
    return None









#Saves data in .csv
def SaveData(Data, LayerIndexes, name):
    
#Data is a matrix with dimensions W x N, where W is the number of weight in the
#model and N is the number of data points

#Indexes is a matrix with dimensions L x 1 which designates which dimensions corresponds to which dimensions
#this is done to identify which layers could be interesting to investigate further

#name is a string of current date and time

np.savetxt("LgradWeights.csv",Data,delimiter=",")    

np.savetxt("LayerIndexes.csv",LayerIndexes,delimiter=",")
    
return None
    
    
    
    
    
    
    
    
    
    
    return None