import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from datetime import datetime
import torch
from torchsummary import summary
import tracemalloc
import source.neuralnet as nn
import source.datamanager as dman
import source.solver as solver
#import source.clusteringScript as clusteringScript


def folders(folderpath):
    folderpathHist = os.path.join(folderpath, 'histograms')
    folderpathBoxplots = os.path.join(folderpath, 'boxplots')
    folderpathPCA = os.path.join(folderpath, 'PCA_UMAP')
    folderpathClustering = os.path.join(folderpath, 'clustering')
    folderpathWeights = os.path.join(folderpath, 'Weights')
    folderpathPlots = os.path.join(folderpath, 'Plots')
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        os.makedirs(folderpathHist)
        os.makedirs(folderpathBoxplots)
        os.makedirs(folderpathPCA)
        os.makedirs(folderpathClustering)
        os.makedirs(folderpathWeights)
        os.makedirs(folderpathPlots)
        
    return folderpathHist, folderpathBoxplots, folderpathPCA, folderpathClustering, folderpathWeights, folderpathPlots



def main():

    
    tracemalloc.start()



    #Creates folder 
    
    timenow = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    currentpath = os.getcwd()
    folderpath = os.path.join(currentpath, str(timenow))
    folderpathHist, folderpathBoxPlots, folderpathPCA, folderPathClustering, folderpathWeights, folderpathPlots = folders(folderpath)
    paths = [folderpathHist, folderpathBoxPlots, folderpathPCA, folderPathClustering, folderpathWeights, folderpathPlots]
    #Saves all arguments
    
    
    l = len(FLAGS._get_kwargs())
    
    MyFile=open(folderpath+'/Hyperparameter settings.txt','w')
    
   
    for i in range(l):
        s = str(FLAGS._get_kwargs()[i]) + "\n"
        MyFile.writelines(s)
        
        
    MyFile.close()
    

    dataset = dman.Dataset(normalize=FLAGS.datnorm, data=FLAGS.dataset, inlier_classes=FLAGS.Inlier_Classes, size=FLAGS.Outlier_size)

    device = torch.device("cuda" if (torch.cuda.is_available() and FLAGS.ngpu > 0) else "cpu")
    
    
    
    #Intitiating neuralnet
    neuralnet = nn.NeuralNet(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        device=device, ngpu=FLAGS.ngpu, \
        ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr)
    #Training
    solver.training(folderpath, neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, size=FLAGS.Outlier_size, Lgrad_weight=FLAGS.Lgrad_weight, \
    Enc_weight=FLAGS.Enc_weight, Adv_weight=FLAGS.Adv_weight, Con_weight=FLAGS.Con_weight)
    #Validation
    #solver.validation(neuralnet=neuralnet, dataset=dataset, split=FLAGS.Split)
    #Testing
    folderpath = solver.test(folderpath ,paths, neuralnet=neuralnet, dataset=dataset,inlier_classes=FLAGS.Inlier_Classes,size=FLAGS.Outlier_size, Lgrad_weight=FLAGS.Lgrad_weight, \
    Enc_weight=FLAGS.Enc_weight, Adv_weight=FLAGS.Adv_weight, Con_weight=FLAGS.Con_weight)
    #Clusterng
#    clusteringScript.clustering(folderpath)
    
    #All evaluation occurs in the end of these methods.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--dataset', type=int, default=1, help='1 = MNIST, 2 = fMNIST, 3 = CIFAR10, 4 = CIFAR100')
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size for constructing Neural Network')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=2, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')
    parser.add_argument('--Inlier_Classes', type=int, default=[1], help='Inlier Classes')
    parser.add_argument('--Split', type=list, default = [0.5, 0.25, 0.25], help = 'Train/Valid/Test Split')
    parser.add_argument('--Lgrad_weight', type=float, default=1, help='Weight for Lgrad')
    parser.add_argument('--Enc_weight', type=float, default=1, help='Weight for Encoder')
    parser.add_argument('--Adv_weight', type=float, default=1, help='Weight for Adverserial')
    parser.add_argument('--Con_weight', type=float, default=1, help='Weight for Contextual')
    parser.add_argument('--Inlier_size', type=int, default=10000, help='Amount of data points in the normal class')
    parser.add_argument('--Outlier_size', type=int, default=10000, help='Amount of data points in the abnormal class')
    
    
    FLAGS, unparsed = parser.parse_known_args()

    main()




