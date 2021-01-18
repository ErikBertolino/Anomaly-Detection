import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import tracemalloc
import source.neuralnet as nn
import source.datamanager as dman
import source.solver as solver




def main():

    dataset = dman.Dataset(normalize=FLAGS.datnorm, data=FLAGS.dataset, inlier=FLAGS.Inlier_Classes, Inlier_size=FLAGS.Inlier_size, Outlier_size=FLAGS.Outlier_size)

    if(not(torch.cuda.is_available())): FLAGS.ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and FLAGS.ngpu > 0) else "cpu")
    
    
    tracemalloc.start()
    
    #Intitiating neuralnet
    neuralnet = nn.NeuralNet(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        device=device, ngpu=FLAGS.ngpu, \
        ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr, inlier=FLAGS.Inlier_Classes)
    #Training
    solver.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, Lgrad_weight=FLAGS.Lgrad_weight, split=FLAGS.Split)
    #Validation
    solver.validation(neuralnet=neuralnet, dataset=dataset, split=FLAGS.Split)
    #Testing
    solver.test(neuralnet=neuralnet, dataset=dataset, split=FLAGS.Split,inlier=FLAGS.Inlier_Classes )
    

    #All evaluation occurs in the end of these methods.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--dataset', type=int, default=1, help='1 = MNIST, 2 = fMNIST, 3 = CIFAR10, 4 = CIFAR100')
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size for constructing Neural Network')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=1, help='Training epoch')
    parser.add_argument('--batch', type=int, default=5, help='Mini batch size')
    parser.add_argument('--Inlier_Classes', type=int, default=[1], help='Inlier Classes')
    parser.add_argument('--Split', type=list, default = [0.5, 0.25, 0.25], help = 'Train/Valid/Test Split')
    parser.add_argument('--Lgrad_weight', type=float, default=1, help='Weight for Lgrad')
    parser.add_argument('--Inlier_size', type=int, default=1000, help='Amount of data points in the normal class')
    parser.add_argument('--Outlier_size', type=int, default=1000, help='Amount of data points in the abnormal class')
    
    
    FLAGS, unparsed = parser.parse_known_args()

    main()




