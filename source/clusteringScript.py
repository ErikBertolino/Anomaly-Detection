# -*- coding: utf-8 -*-

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import umap
import torch
import hdbscan
import pandas as pd 


from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
################################################
#UMAP and HDBSCAN function
################################################



def draw_umap(data, n_neighbors, min_dist, n_components, metric, title):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data)
    plt.title(title, fontsize=18)
    
    
def draw_umap_hdbscan(data, n_neighbors, min_dist, n_components, metric, min_samples, min_cluster_size):
    
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u_data = fit.fit_transform(data);
    
    
    hdbscan_labels = hdbscan.HDBSCAN( min_samples, min_cluster_size).fit_predict( u_data )

    num_clusters_found = hdbscan_labels.max() + 1
    
    clusterer = hdbscan.HDBSCAN( min_samples, min_cluster_size ).fit( u_data )    

    

    clustered = (hdbscan_labels >= 0)
    
    ratio_clustered = np.sum(clustered) / u_data.shape[0]
    
    #if q_plot == 1:
    #    fig = plt.figure()
    #    if n_components == 1:
    #        ax = fig.add_subplot(111)
    #        ax.scatter(u_data[:,0], range(len(u_data)), c=data)
    #    if n_components == 2:
    #        ax = fig.add_subplot(111)
    #        ax.scatter(u_data[:,0], u_data[:,1], c=data)
    #    if n_components == 3:
    #        ax = fig.add_subplot(111, projection='3d')
    #        ax.scatter(u_data[:,0], u_data[:,1], u_data[:,2], c=data)
        #plt.title(title, fontsize=18)
    
    return (num_clusters_found,
            ratio_clustered,
            clusterer)


################################################
#Evaluation function for clustering
################################################
def NIDandNVD(Cluster_1, Cluster_2):
    
    l_1, l_2 = len(Cluster_1), len(Cluster_2)
    
    if l_1 == 0 or l_2 == 0:
        NID, NVD = 0, 0
    
    if l_1 > 0 and l_2 > 0:
        NID = 0
        NVD = 0
        l = int(max(Cluster_1))
        k = int(max(Cluster_2))
        
        #Creation of table
        N = 0
        table = np.zeros([l,k])
        
        
        #Calculation of table
        for i_1 in range(0,l):
            for i_2 in range(0,k):
                v_1 = Cluster_1 == i_1
                v_2 = Cluster_2 == i_2
                v_1 = v_1.astype(int)
                v_2 = v_2.astype(int)
               
                
                table[i_1,i_2] = np.dot(np.squeeze(v_1),v_2)
                N = N + np.dot(np.squeeze(v_1),v_2)
        
        
        #Calculation of entropy for first cluster
        if(N == 0):
            NID = 0
            NVD = 0
        else:
            entropy_1 = 0
            for i_1 in range(0,l):
                a = np.sum(table[i_1,:])
                if(a == 0):
                    entropy_1 = entropy_1
                else:
                    entropy_1 = entropy_1 + (a/N)*np.log(a/N)
            
            entropy_1 = -entropy_1
            
            #Calculation of entropy for second cluster
            
            entropy_2 = 0
            for i_2 in range(0,k):
                a = np.sum(table[:,i_2])
                
                if(a == 0):
                    entropy_2 = entropy_2
                else:
                    entropy_2 = entropy_2 + (a/N)*np.log(a/N)
                    
            entropy_2 = -entropy_2
            
            #Calculation of joint entropy
            entropy_joint = 0
            for i_1 in range(0,l):
                for i_2 in range(0,k):
                    if(table[i_1,i_2] != 0):
                       entropy_joint = entropy_joint + ((table[i_1,i_2])/N)*np.log((table[i_1,i_2])/N)
                       
            
            
            entropy_joint = -entropy_joint
            
            
            #Calculation of mutual entropy
            
            entropy_mutual = entropy_1 + entropy_2 - entropy_joint
            
            NID = 1-entropy_mutual/max(entropy_1,entropy_2)
            
            NVD = 1-entropy_mutual/entropy_joint
            
       
        
        
    return NID, NVD




def clustering(folderpath):
    
        
    ###############################################
    #Importing data, including ground_truth_labels
    ###############################################
        
    data = torch.load(os.path.join(folderpath,'lastLayerEnc.pt'))
    ground_truth_clustering = pd.read_csv(os.path.join(folderpath,"labels.csv"), header = None)
    ground_truth_clustering = np.asarray(ground_truth_clustering)
    np.squeeze(ground_truth_clustering, axis=None)
    
    ###############################################
    #Setting up Hyperparameters
    ###############################################
    
    
    #We want to save the performance parameters, the clustering labels, the plot
    #and the hyperparameters
    
    hyperParam, performanceParam, clusteringLabels = [], [], []
    
    ##Hyperparameters
    #metric  [ 'euclidean', "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis", "mahalanobis", "wminkowski", "seuclidean", "cosine", "correlation", "hamming", "jaccard", "dice", "kulsinski", "ll_dirichlet", "hellinger", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule" ]:
    metrics = ['euclidean']
    
    #n_neighbors
    n_neighbors_v = range(2,10)
    
    #min_dist
    
    min_dist_v = [0.1, 0.3, 0.5, 0.8,1]
    
    #n_components
    
    n_components_v = range(2,10)
    
    #min_samples
    
    min_samples_v = [ 2,3,5, 10,100]
    
    #min_cluster_size
    
    min_cluster_size_v = [9, 10, 100]
    
    #cluster_selection_epsilon
    
    #cluster_selection_epsilon_v = [0.1, 0.2, 0.3, 0.4, 0.5, 4.0]
    
    #q_plot - 1 if one wants plot, 0 if not.
    q_plot = 1
    
    l = len(n_neighbors_v)*len(min_dist_v)*len(n_components_v)*len(min_samples_v)*len(min_cluster_size_v)*len(metrics)
    k = 0
    ###############################################
    #Main loop
    ###############################################
    for metric in metrics:
        for n_neighbors in n_neighbors_v:
            for min_dist in min_dist_v:
                for n_components in n_components_v:
                    for min_sample in min_samples_v:
                        for min_cluster in min_cluster_size_v:
                                
                                k = k +1
                                print((k/l)*100)
                                print("Percentage done")
                                
                                num_clusters_found, ratio_clustered, clusterer = draw_umap_hdbscan(data,n_neighbors, 
                                                        min_dist, 
                                                        n_components, 
                                                        metric, 
                                                        min_sample, 
                                                        min_cluster)
                                
                                
                                labels = clusterer.labels_
                                
                                clustered = (labels >= 0)
                               
                               # indexes = np.where(clusterer.labels_ != -1)
                                
                                ground_truth_cluster_p = ground_truth_clustering[clustered]
                                clusterer_p = clusterer.labels_[clustered]
                                NID, NVD = NIDandNVD(ground_truth_cluster_p, clusterer_p)
                                clusteringLabels.append(clusterer)
                                hyperParam.append(np.array([n_neighbors, min_dist, n_components, min_sample, min_cluster]))
                                performanceParam.append(np.array([num_clusters_found,NID,NVD,adjusted_rand_score(np.squeeze(ground_truth_clustering[clustered]), labels[clustered]), adjusted_mutual_info_score(np.squeeze(ground_truth_clustering[clustered]), labels[clustered]), ratio_clustered]))
                                
    
    ##############################################
    #Creating folder and Saving results
    ##############################################
    np.savetxt(os.path.join(folderpath,"clusteringLabels.csv"),  
               clusteringLabels, 
               delimiter =", ",  
               fmt ='% s')
    np.savetxt(os.path.join(folderpath,"hyperParam.csv"),  
               hyperParam, 
               delimiter =", ",  
               fmt ='% s')
    np.savetxt(os.path.join(folderpath,"performanceParam.csv"),  
               performanceParam, 
               delimiter =", ",  
               fmt ='% s')