import torch
import numpy as np
import tensorflow as tf

import torchvision.transforms as transforms

from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, normalize, data, inlier_classes, size):
        #Nota Bene: data is an integer: 1,2,3, or 4
        print("\nInitializing Dataset...")

        self.normalize = normalize
        if(data == 1):
            print("MNIST")
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
            
        if(data == 2):
            print("fMNIST")
            fashion_mnist = tf.keras.datasets.fashion_mnist
            (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
            
        if(data == 3):
            print("CIFAR-10")
            cifar_10 = tf.keras.datasets.cifar10
            (x_tr, y_tr), (x_te, y_te) = cifar_10.load_data()
            
        if(data == 4):
            print("CIFAR-100")
            cifar_100 = tf.keras.datasets.cifar100
            (x_tr, y_tr), (x_te, y_te) = cifar_100.load_data()
            
        self.x_vd, self.y_vd = None, None #We will fill these two later
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)

        self.split_dataset(inlier_classes, size, data)

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te, self.idx_vd = 0, 0, 0



        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        self.num_class = (y_te.max()+1)

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
        print("Class  %d" %(self.num_class))
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

    def split_dataset(self, inlier_classes, size, data):
        
        #Train - contains solely inliers
        
        #Valid - contains solely inliers, different from Train
        
        #Test - contrains inliers and outliers. Inliers different from Train and Valid.
        
        size = int(size)
        x_tot = np.append(self.x_tr, self.x_te, axis=0)
        y_tot = np.append(self.y_tr, self.y_te, axis=0)
        
        print("Inlier classes are:")
        print(*inlier_classes)
        print("Splitting dataset")
    
        
        x_normal, y_normal = None, None
        x_abnormal, y_abnormal = None, None
        k, l = 0, 0
        for yidx, y in enumerate(y_tot):
            
            if(k > 4*size and l > 4*size): break
            
            x_tmp = np.expand_dims(x_tot[yidx], axis=0)
            y_tmp = np.expand_dims(y_tot[yidx], axis=0)

            if(y in inlier_classes): # as normal
                if(x_normal is None):
                    x_normal = x_tmp
                    y_normal = y_tmp
                else:
                    if(x_normal.shape[0] < 4*size):
                        k = k + 1
                        x_normal = np.append(x_normal, x_tmp, axis=0)
                        y_normal = np.append(y_normal, y_tmp, axis=0)

            else: # as abnormal
                if(x_abnormal is None):
                    x_abnormal = x_tmp
                    y_abnormal = y_tmp
                else:
                    if(x_abnormal.shape[0] < 4*size):
                        l = l + 1
                        x_abnormal = np.append(x_abnormal, x_tmp, axis=0)
                        y_abnormal = np.append(y_abnormal, y_tmp, axis=0)
                        
                        
           # if(not(x_normal is None) and not(x_abnormal is None)):
          #      if((x_normal.shape[0] >= 2*size) and x_abnormal.shape[0] >= 2*size+1 ): break

        self.x_tr, self.y_tr = x_normal[:2*size], y_normal[:2*size]
        
        self.x_vd, self.y_vd = x_normal[2*size:3*size], y_normal[2*size:3*size]
        self.x_te, self.y_te = x_normal[3*size:4*size], y_normal[3*size:4*size]
        
        #Some classes are picked out at random for use in the validation stage
        #These classes will reappear in the testing stage, but it will be new
        #data instances
        
        print(len(x_abnormal))
        print(len(x_normal))
        classes = np.unique([y_abnormal])
        #length = len(classes)
        #rndm = np.random.permutation(classes)
        
       # h = int(round(length/2))
        
        #rndm = rndm[h:]
        
        print("Outlier classes used in validation : " )
        #print(rndm)
        
        IndexList = np.asarray([])
        for label in classes:
            indexes = np.where(y_abnormal==label)
            indexes = np.asarray(indexes)
            indexes = indexes.astype(int)
            IndexList = np.append(IndexList,indexes)
            
            
        IndexList = IndexList.astype(int)
        #self.x_vd = x_abnormal[IndexList]
        #self.y_vd = y_abnormal[IndexList]
        
       # v=np.array([range(1,2*size)])
       # v = np.delete(v,IndexList)
        
        self.x_te = np.append(self.x_te, x_abnormal[IndexList], axis = 0) #Adding abnormal 
        self.y_te = np.append(self.y_te, y_abnormal[IndexList], axis = 0)


    #TODO: Remake such that the validation set only contains inlier images - every datapoint should belong to only one set in the split.
        
    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
        x_tr = np.expand_dims(x_tr, axis=3)

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
            x_tr = np.expand_dims(x_tr, axis=3)

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)
            
        if(len(x_tr.shape) == 5):
            x_tr_torch = torch.from_numpy(np.transpose(x_tr, (0, 1, 2, 4, 3)))
            y_tr_torch = torch.from_numpy(y_tr)
        if(len(x_tr.shape) == 4):
            x_tr_torch = torch.from_numpy(np.transpose(x_tr, (0, 3, 1, 2)))
            y_tr_torch = torch.from_numpy(y_tr)

        return x_tr, x_tr_torch, y_tr, y_tr_torch, terminator
    
    
    
    def next_validate(self, batch_size=1):
        
        start, end = self.idx_vd, self.idx_vd+batch_size
        x_vd, y_vd = self.x_vd[start:end], self.y_vd[start:end]
        x_vd = np.expand_dims(x_vd, axis=3)
        
        
        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_vd = 0
        else: self.idx_vd = end

        
        
        if(self.normalize):
            min_x, max_x = x_vd.min(), x_vd.max()
            x_vd = (x_vd - min_x) / (max_x - min_x)
            
        if(len(x_vd.shape) == 5):
            x_vd_torch = torch.from_numpy(np.transpose(x_vd, (0, 1, 2, 4, 3)))
            y_vd_torch = torch.from_numpy(y_vd)

        if(len(x_vd.shape) == 4):
            x_vd_torch = torch.from_numpy(np.transpose(x_vd, (0, 3, 1, 2)))
            y_vd_torch = torch.from_numpy(y_vd)

        return x_vd, x_vd_torch, y_vd, y_vd_torch, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]
        x_te = np.expand_dims(x_te, axis=3)

        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)
            
        if(len(x_te.shape) == 5):
            x_te_torch = torch.from_numpy(np.transpose(x_te, (0, 1, 2, 4, 3)))
            y_te_torch = torch.from_numpy(y_te)
        

        if(len(x_te.shape) == 4):
            x_te_torch = torch.from_numpy(np.transpose(x_te, (0, 3, 1, 2)))
            y_te_torch = torch.from_numpy(y_te)
            
        return x_te, x_te_torch, y_te, y_te_torch, terminator
