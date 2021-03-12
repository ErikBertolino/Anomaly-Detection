import os, glob, inspect, time, math, torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
import source.loss_functions as lfs
import torch.nn.functional as func
import pickle

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter
import source.utils as utils
import tracemalloc
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from scipy.optimize import brentq
from scipy.interpolate import interp1d


def HistogramsMSE(Data, labels, folderpath):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.asarray(np.where(labels == labelclass))[0]
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=50, alpha=0.5, label=label_hist)
    
    
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.xlim(0, np.amax(Data))
    plt.title("Histograms of MSE")
    plt.legend(loc='upper right')
    plt.savefig(folderpath + "\Multiple Histograms - MSE.png")

def HistogramsEnc(Data, labels, folderpath):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.asarray(np.where(labels == labelclass))[0]
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=500, alpha=0.5, label=label_hist)
    plt.xlabel("Enc")
    plt.ylabel("Frequency")
    plt.xlim(0, np.amax(Data))
    plt.title("Histograms of Enc")
    plt.legend(loc='upper right')
    plt.savefig(folderpath +"\Multiple Histograms - Enc.png")


def HistogramsAdv(Data, labels, folderpath):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.asarray(np.where(labels == labelclass))[0]
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=500, alpha=0.5, label=label_hist)
    plt.xlabel("Adv")
    plt.ylabel("Frequency")
    plt.xlim(0, np.amax(Data))
    plt.title("Histograms of Adv")
    plt.legend(loc='upper right')
    plt.savefig(folderpath + "\Multiple Histograms - Adv.png")
    
    
def HistogramsGrad(Data, labels,folderpath):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.asarray(np.where(labels == labelclass))[0]
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=500, alpha=0.5, label=label_hist)
    plt.title("Histograms of LGrad")
    plt.xlim(0, np.amax(Data))
    plt.legend(loc='upper right')
    plt.savefig(folderpath + "\Multiple Histograms - Lgrad.png")

def HistogramsCustomAnomaly(Data, labels,folderpath):
    differentLabels = np.unique(labels)
    for labelclass in differentLabels:
        indexes = np.asarray(np.where(labels == labelclass))[0]
        data_hist = Data[indexes]
        label_hist = str(labelclass)
        plt.hist(data_hist, bins=500, alpha=0.5, label=label_hist)
    plt.title("Histograms of custom anomaly score")
    plt.xlim(0, np.amax(Data))
    plt.legend(loc='upper right')
    plt.savefig(folderpath + "\Multiple Histograms - custom anomaly score.png")


def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)
        
    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n,folderpathPCA):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    
    savename = '\\PCAPlotOfLatentSpace.png'
    plt.savefig(folderpathPCA + savename)
    plt.close()

def boxplotMSE(contents, folderpath):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    #bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(folderpath + "\MSE.png")
    plt.close()
    
def boxplotEnc(contents, folderpath):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    #bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(folderpath + "\Encoder.png")
    plt.close()

def boxplotAdv(contents, folderpath):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    #bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(folderpath + "\Adverserial.png")
    plt.close()

def boxplotGrad(contents, folderpath):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    #bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(folderpath + "\Lgrad.png")
    plt.close()    



def boxplotCustomAnomaly(contents, folderpath):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
   # bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(folderpath + "\boxplot custom score.png")
    plt.close()

    
def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename + "\Master histogram")
    plt.close()

def save_graph( folderpath, contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    s = folderpath + "\%s.png" %(savename)
    plt.savefig(s)
    plt.close()

def torch2npy(input):

    input = input.cpu()
    output = input.detach().numpy()
    return output



def roc(labels, scores, folderpath, name):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

   
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
    plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with ' + name)
    plt.legend(loc="lower right")
    name = name + 'ROC.png'
    plt.savefig(folderpath + name)
    plt.close()

    return roc_auc



def training(folderpath, neuralnet, dataset, epochs, batch_size,size, Lgrad_weight,  Enc_weight, Adv_weight, Con_weight):

    torch.float32
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    start_time = time.time()

    iteration = 0
    writer = SummaryWriter()

    test_sq = 20
    test_size = test_sq**2
    list_enc, list_con, list_adv, list_tot, list_grad = [], [], [], [], []
    
    ref_grad_enc = []
    ref_grad_dec = []
    
    
    for name, param in neuralnet.encoder.named_parameters():
        if name.endswith('weight'):
            layer_grad = utils.AverageMeter()
            layer_grad.avg = torch.zeros_like(param)
            ref_grad_enc.append(layer_grad)
   # for name, param in neuralnet.decoder.named_parameters():
   #     if name.endswith('weight'):
   #         layer_grad = utils.AverageMeter()
   #         layer_grad.avg = torch.zeros_like(param)
   #         ref_grad_dec.append(layer_grad)
    AUC = 0
    validation_error = 0
    for epoch in range(epochs):

        x_tr, x_tr_torch, y_tr, y_tr_torch, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch

        if(len(x_tr.shape) == 5):
            x_tr_torch = torch.squeeze(x_tr_torch)
            x_tr_torch = x_tr_torch.permute(0,3,2,1)
            #Shorten the length of the vector.
        


        z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))
        
         
        dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

     
        batch_iter = 0
        while(batch_iter < 32):
            batch_iter = batch_iter + 1
           
            x_tr, x_tr_torch, y_tr, y_tr_torch, terminator = dataset.next_train(batch_size)
            if(len(x_tr.shape) == 5):
                x_tr_torch = torch.squeeze(x_tr_torch)
                x_tr_torch = x_tr_torch.permute(0,3,2,1)
            
          
            z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
           
            x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
           
            z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))
           

            dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
            
            dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))
           

            l_tot, l_enc, l_con, l_adv = \
                lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, \
                dis_x, dis_x_hat, features_real, features_fake,  Lgrad_weight, Enc_weight, Adv_weight, Con_weight, False)
           
            
            x_hat_copy = x_hat.clone()
            x_hat_copy = x_hat_copy.permute(0,2,3,1)
           # x_tr_copy = x_tr.clone().detach()
           # x_tr_copy.requires_grad_()
            x_tr_copy = torch.from_numpy(x_tr)
      
        

            x_tr_copy.requires_grad = True
            x_tr_copy = x_tr_copy.to(neuralnet.device)
            if(len(x_tr_copy) == 5):
                x_tr_copy = torch.squeeze(x_tr_copy)
            recon_loss = func.mse_loss(x_tr_copy,x_hat_copy)
           
          
            
          
            #This is for evaluation of gradloss, which is a bit more cumbersome.
            nlayer = 16
            grad_loss = 0
            target_grad = 0
            k = 0
            
            
            
            if(ref_grad_enc[0].count == 0):
              print("Inside ref_grad count")
              grad_loss = torch.FloatTensor([0.0]).to(device)
            else:
                for name, param in neuralnet.encoder.named_parameters():
                    if name.endswith('weight'):
                  
                        target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
                        target_grad = target_grad.contiguous()
                        #ref_grad_enc[k].contiguous()
                        grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_enc[k].avg.view(-1,1), dim = 0).item()
                        del target_grad
                        k = k + 1
                  
                    if k == nlayer: break
                # print("Gradloss in encoder is")
                #print(grad_loss)
                    
                  
            #    j = 0      
            #    for name, param in neuralnet.decoder.named_parameters():
            #        if name.endswith('weight'):
                  
            #            target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
            #            target_grad = target_grad.contiguous()
            #            grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_dec[j].avg.view(-1,1), dim = 0).item()
            #           del target_grad
            #            j = j + 1
            #       if j == nlayer:  break
                # print("Gradloss in decoder is")
                # print(grad_loss)
               
                
                
            
            
            
            l_tot = l_tot + grad_loss
            neuralnet.optimizer.zero_grad()
            #l_tot.backward(retain_graph = True)
            l_tot.backward()
            # Update the reference gradient
            l = 0
            for (name, param) in neuralnet.encoder.named_parameters():
               if name.endswith('weight'):
                   ref_grad_enc[l].update(param.grad, 1)
                   l = l + 1
           # i = 0
           # for (name, param) in neuralnet.decoder.named_parameters():
           #     if name.endswith('weight'):
           #         ref_grad_dec[i].update(param.grad, 1)
           #         i = i + 1
            
            neuralnet.optimizer.step()
            
            #z_code = torch2npy(z_code)
            #x_hat = np.transpose(torch2npy(x_hat), (0, 2, 3, 1))


           # for i in range(2):
            #    l = 0
             #   for param in neuralnet.encoder.parameters():
              #      l = l +1
               #     if(l==28-2*i):
                #        ref_grad[i].update(param.grad,1)
                
            
            
            print("Batch iteration is: ")
            print(batch_iter)
            
            print("Percentage of RAM available memory")
            print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

            current, peak = tracemalloc.get_traced_memory()

            print("Current memory usage is MB")
            print(current/10**6)
            print("Peak was MB")
            print(peak/10**6)
            
            if(torch.cuda.is_available()):
                
                print('GPU Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
              
            list_enc.append(l_enc.item())
            list_con.append(l_con.item())
            list_adv.append(l_adv.item())
            list_tot.append(l_tot.item())
            list_grad.append(grad_loss)
            
            writer.add_scalar('GANomaly/restore_error', l_enc.item(), iteration)
            writer.add_scalar('GANomaly/restore_error', l_con.item(), iteration)
            writer.add_scalar('GANomaly/kl_divergence', l_adv.item(), iteration)
            writer.add_scalar('GANomaly/L_grad', grad_loss, iteration)
            writer.add_scalar('GANomaly/loss_total', l_tot.item(), iteration)
            
            
            
            if(epoch % 10 == 0 and epoch > 1):
                
                
                AUC_new, validation_error_new = validation(neuralnet, dataset,size, Lgrad_weight, Enc_weight, Adv_weight, Con_weight)
                
                
                if(validation_error_new < validation_error and validation_error != 0): #This indicates overfitting. It performs worse on the test-set
                    print("Validation error is increasing - indicating overfitting. Cancelling training.")
                    break
                else:
                    validation_error = validation_error_new
                
                
        

        print("Epoch [%d / %d] (%d iteration)  Enc:%.3f, Con:%.3f, Adv:%.3f, Grad:%3f, Total:%.3f" \
            %(epoch, epochs, iteration, l_enc, l_con, l_adv,grad_loss, l_tot))
        del l_tot, l_con, l_adv, l_enc
        del x_tr_copy, x_hat_copy, x_hat
        del z_code, z_code_hat
            

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(folderpath,contents=list_enc, xlabel="Iteration", ylabel="Enc Error", savename="l_enc")
    save_graph(folderpath,contents=list_con, xlabel="Iteration", ylabel="Con Error", savename="l_con" )
    save_graph(folderpath,contents=list_adv, xlabel="Iteration", ylabel="Adv Error", savename="l_adv")
    save_graph(folderpath,contents=list_grad, xlabel="Iteration", ylabel="Adv Error", savename="l_grad")
    save_graph(folderpath,contents=list_tot, xlabel="Iteration", ylabel="Total Loss", savename="l_tot")

    pickle_out = open(folderpath+"\\ref_grad_enc","wb")
    pickle.dump(ref_grad_enc, pickle_out)
    pickle_out.close()
    pickle_out = open(folderpath+"\\ref_grad_dec","wb")
    pickle.dump(ref_grad_dec, pickle_out)
    pickle_out.close()

    for idx_m, model in enumerate(neuralnet.models):
        torch.save(model.state_dict(), folderpath+"/params-%d" %(idx_m))

    

    
    return ref_grad_enc
#Validation, meant to curb overfitting
#The coefficient of interest is the AUC score. 




def validation(neuralnet, dataset, size, Lgrad_weight, Enc_weight, Adv_weight, Con_weight):
    
    print("Validating ...")
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    test_size = 32
    epochs = int(np.floor(size/test_size))
   
    
   
    
    AUC = 0
    validation_error = 0
    
    
    
    
    for epoch in range(epochs):

        x_vd, x_vd_torch, y_vd, y_vd_torch, _ = dataset.next_validate(batch_size=test_size) 
        
        if(len(x_vd.shape) == 5):
            x_vd_torch = torch.squeeze(x_vd_torch)
            x_vd_torch = x_vd_torch.permute(0,3,2,1)
            
        z_code = neuralnet.encoder(x_vd_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))
        
         
        dis_x, features_real = neuralnet.discriminator(x_vd_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))


        l_tot, l_enc, l_con, l_adv = \
          lfs.loss_ganomaly(z_code, z_code_hat, x_vd_torch, x_hat, \
          dis_x, dis_x_hat, features_real, features_fake, Lgrad_weight, Enc_weight, Adv_weight, Con_weight, False)

            
        validation_error = validation_error + l_tot.item()
        
        
            #x_hat_copy = x_hat.clone()
          #  x_hat_copy = x_hat_copy.permute(0,2,3,1)
           # x_tr_copy = x_tr.clone().detach()
           # x_tr_copy.requires_grad_()
         #   x_vd_copy = torch.from_numpy(x_vd)
      
        

           # x_vd_copy.requires_grad = True
          #  x_vd_copy = x_vd_copy.to(neuralnet.device)
          #  recon_loss = func.mse_loss(x_vd_copy,x_hat_copy)
            
            
           
            #This is for evaluation of gradloss, which is a bit more cumbersome.
            #nlayer = 16
            #grad_loss = 0
            #target_grad = 0
            #k = 0
            #for name, param in neuralnet.encoder.named_parameters():
            #  if name.endswith('weight'):
                  
            #      target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
                  
            #      grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_enc[k].avg.view(-1,1), dim = 0).item()
                  
           #       k = k + 1
            #      del target_grad
            #      torch.cuda.empty_cache()
            #  if k == nlayer:
               # print("Gradloss in encoder is")
               #print(grad_loss)
             #   break
                  
          #  j = 0      
         #   for name, param in neuralnet.decoder.named_parameters():
          #    if name.endswith('weight'):
                  
         #         target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
                  
         #         grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_dec[j].avg.view(-1,1), dim = 0).item()
                  
         #         j = j + 1
         #         del target_grad
         #         torch.cuda.empty_cache()
         #     if j == nlayer:
               # print("Gradloss in decoder is")
               # print(grad_loss)
        #        break
                
          #  nlayer = 16
          #  grad_loss = grad_loss/nlayer
          #  if ref_grad_enc[0].count == 0:
           #   print("Inside ref_grad count")
           #   grad_loss = torch.FloatTensor([0.0]).to(device)
           ## else:
           #   grad_loss = grad_loss / nlayer
            
  

            #l_grad = grad_loss
            
            #l_tot.backward(retain_graph = True)
           # l_tot.backward()
            # Update the reference gradient
            #l = 0
            #for (name, param) in neuralnet.encoder.named_parameters():
            #  if name.endswith('weight'):
            #    ref_grad_enc[l].update(param.grad, 1)
            #    l = l + 1
        #    i = 0
        #    for (name, param) in neuralnet.decoder.named_parameters():
        #      if name.endswith('weight'):
         #       ref_grad_dec[i].update(param.grad, 1)
         #       i = i + 1


            #Evaluation stage
            
          
            
            
            
            
        
        

    return AUC, validation_error




def test(folderpath,  paths, neuralnet, dataset, inlier_classes, Lgrad_weight, Enc_weight, Adv_weight, Con_weight):


    #Preperation stage

    folderpathHist = paths[0]
    folderpathBoxPlots = paths[1]
    folderpathPCA = paths[2]
    folderPathClustering = paths[3]
    folderpathWeights = paths[4]
    folderpathPlots = paths[5]

     
    l = []
    l.append("These are the inlier classes: ")
    l.append(str(inlier_classes))
    
    MyFile=open(folderpath+'/Inlier Classes.txt','w')
    MyFile.writelines(l)
    MyFile.close()


    param_paths = glob.glob(os.path.join(PACK_PATH, "params*"))
    param_paths.sort()

    if(len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH+"/runs/params-%d" %(idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(folderpath+"\params-%d" %(idx_p)))
            neuralnet.models[idx_p].eval()




    pickle_in = open(folderpath + "\\ref_grad_enc","rb")
    ref_grad_enc = pickle.load(pickle_in)
    pickle_in.close()
   # pickle_in = open("runs/ref_grad_dec","rb")
   # ref_grad_dec = pickle.load(pickle_in)
   # pickle_in.close()





    print("\nTest...")

    
    
    
    print("Producing Histograms")
    #########################################################################
    #Inference stage - for producing histograms
    #########################################################################
    label = []
    
    scores_normal = []
    
    scores_abnormal = []
    
    scores_con = []
    
    scores_enc = []
    
    scores_adv = []
    
    scores_grad = []
    
    scores_custom = []
   
    nlayer = 32
    batch_iter = 0
    
    z_code_tot, y_te_tot = None, None
    
    
    print("Collection of Lgrad")
    #########################################################################
    #Inference stage - for collecting Lgrad weights and subsequent clustering
    #########################################################################
    target_grad_list_enc = [] #These contains Lgrad weights for the enc
    target_grad_list_dec = [] #These contains Lgrad weights for the dec
    
   # labels = [] #Contains labels for y_te
    
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(2) # y_te does not used in this prj.
        
        if(terminator): break
    
    
        if(len(x_te.shape) == 5):
            x_te = x_te.reshape(2,32,32,3)
            x_te_torch = x_te_torch.reshape(2,32,32,3)
            x_te_torch = x_te_torch.permute(0,3,2,1)
        z_code = neuralnet.encoder(x_te_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

        dis_x, features_real = neuralnet.discriminator(x_te_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

        l_tot, l_enc, l_con, l_adv = \
            lfs.loss_ganomaly(z_code, z_code_hat, x_te_torch, x_hat, \
            dis_x, dis_x_hat, features_real, features_fake, Lgrad_weight, Enc_weight, Adv_weight, Con_weight, True)
        
        x_hat_copy = x_hat.clone()
        x_hat_copy = x_hat_copy.permute(0,2,3,1)
           # x_tr_copy = x_tr.clone().detach()
           # x_tr_copy.requires_grad_()
        x_te_copy = torch.from_numpy(x_te)
      
        

        x_te_copy.requires_grad = True
        
        x_te_copy = x_te_copy.to(neuralnet.device)
        recon_loss = func.mse_loss(x_te_copy[0],x_hat_copy[0])
        
        
        label.append(y_te[0])
        label.append(y_te[1])
        
        scores_enc.append(l_enc.detach().numpy())
        scores_con.append(l_con.detach().numpy())
        scores_adv.append(l_adv.detach().numpy())
        
        
        
        print("Batch iteration is")
        print(batch_iter)
        batch_iter = batch_iter + 1
        
        if(z_code_tot is None):
            z_code_tot = z_code.detach()
            z_code_tot = z_code_tot.cpu()
            y_te_tot = y_te
        else:
            z_code = z_code.detach()
            z_code = z_code.cpu()
            z_code_tot = np.append(z_code_tot,z_code, axis=0)
            y_te_tot = np.append(y_te_tot, y_te, axis=0)

        

        
        if(y_te[0] in inlier_classes): 
            scores_normal= np.append(scores_normal, l_tot.detach().numpy()[0]) #This has to be edited, should be able to take a whole list of inlier classes!
        
        else:
            scores_abnormal = np.append(scores_abnormal, l_tot.detach().numpy()[0])
        
        if(y_te[1] in inlier_classes): 
            scores_normal= np.append(scores_normal, l_tot.detach().numpy()[1]) #This has to be edited, should be able to take a whole list of inlier classes!
        
        else:
            scores_abnormal = np.append(scores_abnormal, l_tot.detach().numpy()[1])
        
    
        nlayer = 16
        grad_loss = 0
        target_grad = 0
        #This is for the first data point!
        t = 0
        for name, param in neuralnet.encoder.named_parameters():
            if name.endswith('weight'):
                  
                  target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
#                  target_grad_list_enc.append(target_grad.detach().cpu())
                  target_grad = target_grad.contiguous()
                  grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_enc[t].avg.view(-1,1), dim = 0).item()
                  del target_grad
                  torch.cuda.empty_cache()
                  t = t + 1
                  
            if t == nlayer: break
               # print("Gradloss in encoder is")
               # print(grad_loss)
                
        grad_loss = grad_loss/nlayer

        scores_grad = np.append(scores_grad,grad_loss)      
        
        
        #This for the second data point!
        recon_loss = func.mse_loss(x_te_copy[1],x_hat_copy[1])
        
        grad_loss = 0
        target_grad = 0
            
        t = 0
        for name, param in neuralnet.encoder.named_parameters():
            if name.endswith('weight'):
                  
                  target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
#                  target_grad_list_enc.append(target_grad.detach().cpu())
                  target_grad = target_grad.contiguous()
                  grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_enc[t].avg.view(-1,1), dim = 0).item()
                  del target_grad
                  torch.cuda.empty_cache()
                  t = t + 1
                  
            if t == nlayer: break
               # print("Gradloss in encoder is")
               # print(grad_loss)
                
        grad_loss = grad_loss/nlayer

        scores_grad = np.append(scores_grad,grad_loss)      
                  
       # o = 0      
       # for name, param in neuralnet.decoder.named_parameters():
        #    if name.endswith('weight'):
                  
       #         target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
          #      target_grad_list_dec.append(target_grad.detach().cpu())
       #         target_grad = target_grad.contiguous()
       #         grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_dec[o].avg.view(-1,1), dim = 0).item()
       #         del target_grad
       #         torch.cuda.empty_cache()
                  
      #          o = o + 1
       #     if o == nlayer: break
               # print("Gradloss in decoder is")
               # print(grad_loss)
               
               
                
        
       
            
        scores_custom = np.append(scores_custom, (np.asarray(grad_loss*4 + l_tot.detach().numpy())))

        
        del l_enc, l_con, l_adv
        del x_hat_copy, x_te_copy, x_te_torch
        del z_code, x_hat, z_code_hat
        del dis_x, features_real, dis_x_hat, features_fake, y_te, y_te_torch, recon_loss
            
        current, peak = tracemalloc.get_traced_memory()

        print("Current memory usage is MB")
        print(current/10**6)


    
    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    print("Normal  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnormal  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    outbound = normal_avg + (normal_std * 3)
    print("Outlier boundary of normal data: %.5f" %(outbound))
    
    
    if(neuralnet.z_dim == 2): 
            latent_plot(latent=z_code_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join(folderpathPCA + "\test-latent.png"))
    else:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(z_code_tot)
        k=dataset.num_class
        latent_plot(pca_features, y_te_tot, k, folderpathPCA)

    contents = [scores_normal, scores_abnormal]
    histogram(contents, folderpathHist)
    
    
    
    scores_con = np.asarray(scores_con)
    
    scores_enc = np.asarray(scores_enc)
    
    scores_adv = np.asarray(scores_adv)
    
    label = np.asarray(label)
    print("Creation of histograms")
    HistogramsMSE(scores_con.reshape(-1),label,folderpathHist)
    print("MSE Created")
    HistogramsEnc(scores_enc.reshape(-1),label,folderpathHist)
    print("Enc Created")
    HistogramsAdv(scores_adv.reshape(-1),label,folderpathHist)
    print("Adv Created")
  
    print("Creation of box plots")
    contents = [scores_con.reshape(-1), label]
    boxplotMSE(contents, folderpathBoxPlots)
    print("MSE Created")
    contents = [scores_enc.reshape(-1), label]
    boxplotEnc(contents, folderpathBoxPlots)
    print("Enc Created")
    contents = [scores_adv.reshape(-1), label]
    boxplotAdv(contents, folderpathBoxPlots)
    print("Adv Created")
    
    target_grad_list_dec = np.array(target_grad_list_dec)
    target_grad_list_enc = np.array(target_grad_list_enc)
    
    
    #Needed: formatting
    
    
    
    
    np.savetxt(folderpathWeights + "\LgradWeightsEnc.csv",  
           target_grad_list_enc, 
           delimiter =", ",  
           fmt ='% s')
    np.savetxt(folderpathWeights + "\LgradWeightsDec.csv",  
           target_grad_list_dec, 
           delimiter =", ",  
           fmt ='% s')
    np.savetxt(folderpathWeights + "labels.csv",  
           label, 
           delimiter =", ",  
           fmt ='% s')
    
    scores_grad = np.array(scores_grad)
    scores_custom = np.array(scores_custom)
    HistogramsCustomAnomaly(scores_grad.reshape(-1), label, folderpathHist)
    HistogramsCustomAnomaly(scores_custom.reshape(-1), label, folderpathHist)
    
    
    #########################################################################
    #Inference stage - producing ROC curves from anomaly scores
    #########################################################################
    
    labels_two_classes = []
    
    for i in label:
        if(i in inlier_classes):
            labels_two_classes.append(0)
        else:
            labels_two_classes.append(1)
    
    
    print("ROC Curves")
    roc(np.asarray(labels_two_classes), scores_custom, folderpathPlots, ".\Custom Score")
    roc(np.asarray(labels_two_classes), scores_grad, folderpathPlots, ".\Lgrad Score")
    roc(np.asarray(labels_two_classes), scores_con.reshape(-1), folderpathPlots, ".\Con score")
    roc(np.asarray(labels_two_classes), scores_enc.reshape(-1), folderpathPlots, ".\Enc score")
    roc(np.asarray(labels_two_classes), scores_adv.reshape(-1), folderpathPlots, ".\Adv score")
    
    
    




