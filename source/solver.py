import os, glob, inspect, time, math, torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
import source.loss_functions as lfs
import torch.nn.functional as func
from torch.nn import functional as F
from itertools import chain
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
import source.utils as utils
from time import sleep
import tracemalloc
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."
from pympler import muppy, summary

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

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
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
    plt.savefig(savename)
    plt.close()

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def torch2npy(input):

    input = input.cpu()
    output = input.detach().numpy()
    return output

def training(neuralnet, dataset, epochs, batch_size):
    
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")
    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="results")
    result_list = ["tr_latent", "tr_resotring"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

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
    for name, param in neuralnet.decoder.named_parameters():
        if name.endswith('weight'):
            layer_grad = utils.AverageMeter()
            layer_grad.avg = torch.zeros_like(param)
            ref_grad_dec.append(layer_grad)
    
    for epoch in range(epochs):

        x_tr, x_tr_torch, y_tr, y_tr_torch, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch

        z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))
        
         
        dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

        #z_code = torch2npy(z_code)
        #x_hat = np.transpose(torch2npy(x_hat), (0, 2, 3, 1))
        
        
        if(neuralnet.z_dim == 2):
            latent_plot(latent=z_code, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
        else:
            pca = PCA(n_components=2)
            try:
                pca_features = pca.fit_transform(z_code)
                latent_plot(latent=pca_features, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
            except: pass

        save_img(contents=[x_tr, x_hat, (x_tr-x_hat)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        if(neuralnet.z_dim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val], dtype=np.float32), (1, neuralnet.z_dim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            x_samples = neuralnet.decoder(torch.from_numpy(z_latents).to(neuralnet.device))
            x_samples = np.transpose(torch2npy(x_samples), (0, 2, 3, 1))
            plt.imsave(os.path.join("results", "tr_latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_samples))
        batch_iter = 0
        while(True):
            batch_iter = batch_iter + 1
           
            
            
            
            x_tr, x_tr_torch, y_tr, y_tr_torch, terminator = dataset.next_train(batch_size)

            z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
            x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
            z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

            dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
            dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

            l_tot, l_enc, l_con, l_adv = \
                lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, \
                dis_x, dis_x_hat, features_real, features_fake)

            
            
            x_hat_copy = x_hat.clone()
            x_hat_copy = x_hat_copy.permute(0,2,3,1)
           # x_tr_copy = x_tr.clone().detach()
           # x_tr_copy.requires_grad_()
            x_tr_copy = torch.from_numpy(x_tr)
      
        

            x_tr_copy.requires_grad = True
            recon_loss = func.mse_loss(x_tr_copy,x_hat_copy)
           
          

            
            nlayer = 16
            grad_loss = 0
            target_grad = 0
            k = 0
            for name, param in neuralnet.encoder.named_parameters():
              if name.endswith('weight'):
                  
                  target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
                  
                  grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_enc[k].avg.view(-1,1), dim = 0).item()
                  
                  k = k + 1
                  
              if k == nlayer:
               # print("Gradloss in encoder is")
              #  print(grad_loss)
                break
                  
            j = 0      
            for name, param in neuralnet.decoder.named_parameters():
              if name.endswith('weight'):
                  
                  target_grad = torch.autograd.grad(recon_loss, param, create_graph = True)[0]
                  
                  grad_loss = grad_loss + -1*func.cosine_similarity(target_grad.view(-1,1), ref_grad_dec[j].avg.view(-1,1), dim = 0).item()
                  
                  j = j + 1
              if j == nlayer:
               # print("Gradloss in decoder is")
               # print(grad_loss)
                break
                
             
                

            nlayer = 16
            grad_loss = grad_loss/nlayer
            if ref_grad_enc[0].count == 0:
              print("Inside ref_grad count")
              grad_loss = torch.FloatTensor([0.0]).to(device)
            else:
              grad_loss = grad_loss / nlayer
            
  

            l_grad = grad_loss
            
            
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
            i = 0
            for (name, param) in neuralnet.decoder.named_parameters():
              if name.endswith('weight'):
                ref_grad_dec[i].update(param.grad, 1)
                i = i + 1

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
            
            print("Percentage of available memory")
            print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

            current, peak = tracemalloc.get_traced_memory()

            print("Current memory usage is MB")
            print(current/10**6)
            print("Peak was MB")
            print(peak/10**6)
              
            list_enc.append(l_enc)
            list_con.append(l_con)
            list_adv.append(l_adv)
            list_tot.append(l_tot)
            list_grad.append(l_grad)

            writer.add_scalar('GANomaly/restore_error', l_enc, iteration)
            writer.add_scalar('GANomaly/restore_error', l_con, iteration)
            writer.add_scalar('GANomaly/kl_divergence', l_adv, iteration)
            writer.add_scalar('GANomaly/L_grad', grad_loss, iteration)
            writer.add_scalar('GANomaly/loss_total', l_tot, iteration)
            
            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Enc:%.3f, Con:%.3f, Adv:%.3f, Grad:%3f, Total:%.3f" \
            %(epoch, epochs, iteration, l_enc, l_con, l_adv,grad_loss, l_tot))
        for idx_m, model in enumerate(neuralnet.models):
            torch.save(model.state_dict(), PACK_PATH+"/runs/params-%d" %(idx_m))

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_enc, xlabel="Iteration", ylabel="Enc Error", savename="l_enc")
    save_graph(contents=list_con, xlabel="Iteration", ylabel="Con Error", savename="l_con")
    save_graph(contents=list_adv, xlabel="Iteration", ylabel="Adv Error", savename="l_adv")
    save_graph(contents=list_grad, xlabel="Iteration", ylabel="Adv Error", savename="l_grad")
    save_graph(contents=list_tot, xlabel="Iteration", ylabel="Total Loss", savename="l_tot")

def test(neuralnet, dataset):

    param_paths = glob.glob(os.path.join(PACK_PATH, "runs", "params*"))
    param_paths.sort()

    if(len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH+"/runs/params-%d" %(idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(PACK_PATH+"/runs/params-%d" %(idx_p)))
            neuralnet.models[idx_p].eval()

    print("\nTest...")

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    scores_normal, scores_abnormal = [], []
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1) # y_te does not used in this prj.

        z_code = neuralnet.encoder(x_te_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

        dis_x, features_real = neuralnet.discriminator(x_te_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

        l_tot, l_enc, l_con, l_adv = \
            lfs.loss_ganomaly(z_code, z_code_hat, x_te_torch, x_hat, \
            dis_x, dis_x_hat, features_real, features_fake)
        score_anomaly = l_con.item()

        if(y_te == 1): scores_normal.append(score_anomaly)
        else: scores_abnormal.append(score_anomaly)

        if(terminator): break

    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    print("Noraml  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnoraml  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    outbound = normal_avg + (normal_std * 3)
    print("Outlier boundary of normal data: %.5f" %(outbound))

    histogram(contents=[scores_normal, scores_abnormal], savename="histogram-test.png")

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_code_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1) # y_te does not used in this prj.

        z_code = neuralnet.encoder(x_te_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

        dis_x, features_real = neuralnet.discriminator(x_te_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

        l_tot, l_enc, l_con, l_adv = \
            lfs.loss_ganomaly(z_code, z_code_hat, x_te_torch, x_hat, \
            dis_x, dis_x_hat, features_real, features_fake)
        score_anomaly = l_con.item()

        z_code = torch2npy(z_code)
        x_hat = np.transpose(torch2npy(x_hat), (0, 2, 3, 1))

        loss4box[y_te[0]].append(score_anomaly)

        if(z_code_tot is None):
            z_code_tot = z_code
            y_te_tot = y_te
        else:
            z_code_tot = np.append(z_code_tot, z_code, axis=0)
            y_te_tot = np.append(y_te_tot, y_te, axis=0)

        outcheck = score_anomaly > outbound
        fcsv.write("%d, %.3f, %r\n" %(y_te, score_anomaly, outcheck))

        [h, w, c] = x_te[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_hat[0]
        canvas[:, w*2:, :] = (x_te[0]-x_hat[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")

    if(neuralnet.z_dim == 2):
        latent_plot(latent=z_code_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent.png"))
    else:
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(z_code_tot)
        latent_plot(latent=pca_features, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent.png"))
