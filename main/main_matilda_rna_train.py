import os
import argparse

import pandas as pd
import numpy as np
from captum.attr import *
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

from learn.model_rna import CiteAutoencoder
from learn.train import train_model
from util import setup_seed, MyDataset,ToTensor, read_h5_data, read_fs_label, get_vae_simulated_data_from_sampling, get_encodings, compute_zscore, compute_log2,save_checkpoint

parser = argparse.ArgumentParser("Matilda")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--augmentation', type=bool, default= True, help='if augmentation or not')

############# for data build ##############
parser.add_argument('--rna', metavar='DIR', default='NULL', help='path to train rna data')
parser.add_argument('--cty', metavar='DIR', default='NULL', help='path to train cell type label')

##############  for training #################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.02, help='init learning rate')

############# for model build ##############
parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')

args = parser.parse_args()
setup_seed(args.seed) ### set random seed in order to reproduce the result
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


mode = "rna_only"
train_rna_data_path = args.rna
train_label_path = args.cty
train_rna_data = read_h5_data(train_rna_data_path)
train_label = read_fs_label(train_label_path)
classify_dim = (max(train_label)+1).cpu().numpy()
nfeatures_rna = train_rna_data.shape[1]
feature_num = nfeatures_rna
train_rna_data = compute_log2(train_rna_data)
train_rna_data = compute_zscore(train_rna_data)
train_data = train_rna_data
train_transformed_dataset = MyDataset(train_data, train_label)
train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)


test_dl = "NULL"

        
print("The dataset is", mode)    
output_v = []
model_save_path = "../trained_model/{}/".format(mode)   
model_save_path_1stage = "../trained_model/{}/simulation_".format(mode)    
save_fs_eachcell = "../output/marker/{}/".format(mode)   

model = CiteAutoencoder(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)

#model = nn.DataParallel(model).to(device) #multi gpu
model = model.to(device) #one gpu
########train model#########
model, acc1, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=args.epochs, classify_dim = classify_dim, best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)
##################prepare to do augmentation##################            
if args.augmentation == True:
    stage1_list = []
    for i in np.arange(0, classify_dim):
        stage1_list.append([i, train_num[i]])
        stage1_df = pd.DataFrame(stage1_list)
    if classify_dim%2==0:
        train_median = np.sort(train_num)[int(classify_dim/2)-1]
    else: 
        train_median = np.median(train_num)
    median_anchor = stage1_df[stage1_df[1] == train_median][0]
    train_major = stage1_df[stage1_df[1] > train_median]
    train_minor = stage1_df[stage1_df[1] < train_median]
    anchor_fold = np.array((train_median)/(train_minor[:][1]))
    minor_anchor_cts = train_minor[0].to_numpy()
    major_anchor_cts = train_major[0].to_numpy()

    index = (train_label == int(np.array(median_anchor))).nonzero(as_tuple=True)[0]
    anchor_data = train_data[index.tolist(),:]
    anchor_label = train_label[index.tolist()]
    new_data = anchor_data 
    new_label = anchor_label

    ##############random downsample major cell types##############
    j=0
    for anchor in major_anchor_cts:     
        anchor_num = np.array(train_major[1])[j]
        N = range(anchor_num)
        ds_index = random.sample(N,int(train_median))
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_data = anchor_data[ds_index,:]
        anchor_label = anchor_label[ds_index]
        new_data = torch.cat((new_data,anchor_data),0)
        new_label = torch.cat((new_label,anchor_label.to(device)),0)
        j = j+1

    ###############augment for minor cell types##################
    j = 0
    for anchor in minor_anchor_cts:
        aug_fold = int((anchor_fold[j]))    
        remaining_cell = int(train_median - (int(anchor_fold[j]))*np.array(train_minor[1])[j])
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_transfomr_dataset = MyDataset(anchor_data, anchor_label)
        anchor_dl = DataLoader(anchor_transfomr_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
        reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label, reconstructed_label),0)
        for i in range(aug_fold-1):
            reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
            reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
            reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = torch.cat((new_data,reconstructed_data),0)
            new_label = torch.cat((new_label,reconstructed_label.to(device)),0)

        reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        #add remaining cell
        N = range(np.array(train_minor[1])[j])
        ds_index = random.sample(N, remaining_cell)
        reconstructed_data = reconstructed_data[ds_index,:]
        reconstructed_label = reconstructed_label[ds_index]
        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label,reconstructed_label.to(device)),0)
        j = j+1               

filename = os.path.join('../trained_model/TEAseq/simulation_model_best.pth.tar')
torch.save({'state_dict': model.state_dict()}, filename)
      
#######load the model #########
model = CiteAutoencoder(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)

#model = nn.DataParallel(model).to(device) #multi gpu
model = model.to(device) #one gpu

############process new data after augmentation###########
train_transformed_dataset = MyDataset(new_data, new_label)
train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)

############## train model ###########
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=int(args.epochs/2),classify_dim=classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)
checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
if os.path.exists(checkpoint_tar):
    checkpoint = torch.load(checkpoint_tar)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("load successfully")
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr/10, epochs=int(args.epochs/2),classify_dim=classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)



    
            

