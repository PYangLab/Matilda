import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import random
import os
import scanpy as sc
from torch.autograd import Variable
import h5py
import scipy

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     os.environ['PYTHONHASHSEED']=str(seed)

def real_label(label_path,classify_dim):
    output_v = []
    label = pd.read_csv(label_path,header=None,index_col=False)  #
    label_real = label.iloc[1:(label.shape[0]),1]
    label_num = read_fs_label(label_path)
    for i in range(classify_dim):    
        temp = label_real[np.array(torch.where(label_num==i)[0][0].cpu()).astype('int32')+1]
        output_v.append(temp)
    return output_v

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):#返回的是tensor
        img, target = self.data[index,:], self.label[index]
        sample = {'data': img, 'label': target}
        return sample

    def __len__(self):
        return len(self.data)
        

class ToTensor(object):
    def __call__(self, sample):
        data,label = sample['data'], sample['label']
        data = data.transpose((1, 0))
        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label),
               }
               
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print(target, pred,"!!!!!!!!!!!!!!!")

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            #print(batch_size,"!!!!")
            res.append(correct_k.mul_(100.0 / batch_size))
            
        return res
    
def save_checkpoint(state, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save,'model_best.pth.tar')
    torch.save(state, filename)
    



class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes=17, epsilon=0.1):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss



def read_h5_data(data_path):
    data = h5py.File(data_path,"r")
    h5_data = data['matrix/data']
    sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).transpose())
    data_fs = torch.from_numpy(np.array(sparse_data.todense()))
    data_fs = Variable(data_fs.type(FloatTensor))
    return data_fs



    
def read_fs_label(label_path):
    label_fs = pd.read_csv(label_path,header=None,index_col=False)  #
    label_fs = pd.Categorical(label_fs.iloc[1:(label_fs.shape[0]),1]).codes
    label_fs = np.array(label_fs[:]).astype('int32')
    label_fs = torch.from_numpy(label_fs)#
    label_fs = label_fs.type(LongTensor)
    return label_fs

def get_encodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    encodings = []
    ori_data = []
    label = [] 
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
                x = batch_sample['data']
                x = Variable(x.type(FloatTensor))
                x = torch.reshape(x,(x.size(0),-1))
                rna_valid_label = batch_sample['label']
                rna_valid_label = Variable(rna_valid_label.type(LongTensor))
            
                x_prime = model.encoder(x.to(device))
                encodings.append(x_prime)
                ori_data.append(x)
                label.append(rna_valid_label)
                
                
    return torch.cat(encodings, dim=0),torch.cat(ori_data,dim=0),torch.cat(label,dim=0)

def get_decodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
                x = batch_sample['data']
                x = Variable(x.type(FloatTensor))
                x = torch.reshape(x,(x.size(0),-1))
                rna_valid_label = batch_sample['label']
                rna_valid_label = Variable(rna_valid_label.type(LongTensor))
                x_prime, x_cty,mu,var = model(x.to(device))
                decodings.append(x_prime)
                ori_data.append(x)                
    return torch.cat(decodings, dim=0),torch.cat(ori_data,dim=0)

def get_simulated_data_random_generation(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []
    label = []
    for i, batch_sample in enumerate(dl):
            x = batch_sample['data']
            x = Variable(x.type(FloatTensor))
            x = torch.reshape(x,(x.size(0),-1))
            rna_valid_label = batch_sample['label']
            rna_valid_label = Variable(rna_valid_label.type(LongTensor))
            
            x_prime, x_cty,mu,var = model(x.to(device))
            decodings.append(x_prime)
            ori_data.append(x)
            label.append(rna_valid_label)
                
    return torch.cat(decodings, dim=0),torch.cat(label,dim=0)


    
def get_simulated_data(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    decodings = []
    ori_data = []
    label = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
                x = batch_sample['data']
                x = Variable(x.type(FloatTensor))
                x = torch.reshape(x,(x.size(0),-1))
                rna_valid_label = batch_sample['label']
                rna_valid_label = Variable(rna_valid_label.type(LongTensor))
            
                x_prime, x_cty,mu,var = model(x.to(device))
                decodings.append(x_prime)
                ori_data.append(x)
                label.append(rna_valid_label)
                
    return torch.cat(decodings, dim=0),torch.cat(label,dim=0),torch.cat(ori_data,dim=0)
    

def KL_loss(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    return  KLD

def compute_zscore(data):
    temp_mean = torch.mean(data,1)
    temp_mean = temp_mean.repeat(data.size(1),1).transpose(0,1)
    temp_std = torch.std(data,1)
    temp_std = temp_std.repeat(data.size(1),1).transpose(0,1)
    data = (data-temp_mean)/temp_std
    return  data
    
def compute_log2(data):
    temp_colsum = torch.sum(data,1)
    temp_mean_colsum = torch.mean(temp_colsum)
    temp = temp_colsum/temp_mean_colsum
    temp = temp.repeat(data.size(1),1).transpose(0,1)
    data = torch.log2(data/temp+1)
    return  data


def get_vae_simulated_data_from_sampling(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    latent = []
    ori_data = []
    label = []
    decodings = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
                x = batch_sample['data']
                x = Variable(x.type(FloatTensor))
                x = torch.reshape(x,(x.size(0),-1))
                
                ori_data.append(x)
                rna_valid_label = batch_sample['label']
                rna_valid_label = Variable(rna_valid_label.type(LongTensor))
                y, x_cty,mu,var = model(x.to(device))
                decodings.append(y)
                label.append(rna_valid_label)
                
    return torch.cat(decodings, dim=0),torch.cat(label,dim=0),torch.cat(ori_data,dim=0)


