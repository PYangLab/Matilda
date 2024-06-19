import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
import os
import sys
import shutil
from util import AverageMeter,accuracy

def test_model(model, dl, real_label, classify_dim=17, save_path = ""):
    #####set optimizer and criterin#####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsamples_test = 0
    test_top1 = AverageMeter('Acc@1', ':6.2f')
    each_celltype_top1 = []
    each_celltype_num=[]
    best_each_celltype_top1 = []
    for i in range(classify_dim):
        each_celltype_top1.append(AverageMeter('Acc@1', ':6.2f'))
        each_celltype_num.append(0)
        best_each_celltype_top1.append(0)

    model.eval()
    classified_label = []
    groundtruth_label = []
    prob = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            ###load data
            x = batch_sample['data']
            x = Variable(x)
            x = torch.reshape(x,(x.size(0),-1))
            test_label = batch_sample['label']    
            test_label = Variable(test_label)
            ###forward process
            x_prime, x_cty,mu, var = model(x.to(device))
            a = torch.max(nn.Softmax()(x_cty),1)
            
            for j in range(x_prime.size(0)):
                classified_label.append(real_label[a.indices[j]])

                if (torch.max(test_label)!=0):
                    groundtruth_label.append(real_label[test_label[j]])
                else:
                    groundtruth_label.append(-1)
                prob.append(a.values[j])
               

            batch_size = x.shape[0]
            nsamples_test += batch_size
            test_pred1,  = accuracy(x_cty, test_label, topk=(1, ))
            test_top1.update(test_pred1[0], 1)

            ###record accuracy for each celltype
            for j in range(classify_dim):
                if len(test_label[test_label==j])!=0:
                    pred1,  = accuracy(x_cty[test_label==j,:], test_label[test_label==j], topk=(1, ))
                    each_celltype_top1[j].update(pred1[0],1)
                    each_celltype_num[j]=each_celltype_num[j] + len(test_label[test_label==j])

    for j in range(classify_dim):
        best_each_celltype_top1[j] = each_celltype_top1[j].avg
        print('cell type ID: ',j, '\t', '\t', 'cell type:', real_label[j], '\t', '\t', 'prec :', each_celltype_top1[j].avg, 'number:', each_celltype_num[j], file = save_path)
        
    return model,best_each_celltype_top1,each_celltype_num, classified_label, groundtruth_label,prob
