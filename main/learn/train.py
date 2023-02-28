import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
import os
import sys
import shutil
from util import AverageMeter,accuracy,save_checkpoint,CrossEntropyLabelSmooth,KL_loss

def train_model(model, train_dl, test_dl, lr, epochs, classify_dim=17, best_top1_acc=0, save_path = "", feature_num=10000):
    #####set optimizer and criterin#####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ##
    criterion = nn.MSELoss().cuda()
    criterion_smooth_cty = CrossEntropyLabelSmooth().cuda()
    
    best_top1_acc=0
    best_each_celltype_top1 = []
    best_each_celltype_num=[]
    train_each_celltype_num = []

    for i in range(classify_dim):
        best_each_celltype_top1.append(0)
        best_each_celltype_num.append(0)
        train_each_celltype_num.append(0)
                
    ######loop training process, each epoch contains train and test two part#########
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        nsamples_train = 0
        train_top1 = AverageMeter('Acc@1', ':6.2f')
        model = model.train()
        nsamples_test = 0
        test_top1 = AverageMeter('Acc@1', ':6.2f')
        each_celltype_top1 = []
        each_celltype_num=[]
        for i in range(classify_dim):
            each_celltype_top1.append(AverageMeter('Acc@1', ':6.2f'))
            each_celltype_num.append(0)

            
        for i, batch_sample in enumerate(train_dl):
            optimizer.zero_grad()
            ###load data
            x = batch_sample['data']
            x = Variable(x)
            x = torch.reshape(x,(x.size(0),-1))
            train_label = batch_sample['label']
            train_label = Variable(train_label)
            # Forward pass
            x_prime, x_cty,mu, var = model(x.to(device))
            # loss function
            loss1 = criterion(x_prime, x.to(device)) + 1/feature_num*(KL_loss(mu,var))#simulation loss
            loss2 = criterion_smooth_cty(x_cty, train_label.to(device))  #classification loss
            loss = 0.9*loss1 + 0.1*loss2 ##sum up the loss together
            # Backward pass
            loss.backward()
            optimizer.step()
            # log losses
            batch_size = x.shape[0]
            nsamples_train += batch_size
            train_pred1,  = accuracy(x_cty, train_label, topk=(1, ))
            train_top1.update(train_pred1[0], 1)
            if epoch == 1:
                for j in range(classify_dim):
                    if len(train_label[train_label==j])!=0:
                        train_each_celltype_num[j]=train_each_celltype_num[j] + len(train_label[train_label==j])                        


        model = model.eval()
        if test_dl!="NULL":
            with torch.no_grad():
                for i, batch_sample in enumerate(test_dl):
                    ###load data
                    x = batch_sample['data']
                    x = Variable(x)
                    x = torch.reshape(x,(x.size(0),-1))
                    test_label = batch_sample['label']    
                    test_label = Variable(test_label)
                    ###forward process
                    x_prime, x_cty,mu, var = model(x.to(device))

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
        
            ####save the best model
        #if test_top1.avg > best_top1_acc:
        #    best_top1_acc = test_top1.avg
        if epoch==epochs:
            #for j in range(classify_dim):
            #    best_each_celltype_top1[j] = each_celltype_top1[j].avg
            #    best_each_celltype_num[j] = each_celltype_num[j]
            save_checkpoint({'epoch': epoch,
                'state_dict': model.state_dict(),
                #'best_top1_acc': best_top1_acc,
                #'best_top1_celltype_acc': best_each_celltype_top1,
                #'best_top1_celltype_num': best_each_celltype_num,
                'optimizer' : optimizer.state_dict(),
                }, save_path)

        
        #if epoch==epochs:
        #    print('Epoch : ',epoch, '\t')
        #    for j in range(classify_dim):
        #        print('cell type : ',j, '\t', '\t', 'prec :', best_each_celltype_top1[j], 'number:', best_each_celltype_num[j], 'train_cty_num:',train_each_celltype_num[j])
    
    return model,best_each_celltype_top1,best_each_celltype_num,train_each_celltype_num

