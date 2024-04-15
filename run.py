#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import torch
from dataset import *
from model import *
from model1 import *
from loss import *
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from functools import partial
from torchmetrics import SpearmanCorrCoef
from torchmetrics.classification import Accuracy, BinaryPrecision,F1Score,Precision, Recall


torch.manual_seed(47)  # 为 CPU 设置随机种子
torch.cuda.manual_seed(47)  # 为当前 GPU 设置随机种子
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['CUBLAS_WORKSPACE_CONFIG']=":16:8"



import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = True


seed=43
data_dir="/mnt/disk2/wyd/esmgnn/data_process/data"
train_data_version="D1traug1;D1traug1"
#train_data_version="tr;tr_avg"
test_data_version="D1ts;D1ts"
save_path='/mnt/disk2/wyd/esmgnn/model_zoo/'

input_size = 69
feat_dim=13
max_epochs=60
## parameters for tuning ########
batch_size=128
dropout=0.5
#stochastic_weight_avg="1"
lr=1e-3
hidden_dim=512
n_layers=1
weight_decay = 1e-4
lr_scheduler="step"
n_fc_layers=3
num_classes=2

max_epochs1=60

label_name="binary"
loss="ce"


# In[22]:


setup_seed(seed)


# In[6]:


train_data = StandardData(data_dir,
            data_version=train_data_version,
            mode='train', x_field = 'SMILES', label_name = label_name, train=True, ratio = None, seed=seed)
train_dataloader = DataLoader(train_data,shuffle=True, batch_size = batch_size)


val_data = StandardData(data_dir,
            data_version=train_data_version,
            mode='train', x_field = 'SMILES', label_name = label_name, train=False, ratio = None, seed=seed)
val_dataloader = DataLoader(val_data,shuffle=False, batch_size = batch_size)



if loss == 'mse':
    loss_function = F.mse_loss
elif loss == 'l1':
    loss_function = F.l1_loss
elif loss == 'bce':
    loss_function = F.binary_cross_entropy_with_logits
elif loss == 'ce':
    reweight = True
    if reweight:
        #1243, 2670
        pos_num = 215
        total = 430
        s = np.array([total-pos_num,pos_num, ]) # 2:1 的正负样本比
        s = np.sum(s)/ s
        loss_function = partial(F.cross_entropy, weight=torch.tensor(s, dtype=torch.float).cuda()) #F.cross_entropy#
    else:
        loss_function = F.cross_entropy
elif loss  == 'relative':
    loss_function = rel_loss
elif loss == 'rank':
    loss_function = rank_loss
elif loss == 'multitask':
    loss_function = multi_rank_loss
elif loss == "rank_10fold":
    loss_function = rank_loss_10fold
else:
    raise ValueError("Invalid Loss Type!")




loss, loss_function


model0 = StandardNet(input_size = input_size, hidden_dim = hidden_dim, 
                    n_layers = n_layers, n_fc_layers = n_fc_layers,
                    dropout = dropout,
                   feat_dim = feat_dim,
                   num_classes=2)

model1 = GNNnet(input_size = input_size, hidden_dim = hidden_dim, 
                    n_layers = n_layers, n_fc_layers = n_fc_layers,
                    dropout = dropout,
                   feat_dim = feat_dim,
                   num_classes=2)


def evaluate(model, test, metrics, use_cuda = True):
    model.eval()
    if use_cuda:
        metrics = metrics.cuda()
        model.cuda()
    loss_val = []
    label_val = []
    mcc_val = []
    metrics.reset()
    for batch in test:
        labels = batch.y
        #labels = labels.to(torch.float32)
        if use_cuda:
            batch = batch.cuda()
            labels = labels.cuda()
        pred = model(batch)

        loss = loss_function(pred,labels)
        loss_val.append(loss.item())
        metrics(pred, labels)
        
        predy = F.softmax(pred,dim=-1)
        mcc_val.extend(pred.detach().cpu().numpy())
        label_val.extend(labels.detach().cpu().numpy())
    
    mcc_val = np.array(mcc_val)[:,1]
    label_val = np.array(label_val)
    return np.mean(loss_val), metrics.compute().mean(), matthews_corrcoef(mcc_val>0.5, label_val), accuracy_score(mcc_val>0.5, label_val)

def train(model, optim, train_dataloader, val_dataloader, loss_fn, metrics, max_epochs, save_path, typem, use_cuda=True):
    model.train()
    if use_cuda:
        model.cuda()
        metrics.cuda()
    
    best_r = 0
    for e in range(max_epochs):
        train_loss = []
        metrics.reset()
        for batch in train_dataloader:
            model.zero_grad()
            optim.zero_grad()
            labels = batch.y
            #labels = labels.to(torch.float32)
            if use_cuda:
                batch = batch.cuda()
                labels = labels.cuda()
            pred = model(batch)
            #print(len(pred),len(labels))
            loss = loss_function(pred,labels)
            train_loss.append(loss.item()) 
            metrics(pred, labels)
            
            loss.backward()
            optim.step()
        f1_train = metrics.compute().mean()
        loss_train = np.mean(train_loss)
        
        val_loss, val_f1, val_mcc, val_acc = evaluate(model, val_dataloader, metrics, use_cuda)
        print(f'epoch {e}: loss_train: {loss_train:.3f} f1_train: {f1_train:.3f} loss_val: {val_loss:.3f} f1_val: {val_f1:.3f} acc_val: {val_acc:.3f} mcc_val: {val_mcc:.3f}')
        
        if best_r <= val_mcc:
            best_r = val_mcc
            print('best_epoch:', e)
            if typem == 0 :
                torch.save({"state_dict": model.state_dict()}, os.path.join(save_path, "model0.ckpt-best.ptb".format(e)))
            if typem == 1 :
                torch.save({"state_dict": model.state_dict()}, os.path.join(save_path, "model1.ckpt-best.ptb".format(e)))
            

optimizer0 = torch.optim.Adam(
            model0.parameters(), lr=lr, weight_decay=weight_decay)
optimizer1 = torch.optim.Adam(
            model1.parameters(), lr=lr, weight_decay=weight_decay)
if 'ce' not in loss:
    spearmanr  = SpearmanCorrCoef(num_outputs = num_classes)#F1Score('multiclass', num_classes=num_classes)
else:
    spearmanr  = F1Score('multiclass', num_classes=num_classes)



save_path='model_zoo/'


train(model0, optimizer0, train_dataloader, val_dataloader, loss_function, spearmanr, save_path = save_path, max_epochs=max_epochs, typem=0)
train(model1, optimizer1, train_dataloader, val_dataloader, loss_function, spearmanr, save_path = save_path, max_epochs=max_epochs1, typem=1)


model0.load_state_dict(torch.load(os.path.join(save_path, "model0.ckpt-best.ptb"))["state_dict"])
model1.load_state_dict(torch.load(os.path.join(save_path, "model1.ckpt-best.ptb"))["state_dict"])



test_data = StandardData(data_dir,
            data_version=test_data_version,
            mode='dev', x_field = 'smiles', label_name = 'binary', train=True, ratio = None, seed=seed)
test_dataloader = DataLoader(test_data,shuffle=False, batch_size = batch_size)


preds0 = []
preds1 = []
for batch in test_data:
    labels = batch.y
    batch.cuda()
    
    pred = F.softmax(model0(batch),dim=-1)
    gnn = F.softmax(model1(batch),dim=-1)
    preds0.extend(pred.detach().cpu().numpy())
    preds1.extend(gnn.detach().cpu().numpy())



preds0 = np.array(preds0)[:,1]
preds1 = np.array(preds1)[:,1]
predy = (preds0+preds1)/2
label = test_data.y

df = pd.DataFrame({
    'esm':preds0,
    'pesm':preds0>0.5,
    'gnn':preds1,
    'pgnn':preds1>0.5,
    'esmgnn':predy,
    'predy':predy>0.5,
    'label':label
})
df.to_csv('results/D1.csv',index=False)

print(accuracy_score(label, preds0>0.5), f1_score(label, preds0>0.5), matthews_corrcoef(label, preds0>0.5))
print(accuracy_score(label, preds1>0.5), f1_score(label, preds1>0.5), matthews_corrcoef(label, preds1>0.5))
print(accuracy_score(label, predy>0.5), f1_score(label, predy>0.5), matthews_corrcoef(label, predy>0.5))





