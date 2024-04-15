import torch
import torch.nn.functional as F 
import numpy as np
def relative_loss(predicted_values, true_values, threhold = 5):
    errors = predicted_values - true_values
    error_percentages = (errors / true_values) * 100

    in_range_mask = torch.logical_and(error_percentages >= -1*threhold, error_percentages <= threhold)
    loss = torch.where(in_range_mask, torch.tensor(0.0,device='cuda'), torch.abs(errors))
    #print(predicted_values,true_values, loss)
    return loss.mean()

def pair_residue(values):
    values = values.reshape(-1,1)
    b,_ = values.size()

    values = values.expand(b,-1)
    return values - values.t()

th = np.log(3)
def pair_10fold(values):
    values = values.reshape(-1,1)
    b,_ = values.size()
    values = values.expand(b,-1)
    return (values - values.t()) - th

def rel_loss(pred, target, clip_eps=-0.5):
    assert pred.shape==target.shape
    dis_target = target[None]-target[:, None]
    dis_pred=pred[None]-pred[:, None]
    loss=dis_pred[dis_target<0].clamp(clip_eps).mean()
    return loss

def rank_loss(predicted_values, true_values, delta = 1.0):
    predicted_values = pair_residue(predicted_values)
    true_values = torch.sigmoid(pair_residue(true_values))
    return F.binary_cross_entropy_with_logits(predicted_values, true_values)

def rank_loss_10fold_base(predicted_values, true_values, delta = 1.0):
    predicted_values = pair_residue(predicted_values)
    true_values = torch.sigmoid(pair_10fold(true_values))
    return F.binary_cross_entropy_with_logits(predicted_values, true_values)

from functools import partial
pos_num = 1849-1041#1243
total = 1849
s = np.array([pos_num, total-pos_num, ]) # 2:1 的正负样本比
s = np.sum(s)/ s
loss_function = partial(F.cross_entropy, weight=torch.tensor(s, dtype=torch.float).cuda()) #F.cross_entropy#
def rank_loss_10fold(predicted_values, true_values):
    #true_values[:,2] = true_values[:,2]<np.log(100)
    return rank_loss_10fold_base(predicted_values[:,0], true_values[:,0]) + \
    rank_loss_10fold_base(predicted_values[:,1], true_values[:,1])
    #+ \
    #rank_loss_10fold_base(predicted_values[:,2], true_values[:,2])
    
def multi_rank_loss(predicted_values, true_values):
    return rank_loss(predicted_values[:,0], true_values[:,0]) + rank_loss(predicted_values[:,1], true_values[:,1])