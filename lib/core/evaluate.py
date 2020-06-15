# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def compute_metrics(pred, target):
    """Compute precision and recall for binary classification problem
    pred: 1d numpy boolean array
    target: 1d numpy boolean array
    """
    tp = np.intersect1d(np.where(pred==True), np.where(target == True)).size
    tn = np.intersect1d(np.where(pred==False), np.where(target == False)).size
    fn = np.intersect1d(np.where(pred==False), np.where(target == True)).size
    fp = np.intersect1d(np.where(pred==True), np.where(target == False)).size
    
    acc = (tp+tn)/(tp+tn+fn+fp)

    if tp > 0:
        prec = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * (prec * recall) / (prec + recall)
    else:
        prec = 0
        recall = 0
        f1 = 0
    msg = 'Accuracy {:.2%}\t'.format(acc)
    msg += 'Precision {:.2%} Recall {:.2%} F1 {:.2%}\n'.format(prec, recall, f1)
    msg += 'TP {} TN {} FP {} FN {}\n'.format(tp, tn, fp, fn)
    msg += '\n'
    return acc, prec, recall, f1, msg

def metrics_notvisible(pred, target):
    """Compute accuracy, precision and recall to detect visible/not visible landmarks
    True - landmark is not visible/ Fale - visible
    pred - numpy float array of shape (bs, n_ldm) of max values in heatmaps 
            where 0 - no signal (not visible), >0 - signal on heatmap
    target - boolean array of shape (bw, n_ldm), True - not visible, False - visible
    """
    message = ''
    message += 'Analysing landmark visibility prediction scores over different thresholds\n'
    num_ldm = target.size
    num_not_visible = np.sum(target)
    message += 'Not visible landmark {} out of {}, {:.2%}\n'.format(num_not_visible, num_ldm, num_not_visible/num_ldm)
    
    vis_thr = np.linspace(0, 1, 11)
    for thr in vis_thr:
        pred_vis = pred <= thr
        message += 'Threshold {:.2f}\n'.format(thr)
        message +=  compute_metrics(pred_vis.ravel(), target.ravel())[-1]
        
    #Evaluate with sklearn
    fpr, tpr, _ = metrics.roc_curve((~target).astype(int).ravel(), pred.ravel())
    roc_auc = metrics.auc(fpr, tpr)
    message += 'ROC AUC {:.2f}\n'.format(roc_auc)

    return message
