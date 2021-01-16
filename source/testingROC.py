# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:29:59 2021

@author: ÄGARE
"""

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores):
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
    plt.plot([1-eer], [eer], marker='o', markersize=5, color="navy")
    plt.plot([0,1],[0,1],color='navy', linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    saveto = os.getcwd()
    plt.savefig(os.path.join(saveto, "ROC.png"))
    plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap