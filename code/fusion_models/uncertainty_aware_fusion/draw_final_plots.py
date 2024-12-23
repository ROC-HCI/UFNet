import os
import copy
import pickle
import re
import math
import json
import wandb
import random
import click
import imblearn
import scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import subprocess as sp

import baal.bayesian.dropout as mcdropout
from baal.modelwrapper import ModelWrapper
import plotly.graph_objects as go

from pandas import DataFrame
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

import matplotlib as mpl

# Load seeds from the wandb experiments
df_seeds = pd.read_csv("../../performance_analysis/wandb_reports/final_fusion_drop_preds_May132024.csv")
seeds = list(df_seeds["seed"])

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def calibration_curve(y_pred_scores, y, num_buckets=10):
    y_pred_scores = np.asarray(y_pred_scores).flatten()
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, num_buckets + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_mids = (bin_lowers+bin_uppers)/2

    # get predictions from confidences (positional in this case)
    predicted_label = (y_pred_scores>=0.5)

    ece = np.zeros(1)
    bin_true_probs = []
    bin_pred_probs = []
    for bin_lower, bin_upper, bin_mid in zip(bin_lowers, bin_uppers, bin_mids):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(y_pred_scores > bin_lower.item(), y_pred_scores <= bin_upper.item())
        
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            true_prob_in_bin = y[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_pred_prob_in_bin = bin_mid.item()

            bin_pred_probs.append(avg_pred_prob_in_bin)
            bin_true_probs.append(true_prob_in_bin)

    return bin_pred_probs, bin_true_probs

'''
Evaluate performance on validation/test set.
Returns all the metrics defined above and the loss.
'''
def expected_calibration_error(y_pred_scores, y, num_buckets=20):
    y_pred_scores = np.asarray(y_pred_scores).flatten()
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, num_buckets + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.maximum(y_pred_scores, 1.0-y_pred_scores)

    # get predictions from confidences (positional in this case)
    predicted_label = (y_pred_scores>=0.5)
    
    # get a boolean list of correct/false predictions
    accuracies = (predicted_label==y)

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

'''
Given labels and prediction scores, make a comprehensive evaluation. 
i.e., threshold = 0.5 means prediction>0.5 will be considered as positive
'''
def compute_metrics(y_true, y_pred_scores, threshold = 0.5):
    labels = np.asarray(y_true).reshape(-1)
    pred_scores = np.asarray(y_pred_scores).reshape(-1)
    preds = (pred_scores >= threshold)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['average_precision'] = average_precision_score(labels, pred_scores)
    metrics['auroc'] = roc_auc_score(labels, pred_scores)
    metrics['f1_score'] = f1_score(labels, preds)
    metrics['fpr'], metrics['tpr'], metrics['thresholds'] = roc_curve(labels, pred_scores)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["confusion_matrix"] = {"tn":tn, "fp":fp, "fn":fn, "tp":tp}
    metrics["weighted_accuracy"] = (safe_divide(tp, tp + fp) + safe_divide(tn, tn + fn)) / 2.0

    '''
    True positive rate or recall or sensitivity: probability of identifying a positive case 
    (often called the power of a test)
    '''
    metrics['TPR'] = metrics['recall'] = metrics['sensitivity'] = recall_score(labels, preds)
    
    '''
    False positive rate: probability of falsely identifying someone as positive, who is actually negative
    '''
    metrics['FPR'] = safe_divide(fp, fp+tn)
    
    '''
    Positive Predictive Value: probability that a patient with a positive test result 
    actually has the disease
    '''
    metrics['PPV'] = metrics['precision'] = precision_score(labels, preds, zero_division=0.0)
    
    '''
    Negative predictive value: probability that a patient with a negative test result 
    actually does not have the disease
    '''
    metrics['NPV'] = safe_divide(tn, tn+fn)
    
    '''
    True negative rate or specificity: probability of a negative test result, 
    conditioned on the individual truly being negative
    '''
    metrics['TNR'] = metrics['specificity'] = safe_divide(tn,(tn+fp))

    metrics['Brier Score'] = brier_score_loss(labels, pred_scores)

    metrics['ECE'] = expected_calibration_error(pred_scores, labels)

    metrics['pred_probs'], metrics['true_probs'] = calibration_curve(pred_scores, labels)
    
    return metrics

def main(**cfg):
    auroc_curve_metrics = ['auroc', 'fpr', 'tpr', 'thresholds']
    all_results = {m:[] for m in auroc_curve_metrics}
    metrics = ['Brier Score', 'ECE', 'FPR', 'recall', 'specificity', 'average_precision', 'precision', 'NPV', 'auroc', 'f1_score', 'accuracy', 'weighted_accuracy']
    performance_report = {m:[] for m in metrics}

    for seed in seeds:
        print(f"Running experiment for seed: {seed}")
        with open(f'outputs/preds_with_ids_{seed}.pkl', 'rb') as f:
            loaded_results = pickle.load(f)

        test_metrics = compute_metrics(loaded_results['all_labels'], loaded_results['all_preds'])
        
        for m in auroc_curve_metrics:
            all_results[m].append(test_metrics[m])

        for m in metrics:
            performance_report[m].append(test_metrics[m])

    for m in metrics:
        all_runs = np.asarray(performance_report[m])
        mean = all_runs.mean()
        std = all_runs.std()
        error = 1.96*(std/math.sqrt(len(all_runs)))
        print(f"{m}: {mean:.4f} [{(mean-error):.4f}, {(mean+error):.4f}]")
    
    fpr_mean    = np.linspace(0, 1, 100)

    interp_tprs = []
    for i in range(len(seeds)):
        fpr = all_results['fpr'][i]
        tpr = all_results['tpr'][i]
        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2*np.std(interp_tprs, axis=0)
    tpr_upper = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower = tpr_mean-tpr_std
    auc = np.mean(all_results['auroc'])

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(fpr_mean,tpr_mean, color='green')  
    ax.plot([0, 1], [0, 1], 'b--', label='Random')
    ax.fill_between(fpr_mean, tpr_lower, tpr_upper, color='b', alpha=.1)
    # textstr = '\n'.join((
    # '',
    # f'AUROC = {auc:.3f}'))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    #plt.text(0.40, 0.05, f"AUROC = {auc:.3f}", weight="bold")
    mpl.rc('text', usetex=True)
    textstr = 'AUROC: $0.93 \pm 0.002$'    
    plt.text(0.30, 0.05, textstr, fontsize=18, weight="bold")
    # plt.text(0.40, 0.05, textstr, bbox=props)

    # col_labels=['PD','Non-PD']
    # row_labels=['PD','Non-PD']
    # table_vals=[64,12,10,139]
    # mpl.rc('text', usetex=True)
    # table = r'''\begin{tabular}{|l|l|l|} \hline & PD & Non-PD \\ \hline PD     & 64 & 12     \\ \hline Non-PD & 10 & 139    \\ \hline \end{tabular}'''
    # plt.text(0.60,0.50,table,size=12)
    # plt.text(0.75,0.62,"True Classes",weight="bold")
    # plt.text(0.57,0.42,"Predictions",weight="bold",rotation=90)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.legend(loc = 'upper left')
    plt.savefig("hybrid_final_model_auroc.png", bbox_inches='tight', dpi=300)

    '''
    Calibration curves
    '''
    cal_curve_metrics = ['ECE', 'Brier Score', 'pred_probs', 'true_probs']
    all_results = {m:[] for m in cal_curve_metrics}

    for seed in seeds:
        print(f"Running experiment for seed: {seed}")
        with open(f'outputs/preds_with_ids_{seed}.pkl', 'rb') as f:
            loaded_results = pickle.load(f)

        test_metrics = compute_metrics(loaded_results['all_labels'], loaded_results['all_preds'])
        
        for m in cal_curve_metrics:
            all_results[m].append(test_metrics[m])
    
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    pred_probs_mean = (bin_lowers+bin_uppers)/2
    true_probs_all = [[] for i in range(10)]
    true_probs_mean = np.zeros((len(pred_probs_mean),))
    true_probs_std = np.zeros((len(pred_probs_mean),))
    
    for i in range(len(seeds)):
        true_probs = all_results['true_probs'][i]
        pred_probs = all_results['pred_probs'][i]
        for tp, pp in zip(true_probs, pred_probs):
            idx = (int)(np.floor(pp/0.10))
            true_probs_all[idx].append(tp)

    for i in range(len(true_probs_all)):
        true_probs_mean[i] = np.asarray(true_probs_all[i]).mean()
        true_probs_std[i] = 1.96*(np.asarray(true_probs_all[i]).std()/math.sqrt(len(true_probs_all[i])))

    true_probs_upper = np.clip(true_probs_mean+true_probs_std, 0, 1)
    true_probs_lower = np.clip(true_probs_mean-true_probs_std, 0, 1)
    ece = f"{np.mean(all_results['ECE']):.4f} [{((1.96*np.std(all_results['ECE']))/math.sqrt(len(seeds))):.4f}]"
    bs = f"{np.mean(all_results['Brier Score']):.4f} [{((1.96*np.std(all_results['Brier Score']))/math.sqrt(len(seeds))):.4f}]"
    print(ece)
    print(bs)

    plt.clf()
    plt.plot([0, 1], [0, 1], 'b--', label = 'Ideally calibrated')
    plt.errorbar(pred_probs_mean, true_probs_mean, yerr=true_probs_std, linestyle = '-', color='green', ecolor='orange', marker = 's')
    
    mpl.rc('text', usetex=True)
    textstr = 'Brier Score: $0.097 \pm 0.002$\nECE: $0.054 \pm 0.005$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.40,0.05,textstr, size=16, bbox=props)
    
    # textstr = '\n'.join((
    #     f'Brier score: 0.0942 [0.0919, 0.0966]',
    #     f'Expected Calibration Error (ECE): 0.0535 [0.0497, 0.0573]'
    #     ))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.text(0.40, 0.05, textstr, bbox=props)

    # textstr = '\n'.join((
    #     'Hybrid fusion model',
    #     'Unimodal models are trained with MC Dropout'
    # ))
    # props = dict(boxstyle='round', alpha=0.1)
    # plt.text(0.01, 0.90, textstr, weight="bold",bbox=props)

    plt.xlabel('Predicted probability', fontsize=18)
    plt.ylabel('True probabilities in each bin', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("final_model_calibration.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()