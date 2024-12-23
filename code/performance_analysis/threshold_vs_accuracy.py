import os
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy import stats
import pickle

def standard_error(n, std, test_type='z'):
    if n==0:
        return 0
    if test_type=="t":
        z_critical = scipy.stats.t.ppf(q=0.975, df = n-1)
    else:
        z_critical = 1.96
    return z_critical*(std/math.sqrt(n))

# Load seeds from the wandb experiments
df_seeds = pd.read_csv("wandb_reports/final_fusion_drop_preds_May132024.csv")
seeds = list(df_seeds["seed"])

thresholds = np.arange(0.50,1.01,0.01)
accuracies = []
accuracies_low = []
accuracies_high = []

coverages = []
coverages_low = []
coverages_high = []

for threshold in thresholds:
    accs = []
    covs = []
    for seed in seeds:
        filename = f'../fusion_models/uncertainty_aware_fusion/outputs/preds_with_ids_{seed}_no_drop.pkl'
        with open(filename, 'rb') as f:
            loaded_results = pickle.load(f)
            ids, labels, pred_scores = loaded_results['all_ids'], loaded_results['all_labels'], loaded_results['all_preds']
            confidence = np.maximum(pred_scores, 1.0-pred_scores)
            pred_scores = pred_scores[confidence>=threshold]
            labels = labels[confidence>=threshold]
            try:
                acc = 1.0*((pred_scores>=0.5)==labels).mean()
                accs.append(acc)
                cov = np.sum(confidence>=threshold)/len(confidence)
                covs.append(cov)
            except:
                print(threshold)
                accs.append(1.0)
                covs.append(0.0)

    mean_acc = np.asarray(accs).mean()
    std_acc = np.asarray(accs).std()
    std_error = standard_error(len(seeds), std_acc)
    accuracies.append(mean_acc)
    accuracies_low.append(mean_acc-std_error)
    accuracies_high.append(mean_acc+std_error)

    mean_cov = np.asarray(covs).mean()
    std_cov = np.asarray(covs).std()
    std_error = standard_error(len(seeds), std_cov)
    coverages.append(mean_cov)
    coverages_low.append(mean_cov-std_error)
    coverages_high.append(mean_cov+std_error)
    
plt.plot(thresholds, accuracies, color='blue', label="Accuracy")
plt.fill_between(thresholds, accuracies_low, accuracies_high, color='blue', alpha=.2)
plt.plot(thresholds, coverages, color='green', label="Coverage")
plt.fill_between(thresholds, coverages_low, coverages_high, color='green', alpha=.2)
plt.xlabel("Model confidence", fontsize=18)
plt.ylabel("Performance metric", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.savefig("accuracy_coverage.jpg",bbox_inches='tight',dpi=300)