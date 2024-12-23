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

thresholds = np.arange(0.00,1.01,0.01)

accuracies = []
accuracies_low = []
accuracies_high = []

sensitivities = []
sensitivities_low = []
sensitivities_high = []

specificities = []
specificities_low = []
specificities_high = []

for threshold in thresholds:
    accs = []
    senss = []
    specs = []
    for seed in seeds:
        filename = f'../fusion_models/uncertainty_aware_fusion/outputs/preds_with_ids_{seed}_no_drop.pkl'
        with open(filename, 'rb') as f:
            loaded_results = pickle.load(f)
            ids, labels, pred_scores = loaded_results['all_ids'], loaded_results['all_labels'], loaded_results['all_preds']

            accurate_preds = (pred_scores>=threshold)==labels
            acc = 1.0*(accurate_preds).mean()
            accs.append(acc)

            sens = accurate_preds[labels==1.0].sum()/len(accurate_preds[labels==1.0])
            senss.append(sens)

            spec = accurate_preds[labels==0.0].sum()/len(accurate_preds[labels==0.0])
            specs.append(spec)
                

    mean_acc = np.asarray(accs).mean()
    std_acc = np.asarray(accs).std()
    std_error = standard_error(len(seeds), std_acc)
    accuracies.append(mean_acc)
    accuracies_low.append(mean_acc-std_error)
    accuracies_high.append(mean_acc+std_error)

    mean_sens = np.asarray(senss).mean()
    std_sens = np.asarray(senss).std()
    std_error = standard_error(len(seeds), std_sens)
    sensitivities.append(mean_sens)
    sensitivities_low.append(mean_sens-std_error)
    sensitivities_high.append(mean_sens+std_error)

    mean_spec = np.asarray(specs).mean()
    std_spec = np.asarray(specs).std()
    std_error = standard_error(len(seeds), std_spec)
    specificities.append(mean_spec)
    specificities_low.append(mean_spec-std_error)
    specificities_high.append(mean_spec+std_error)
    
plt.plot(thresholds, accuracies, color="blue", label="Accuracy")
plt.fill_between(thresholds, accuracies_low, accuracies_high, color='blue', alpha=.2)
plt.plot(thresholds, sensitivities, color="green", label="Sensitivity")
plt.fill_between(thresholds, sensitivities_low, sensitivities_high, color='green', alpha=.2)
plt.plot(thresholds, specificities, color="red", label="Specificity")
plt.fill_between(thresholds, specificities_low, specificities_high, color='red', alpha=.2)

plt.xlabel("Decision threshold", fontsize=18)
plt.ylabel("Performance metric", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.savefig("specificity_vs_sensitivity.jpg",bbox_inches='tight',dpi=300)