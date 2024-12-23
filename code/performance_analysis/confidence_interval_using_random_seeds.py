import pandas as pd 
import math

runs_df = pd.read_csv("wandb_reports/final_model_stats_speech_smile.csv")

logs = []
seeds = []
# metrics = ['BS', 'ECE', 'FPR', 'recall', 'specificity', 'average_precision', 'precision', 'NPV', 'auroc', 'f1_score', 'accuracy', 'weighted_accuracy']
metrics = ['coverage', 'BS', 'ECE', 'FPR', 'recall', 'specificity', 'average_precision', 'precision', 'NPV', 'auroc', 'f1_score', 'accuracy', 'weighted_accuracy']

for i,r in runs_df.iterrows():
    best_metrics = {x:r[x] for x in metrics}
    logs.append(best_metrics)
    seeds.append(r['seed'])

metrics_df = pd.DataFrame.from_dict(logs)

mean_metrics = {}
ci_metrics = {}
metrics_str = {}
n = len(metrics_df)
for c in metrics_df.columns:
    mean_metrics[c] = metrics_df[c].mean()
    s = metrics_df[c].std()
    e = 1.96*(s/math.sqrt(n))
    ci_metrics[c] = (mean_metrics[c]-e, mean_metrics[c]+e)
    if c in ["BS", "ECE"]:
        metrics_str[c] = f"{metrics_df[c].mean():.4f} [{(mean_metrics[c]-e):.4f}, {(mean_metrics[c]+e):.4f}]"
    else:
        metrics_str[c] = f"{metrics_df[c].mean()*100.00:.2f} [{(mean_metrics[c]-e)*100.00:.2f}, {(mean_metrics[c]+e)*100.00:.2f}]"

#print(mean_metrics)
#print(ci_metrics)

for key in metrics_str.keys():
    print(f"{key}: {metrics_str[key]}")

print(seeds)