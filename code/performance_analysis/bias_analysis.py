import os
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy import stats
import pickle

'''
Tow-sampled Z test for proportions
'''
def prop_test(n1, n2, p1, p2):
    p = (p1*n1+p2*n2)/(n1+n2)

    z = (p1-p2)/math.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    p = scipy.stats.norm.sf(abs(z))*2
    
    return (z,p)

def confidence_interval(n, p, test_type='z'):
    if n==0:
        return 0
    if test_type=="t":
        z_critical = scipy.stats.t.ppf(q=0.975, df = n-1)
    else:
        z_critical = 1.96
    return z_critical*math.sqrt((p*(1-p))/n)

#Load demography info
df_demo = pd.read_csv("../../data/demography_details.csv")

# Load seeds from the wandb experiments
df_seeds = pd.read_csv("wandb_reports/final_fusion_drop_preds_May132024.csv")
seeds = list(df_seeds["seed"])

subgroup_stats = {"sex":{}, "age":{}, "race":{}, "environment":{}}
subgroup_vals = {
    "sex":["male","female"], 
    "age":["20-29","30-39","40-49","50-59","60-69","70-79","80+"],
    "race":["white","non-white"],
    "environment":["home","clinic","care"]
    }
for grouping_criteria in subgroup_stats.keys():
    subgroup_stats[grouping_criteria] = {}
    for subgroup_value in subgroup_vals[grouping_criteria]:
        subgroup_stats[grouping_criteria][subgroup_value] = {"mean_errors":[], "std_errors":[], "n":[]}

mean_preds = {}
pd_labels = {}

for seed in seeds:
    filename = f'../fusion_models/uncertainty_aware_fusion/outputs/preds_with_ids_{seed}.pkl'
    with open(filename, 'rb') as f:
        loaded_results = pickle.load(f)
        ids, labels, pred_scores = loaded_results['all_ids'], loaded_results['all_labels'], loaded_results['all_preds']

        for id, label, pred_score in zip(ids, labels, pred_scores):
            if id not in mean_preds.keys():
                mean_preds[id] = [pred_score]
                pd_labels[id] = label
            else:
                mean_preds[id].append(pred_score)

mean_preds = {key: np.mean(mean_preds[key]) for key in mean_preds.keys()}
demo_pred_dicts = []
for key in mean_preds.keys():
    pred_dict = {"id":key, "label":pd_labels[key], "pred_score":mean_preds[key]}
    demo_pred_dicts.append(pred_dict)

df_preds = pd.DataFrame.from_dict(demo_pred_dicts)
df_demo_preds = pd.merge(left=df_demo, right=df_preds, on="id")
df_demo_preds.to_csv("demography_and_predictions_final_model.csv")
#,id,pipeline,Diagnosis,Age,Sex,Race,label,pred_score

filename = 'demography_and_predictions_final_model.csv'
df = pd.read_csv(filename)
df = df.rename(columns={"label":"True label"})
df["Predicted label"] = 1.0*(df["pred_score"]>=0.5)

'''
Tow-sampled Z test for proportions
'''
def prop_test(n1, n2, p1, p2):
    p = (p1*n1+p2*n2)/(n1+n2)

    z = (p1-p2)/math.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    p = scipy.stats.norm.sf(abs(z))*2
    
    return (z,p)

def confidence_interval(n, p, test_type='z'):
    if n==0:
        return 0
    if test_type=="t":
        z_critical = scipy.stats.t.ppf(q=0.975, df = n-1)
    else:
        z_critical = 1.96
    return z_critical*math.sqrt((p*(1-p))/n)

'''
Sex subgroups
=============
Null Hypothesis: "The model's misclassification rate is significantly different across male and female subjects"
Alternative Hypothesis: "There is no significant difference in the model's misclassification rate across male and female subjects"
'''
print(df["Sex"].unique()) #['male' 'female' 'Female' 'Male']
print("Male Female")
df_male = df[df["Sex"].str.lower()=="male"]
df_female = df[df["Sex"].str.lower()=="female"]

n_male, n_female = len(df_male), len(df_female)
print(n_male, n_female)

mean_error_male = np.mean(df_male["True label"]!=df_male["Predicted label"])
mean_error_female = np.mean(df_female["True label"]!=df_female["Predicted label"])
print(f"{mean_error_male*100:.1f}+-{confidence_interval(n_male, mean_error_male)*100:.1f}, {mean_error_female*100:.1f}+-{confidence_interval(n_female, mean_error_female)*100:.1f}")

ci_male = confidence_interval(n_male, mean_error_male)
ci_female = confidence_interval(n_female, mean_error_female)

p1 = mean_error_male
p2 = mean_error_female
n1 = n_male
n2 = n_female
z, pval = prop_test(n1, n2, p1, p2)
print(f"Z = {z}, p-value: {pval}")

# df.loc[df["Race"]=="White","Race"] = 'white'
# df.loc[df["Race"]=="['White']","Race"] = 'white'
# df.loc[df["Race"].isna(),"Race"] = 'Unknown'
# df.loc[df["Race"]=="['Prefer not to respond']","Race"] = 'Unknown'
# df.loc[df["Race"]=="['Black or African American']","Race"] = 'Black or African American'
# df.loc[df["Race"]=="['White', 'Asian']","Race"] = 'Asian'
# df.loc[df["Race"]=="['Other']","Race"] = 'Others'
# df.loc[df["Race"]=='white,',"Race"] = 'white'
# df.loc[df["Race"]=='white,black,',"Race"] = 'Unknown'
# df.loc[df["Race"]=='asian,',"Race"] = 'Asian'
# df.loc[df["Race"]=='black,',"Race"] = 'Black or African American'
# df.loc[df["Race"]=='white,race',"Race"] = 'white'
# df.loc[df["Race"]=='asian,white,',"Race"] = 'Asian'
# df.loc[df["Race"]=='on,',"Race"] = 'Unknown'
# df.loc[df["Race"]=='asian,race',"Race"] = 'Asian'
# df.loc[df["Race"]=="white","Race"] = 'white'
# df.loc[df["Race"]=="['Asian', 'White']","Race"] = 'Asian'
# df.loc[df["Race"]=="other","Race"] = 'Others'
# df.loc[df["Race"]=='other,race',"Race"] = 'Others'
# df.loc[df["Race"]=="['Asian']","Race"] = 'Asian'
# df.loc[df["Race"]=='American Indian or Alaska Native',"Race"] = 'American Indian or Alaska Native'
# df.loc[df["Race"]=='black,race', "Race"] = 'Black or African American'
# df.loc[df["Race"]=="Asian","Race"] = 'Asian'
# df.loc[df["Race"]=="Black or African American","Race"] = 'Black or African American'
# df.loc[df["Race"]=='nativeAmerican,race',"Race"] = 'American Indian or Alaska Native'
# df.loc[df["Race"]=='other,',"Race"] = 'Others'

'''
Race subgroups
'''
#print(df["Race"].unique()) #['white' 'Black or African American' 'Asian' 'Unknown' 'Others']
df.loc[df["Race"]=="white","Race"] = 'White'
df.loc[df["Race"]=="Unknown","Race"] = 'Not Mentioned'

print("White Non-white")
df1 = df[df["Race"]=="White"]
df2 = df[(df["Race"]!="White") & (df["Race"]!="Not Mentioned")]

n1, n2 = len(df1), len(df2)

inc1 = np.sum(df1["True label"]!=df1["Predicted label"])
inc2 = np.sum(df2["True label"]!=df2["Predicted label"])
c1 = np.sum(df1["True label"]==df1["Predicted label"])
c2 = np.sum(df2["True label"]==df2["Predicted label"])

p_white = inc1/n1
p_nwhite = inc2/n2
ci_white = confidence_interval(n1, p_white)
ci_nwhite = confidence_interval(n2, p_nwhite, test_type="t")

d = [[c1, inc1],[c2, inc2]]
odd_ratio, p_value = scipy.stats.fisher_exact(d)
print(n1, n2)
print(p_white, ci_white, p_nwhite, ci_nwhite)
print(odd_ratio, p_value)

'''
Correlation with Age
'''
ages = set(sorted(list(df["Age"])))
errors = []

for age in ages:
    data = df[df["Age"]==age]
    error = np.mean(data["True label"]!=data["Predicted label"])
    errors.append(error)
    
scipy.stats.spearmanr(list(ages), errors)
#plt.scatter(list(ages), errors)

df_20 = df[(df["Age"]<30) & (df["Age"]>=20)]
print(f"20-29: {len(df_20)}")
df_30 = df[(df["Age"]<40) & (df["Age"]>=30)]
print(f"30-39: {len(df_30)}")
df_40 = df[(df["Age"]<50) & (df["Age"]>=40)]
print(f"40-49: {len(df_40)}")
df_50 = df[(df["Age"]<60) & (df["Age"]>=50)]
print(f"50-59: {len(df_50)}")
df_60 = df[(df["Age"]<70) & (df["Age"]>=60)]
print(f"60-69: {len(df_60)}")
df_70 = df[(df["Age"]<80) & (df["Age"]>=70)]
print(f"70-79: {len(df_70)}")
df_g80 = df[df["Age"]>=80]
print(f"80+: {len(df_g80)}")

p_20 = np.mean(df_20["True label"]!=df_20["Predicted label"])
ci_20 = confidence_interval(len(df_20),p_20)

p_30 = np.mean(df_30["True label"]!=df_30["Predicted label"])
ci_30 = confidence_interval(len(df_30),p_30)

p_40 = np.mean(df_40["True label"]!=df_40["Predicted label"])
ci_40 = confidence_interval(len(df_40),p_40)

p_50 = np.mean(df_50["True label"]!=df_50["Predicted label"])
ci_50 = confidence_interval(len(df_50),p_50)

p_60 = np.mean(df_60["True label"]!=df_60["Predicted label"])
ci_60 = confidence_interval(len(df_60),p_60)

p_70 = np.mean(df_70["True label"]!=df_70["Predicted label"])
ci_70 = confidence_interval(len(df_70),p_70)

p_g80 = np.mean(df_g80["True label"]!=df_g80["Predicted label"])
ci_g80 = confidence_interval(len(df_g80),p_g80)
print(len(df_g80["True label"]))

'''
Recording environments
'''
#print(df['pipeline'].unique()) 'ParkTest' 'ClusterPD' 'InMotion' 'SuperPD' 'ValidationStudy' 'SuperPD_old' 'ValorPD']
df.loc[df["pipeline"]=="ParkTest","pipeline"] = 'Home'
df.loc[df["pipeline"]=="ClusterPD","pipeline"] = 'Clinic'
df.loc[df["pipeline"]=="InMotion","pipeline"] = 'Care Facility'
df.loc[df["pipeline"]=="SuperPD","pipeline"] = 'Clinic'
df.loc[df["pipeline"]=="ValidationStudy","pipeline"] = 'Care Facility'
df.loc[df["pipeline"]=="SuperPD_old","pipeline"] = 'Clinic'
df.loc[df["pipeline"]=="ValorPD","pipeline"] = 'Clinic'

print("Home Clinic Care-Facility")
df1 = df[df["pipeline"]=="Home"]
df2 = df[df["pipeline"]=="Clinic"]
df3 = df[df["pipeline"]=="Care Facility"]

n1, n2, n3 = len(df1), len(df2), len(df3)
print(n1, n2, n3)

inc1 = np.sum(df1["True label"]!=df1["Predicted label"])
inc2 = np.sum(df2["True label"]!=df2["Predicted label"])
inc3 = np.sum(df3["True label"]!=df3["Predicted label"])

c1 = np.sum(df1["True label"]==df1["Predicted label"])
c2 = np.sum(df2["True label"]==df2["Predicted label"])
c3 = np.sum(df3["True label"]==df3["Predicted label"])

p_home = inc1/n1
p_clinic = inc2/n2
p_care = inc3/n3

ci_home = confidence_interval(n1, p_home)
ci_clinic = confidence_interval(n2, p_clinic)
ci_care = confidence_interval(n3, p_care)

print("Home Clinic Care-facility")
print(p_home, ci_home, p_clinic, ci_clinic, p_care, ci_care)

print("Home vs Clinic")
d = [[c1, inc1],[c2, inc2]]
odd_ratio, p_value = scipy.stats.fisher_exact(d)
print(n1, n2)
print(p_home, p_clinic)
print(odd_ratio, p_value)

print("Home vs Care Facility")
d = [[c1, inc1],[c3, inc3]]
odd_ratio, p_value = scipy.stats.fisher_exact(d)
print(n1, n3)
print(p_home, p_care)
print(odd_ratio, p_value)

print("Clinic vs Care Facility")
d = [[c2, inc2],[c3, inc3]]
odd_ratio, p_value = scipy.stats.fisher_exact(d)
print(n2, n3)
print(p_clinic, p_care)
print(odd_ratio, p_value)

'''
PLOT
'''
eps = 0.001
labels = ['Female', 'Male', '', 'White', 'Non-white', '', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', 'Above 80']
x_pos = np.arange(len(labels))
CTEs = [mean_error_female, mean_error_male, 0, p_white, p_nwhite, 0, p_20, p_30, p_40, p_50, p_60, p_70, p_g80]
error = [ci_female, ci_male, 0, ci_white, ci_nwhite, 0, ci_20, ci_30, ci_40, ci_50, ci_60, ci_70, ci_g80]

for i in range(len(labels)):
    if (labels[i]!='') and (CTEs[i]==0):
        CTEs[i] = eps
        error[i] = eps

# Build the plot
fig, ax = plt.subplots()
errorbars = ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='orangered', color='blue', capsize=10)
ax.set_ylabel('Miss-classification rate', fontsize=16)
#ax.set_xlabel('Subgroup')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, fontsize=14)
#ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)
ax.set_ylim(0,1.0)
ax.tick_params(axis='y', labelsize=14)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(-0.28, 0.25, "85", bbox=props)
ax.text(0.78, 0.16, "77", bbox=props)

ax.text(2.50, 0.16, "118", bbox=props)
ax.text(3.70, 0.21, "18", bbox=props)

ax.text(5.86, 0.04, "6", bbox=props)
ax.text(6.85, 0.59, "5", bbox=props)
ax.text(7.85, 0.94, "6", bbox=props)
ax.text(8.70, 0.15, "26", bbox=props)
ax.text(9.75, 0.13, "66", bbox=props)
ax.text(10.70, 0.30, "29", bbox=props)
ax.text(11.90, 0.71, "4", bbox=props)

spacing = 0.100
fig.subplots_adjust(bottom=spacing)

# Save the figure and show
plt.tight_layout()
plt.savefig('missclassification_bias_updated.png',dpi=800)
plt.show()


'''
Plot with recording environments
'''
plt.clf()
eps = 0.001
labels = ['Female', 'Male', '', 'white', 'Non-white', '', 'Home', 'Clinic', 'PD Care Facility', '', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', 'Above 80']
x_pos = np.arange(len(labels))
CTEs = [mean_error_female, mean_error_male, 0, p_white, p_nwhite, 0, p_home, p_clinic, p_care, 0, p_20, p_30, p_40, p_50, p_60, p_70, p_g80]
error = [ci_female, ci_male, 0, ci_white, ci_nwhite, 0, ci_home, ci_clinic, ci_care, 0, ci_20, ci_30, ci_40, ci_50, ci_60, ci_70, ci_g80]

for i in range(len(labels)):
    if (labels[i]!='') and (CTEs[i]==0):
        CTEs[i] = eps
        error[i] = eps

# Build the plot
fig, ax = plt.subplots()
errorbars = ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='orangered', color='blue', capsize=10)
ax.set_ylabel('Miss-classification rate', fontsize=16)
#ax.set_xlabel('Subgroup')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=90, fontsize=14)
#ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)
ax.set_ylim(0,1.0)
ax.tick_params(axis='y', labelsize=14)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(-0.28, 0.25, "85", bbox=props)
ax.text(0.78, 0.16, "77", bbox=props)

ax.text(2.50, 0.16, "118", bbox=props)
ax.text(3.70, 0.21, "18", bbox=props)

ax.text(5.65, 0.09, "91", bbox=props)
ax.text(6.70, 0.30, "34", bbox=props)
ax.text(7.70, 0.45, "37", bbox=props)

ax.text(9.86, 0.04, "6", bbox=props)
ax.text(10.85, 0.59, "5", bbox=props)
ax.text(11.85, 0.94, "6", bbox=props)
ax.text(12.70, 0.15, "26", bbox=props)
ax.text(13.75, 0.13, "66", bbox=props)
ax.text(14.70, 0.30, "29", bbox=props)
ax.text(15.90, 0.71, "4", bbox=props)

spacing = 0.100
fig.subplots_adjust(bottom=spacing)

# Save the figure and show
plt.tight_layout()
plt.savefig('missclassification_sex_race_age_environment.png',dpi=800)
plt.show()