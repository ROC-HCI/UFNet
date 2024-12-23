'''
Summary:
Generate demographic information about the participants
'''
import torch
import pandas as pd
import numpy as np
import os

BASE_DIR = os.getcwd()+"/../../"
DEMO_FILE = os.path.join(BASE_DIR, "data/demography_details.csv")

df_demo = pd.read_csv(DEMO_FILE)
n = len(df_demo)

'''
PD vs Non-PD
'''
print("Diagnosis",df_demo["Diagnosis"].unique())
pd_count = sum(df_demo["Diagnosis"]=="yes")
pd_pct = (pd_count/n)*100
print("PD", pd_count, pd_pct)

npd_count = sum(df_demo["Diagnosis"]=="no")
npd_pct = (npd_count/n)*100
print("Non-PD", npd_count, npd_pct)
print(f"Out of {n} individuals in this dataset, {pd_count} ({pd_pct}%) had PD. {npd_count} ({npd_pct}%) were non-PD")

'''
Gender
'''
print("Gender",df_demo["Sex"].unique())
male_count = sum(df_demo["Sex"]=="Male")
male_pct = (male_count/n)*100

female_count = sum(df_demo["Sex"]=="Female")
female_pct = (female_count/n)*100

unknown_count = sum(df_demo["Sex"]=="Unknown")
unknown_pct = (unknown_count/n)*100

nbin_count = sum(df_demo["Sex"]=="Nonbinary")
nbin_pct = (nbin_count/n)*100
print(f"Out of {n} individuals in this dataset, {male_count} ({male_pct}%) were Male. {female_count} ({female_pct}%) were Female. {unknown_count} ({unknown_pct}%) were Unknown. {nbin_count} ({nbin_pct}%) were Non-binary.")

female_pd_count = sum((df_demo["Sex"]=="Female") & (df_demo["Diagnosis"]=="yes"))
female_pct = (female_pd_count/pd_count)*100
print(f"female pd {female_pd_count}, {female_pct:.1f}")

male_pd_count = sum((df_demo["Sex"]=="Male") & (df_demo["Diagnosis"]=="yes"))
male_pct = (male_pd_count/pd_count)*100
print(f"male pd {male_pd_count}, {male_pct:.1f}")

nbin_pd_count = sum((df_demo["Sex"]=="Nonbinary") & (df_demo["Diagnosis"]=="yes"))
nbin_pct = (nbin_pd_count/pd_count)*100
print(f"nonbinary pd {nbin_pd_count}, {nbin_pct:.1f}")

unk_pd_count = sum((df_demo["Sex"]=="Unknown") & (df_demo["Diagnosis"]=="yes"))
unk_pct = (unk_pd_count/pd_count)*100
print(f"unknown pd {unk_pd_count}, {unk_pct:.1f}")

female_npd_count = sum((df_demo["Sex"]=="Female") & (df_demo["Diagnosis"]=="no"))
female_pct = (female_npd_count/npd_count)*100
print(f"female npd {female_npd_count}, {female_pct:.1f}")

male_npd_count = sum((df_demo["Sex"]=="Male") & (df_demo["Diagnosis"]=="no"))
male_pct = (male_npd_count/npd_count)*100
print(f"male npd {male_npd_count}, {male_pct:.1f}")

nbin_npd_count = sum((df_demo["Sex"]=="Nonbinary") & (df_demo["Diagnosis"]=="no"))
nbin_pct = (nbin_npd_count/pd_count)*100
print(f"nonbinary npd {nbin_npd_count}, {nbin_pct:.1f}")

unk_npd_count = sum((df_demo["Sex"]=="Unknown") & (df_demo["Diagnosis"]=="no"))
unk_pct = (unk_npd_count/pd_count)*100
print(f"unknown npd {unk_npd_count}, {unk_pct:.1f}")

'''
Age
'''
df_demo = df_demo.rename(columns={"Age":"age", "Race":"race"})
ages = np.asarray(df_demo["age"])
mean_age = np.mean(ages[ages!=-1])
std_age = np.std(ages[ages!=-1])
print("Age mean, std, min, max",mean_age, std_age, np.min(ages[ages!=-1]), np.max(ages))

def pct(x,n):
    return f"{((x*100)/n):.1f}"

print(f"Unknown age: {np.sum(df_demo['age']==-1.0)}, {pct(np.sum(df_demo['age']==-1.0), n)}")
a = sum(df_demo["age"]<0)
b = sum(df_demo["age"]<20)
print("Age group <20",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<30)
print("Age group 20-29",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<40)
print("Age group 30-39",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<50)
print("Age group 40-49",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<60)
print("Age group 50-59",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<70)
print("Age group 60-69",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]<80)
print("Age group 70-79",(b-a), pct((b-a), n))
a = b
b = sum(df_demo["age"]>=-1)
print("Age group 80+",(b-a), pct((b-a), n))


print("\nWITH PD\n")
df_demo_save = df_demo
df_demo = df_demo[df_demo['Diagnosis']=="yes"]
print(f"Unknown age: {np.sum(df_demo['age']==-1.0)}, {pct(np.sum(df_demo['age']==-1.0), pd_count)}")
a = sum(df_demo["age"]<0)
b = sum(df_demo["age"]<20)
print("Age group <20",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<30)
print("Age group 20-29",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<40)
print("Age group 30-39",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<50)
print("Age group 40-49",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<60)
print("Age group 50-59",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<70)
print("Age group 60-69",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]<80)
print("Age group 70-79",(b-a), pct((b-a), pd_count))
a = b
b = sum(df_demo["age"]>=-1)
print("Age group 80+",(b-a), pct((b-a), pd_count))
df_demo = df_demo_save

print("\nWITHOUT PD\n")
df_demo_save = df_demo
df_demo = df_demo[df_demo['Diagnosis']=="no"]
print(f"Unknown age: {np.sum(df_demo['age']==-1.0)}, {pct(np.sum(df_demo['age']==-1.0), npd_count)}")
a = sum(df_demo["age"]<0)
b = sum(df_demo["age"]<20)
print("Age group <20",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<30)
print("Age group 20-29",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<40)
print("Age group 30-39",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<50)
print("Age group 40-49",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<60)
print("Age group 50-59",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<70)
print("Age group 60-69",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]<80)
print("Age group 70-79",(b-a), pct((b-a), npd_count))
a = b
b = sum(df_demo["age"]>=-1)
print("Age group 80+",(b-a), pct((b-a), npd_count))
df_demo = df_demo_save

'''
Ethnicity
'''
print("\nRace\n")
races = df_demo["race"].unique()
print(races)
for r in races:    
    print(r,sum(df_demo["race"]==r), pct(sum(df_demo["race"]==r), n))


print("WITH PD")
df_demo = df_demo[df_demo['Diagnosis']=="yes"]
for r in races:    
    print(r,sum(df_demo["race"]==r), pct(sum(df_demo["race"]==r), pd_count))
df_demo = df_demo_save

print("WITHOUT PD")
df_demo = df_demo[df_demo['Diagnosis']=="no"]
for r in races:    
    print(r,sum(df_demo["race"]==r), pct(sum(df_demo["race"]==r), npd_count))
df_demo = df_demo_save

'''
Recording Environment
'''
print("\nEnvironment\n")
#['ParkTest' 'ClusterPD' 'ValidationStudy' 'InMotion' 'ValorPD'
# 'SuperPD_old' 'SuperPD']
df_demo.loc[df_demo["pipeline"]=="ParkTest","pipeline"] = 'Home'
df_demo.loc[df_demo["pipeline"]=="ClusterPD","pipeline"] = 'Clinic'
df_demo.loc[df_demo["pipeline"]=="ValidationStudy","pipeline"] = 'Care Facility'
df_demo.loc[df_demo["pipeline"]=="InMotion","pipeline"] = 'Care Facility'
df_demo.loc[df_demo["pipeline"]=="ValorPD","pipeline"] = 'Clinic'
df_demo.loc[df_demo["pipeline"]=="SuperPD_old","pipeline"] = 'Clinic'
df_demo.loc[df_demo["pipeline"]=="SuperPD","pipeline"] = 'Clinic'
envs = df_demo["pipeline"].unique()

for env in envs:    
    print(env, sum(df_demo["pipeline"]==env), pct(sum(df_demo["pipeline"]==env), n))

print("WITH PD")
df_demo = df_demo[df_demo['Diagnosis']=="yes"]
for env in envs:    
    print(env, sum(df_demo["pipeline"]==env), pct(sum(df_demo["pipeline"]==env), pd_count))
df_demo = df_demo_save

print("WITHOUT PD")
df_demo = df_demo[df_demo['Diagnosis']=="no"]
for env in envs:    
    print(env, sum(df_demo["pipeline"]==env), pct(sum(df_demo["pipeline"]==env), npd_count))
df_demo = df_demo_save
