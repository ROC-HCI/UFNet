import pandas as pd
import numpy as np
from pandas import DataFrame
import os
import random

BASE_DIR = os.getcwd()+"/../../"
ALL_TASKS_IDS_FILE = os.path.join(BASE_DIR, "data/all_task_ids.txt")
DEMO_FILE = os.path.join(BASE_DIR, "data/demography_details.csv")
METADATA_FILE = os.path.join(BASE_DIR, "data/all_file_user_metadata.csv")

fusion_ids = []
with open(ALL_TASKS_IDS_FILE) as f:
    for x in f.readlines():
        fusion_ids.append(x.strip())

df_demo_all = pd.read_csv(METADATA_FILE)
df_demo_all["pid"] = df_demo_all["Participant_ID"].apply(lambda x: x.split("-")[-1])

'''
Filename, Protocol, Participant_ID, Task ,Duration
FPS, Frame_Height, Frame_Width, gender, age, race
ethnicity ,pd, dob, time_mdsupdrs, pid
'''
demo_all = []
for pid in fusion_ids:
    demo_data = {}
    demo_rows = df_demo_all[df_demo_all["pid"]==pid]
    demo_data["id"] = pid
    demo_data["pipeline"] = demo_rows.iloc[0]["Protocol"]
    demo_data["Diagnosis"] = demo_rows.iloc[0]["pd"]
    
    age = -1
    gender = "Unknown"
    race = "Unknown"
    for i in range(len(demo_rows)):
        row = demo_rows.iloc[i]
        
        if (not pd.isna(row["age"])) and (age==-1):
            age = row["age"]
        
        if (not pd.isna(row["gender"])) and (gender=="Unknown"):
            gender = row["gender"]

        if (not pd.isna(row["race"])) and (race=="Unknown"):
            race = row["race"]

        if (age!=-1) and (gender!="Unknown") and (race!="Unknown"):
            break

    demo_data["Age"] = age
    demo_data["Sex"] = gender
    demo_data["Race"] = race
    demo_all.append(demo_data)

df = pd.DataFrame.from_dict(demo_all)
df.loc[df["Race"]=="White","Race"] = 'white'
df.loc[df["Race"]=="['White']","Race"] = 'white'
df.loc[df["Race"].isna(),"Race"] = 'Unknown'
df.loc[df["Race"]=="['Prefer not to respond']","Race"] = 'Unknown'
df.loc[df["Race"]=="['Black or African American']","Race"] = 'Black or African American'
df.loc[df["Race"]=="['White', 'Asian']","Race"] = 'Asian'
df.loc[df["Race"]=="['Other']","Race"] = 'Others'
df.loc[df["Race"]=='white,',"Race"] = 'white'
df.loc[df["Race"]=='white,black,',"Race"] = 'Unknown'
df.loc[df["Race"]=='asian,',"Race"] = 'Asian'
df.loc[df["Race"]=='black,',"Race"] = 'Black or African American'
df.loc[df["Race"]=='white,race',"Race"] = 'white'
df.loc[df["Race"]=='asian,white,',"Race"] = 'Asian'
df.loc[df["Race"]=='on,',"Race"] = 'Unknown'
df.loc[df["Race"]=='asian,race',"Race"] = 'Asian'
df.loc[df["Race"]=="white","Race"] = 'white'
df.loc[df["Race"]=="['Asian', 'White']","Race"] = 'Asian'
df.loc[df["Race"]=="other","Race"] = 'Others'
df.loc[df["Race"]=='other,race',"Race"] = 'Others'
df.loc[df["Race"]=="['Asian']","Race"] = 'Asian'
df.loc[df["Race"]=='American Indian or Alaska Native',"Race"] = 'American Indian or Alaska Native'
df.loc[df["Race"]=='black,race', "Race"] = 'Black or African American'
df.loc[df["Race"]=="Asian","Race"] = 'Asian'
df.loc[df["Race"]=="Black or African American","Race"] = 'Black or African American'
df.loc[df["Race"]=='nativeAmerican,race',"Race"] = 'American Indian or Alaska Native'
df.loc[df["Race"]=='other,',"Race"] = 'Others'
df.loc[df["Race"]=='black',"Race"] = 'Black or African American'
print(df["Race"].unique())

df.loc[df["Sex"]=='male',"Sex"] = 'Male'
df.loc[df["Sex"]=='female',"Sex"] = 'Female'
df.loc[df["Sex"]=='Male',"Sex"] = 'Male'
df.loc[df["Sex"]=='Female',"Sex"] = 'Female'
df.loc[df["Sex"]=='Prefer not to respond',"Sex"] = 'Unknown'
df.loc[df["Sex"]=='Nonbinary',"Sex"] = 'Nonbinary'
print(df["Sex"].unique())

print(df["Age"].unique())
print(np.sum(df["Age"]==-1.0))
#df.loc[df["Age"]==-1.0,"Age"] = "Unknown"

df.loc[df["Diagnosis"]=='Unlikely',"Diagnosis"] = 'no'
df.loc[df["Diagnosis"]=='Possible',"Diagnosis"] = 'yes'
print(df["Diagnosis"].unique())

df.to_csv(DEMO_FILE, index=False)