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
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

from constants import *

'''
Find the GPU that has max free space
'''
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

# results = get_gpu_memory()
# gpu_id = np.argmax(results)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#1. Load dev and test sets (participant ids)
with open(os.path.join(BASE_DIR,"data/dev_set_participants.txt")) as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])

with open(os.path.join(BASE_DIR,"data/test_set_participants.txt")) as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])
    
with open(os.path.join(BASE_DIR,"data/calib_set_participants.txt")) as f:
    ids = f.readlines()
    calib_ids = set([x.strip() for x in ids])
    

	
print(f"Number of patients in the dev and test set: {len(dev_ids)}, {len(test_ids)}")

#2. process the datasets
'''
Parse date from filenames. 
Some examples:
    2022-03-24T13%3A32%3A36.977Z_NIHNT179KNNF4_finger_tapping_left.mp4 -- 2022-03-24
    2021-08-30T20%3A00%3A03.162Z_ZTi20lXEMSdqXLxtnTotwoyADq03_finger_tapping_left.mp4 -- 2021-08-30
    NIHYM875FLXFF-finger_tapping-2021-03-17T18-13-01-902Z-.mp4 -- 2021-03-17
    2019-10-21T22-16-00-772Z35-finger_tapping.mp4 -- 2019-10-21
'''
def parse_date(name:str):
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    date = match.group()
    return date

def load_smile_data(drop_correlated = True, corr_thr = 0.85):
    df = pd.read_csv(FACIAL_FEATURES_FILE)

    #Fill data point by 0 if it is null
    df.fillna(0, inplace=True)
    
    '''
    Get the expression relavant feature columns and the feature dataframe
    '''
    feature_columns = []
    for feature in df.columns:
        for expression in FACIAL_EXPRESSIONS.keys():
            if FACIAL_EXPRESSIONS[expression] and expression in feature.lower():
                feature_columns.append(feature)
                break
    df_features = df[feature_columns]

    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= corr_thr:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
    
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = df_features.columns
    features = features.to_numpy()

    df["id"] = df['ID']
    df["date"] = df.Filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = 1.0*(df["pd"]!="no")

    return features, df["label"], df["id"], columns, df["id_date"]

def load_qbf_data(drop_correlated = False, corr_thr = 0.85, feature_files=[AUDIO_FEATURES_FILE]):
    def parse_patient_id(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        elif name.endswith("-quick_brown_fox.mp4"): [*_, ID, _] = name.split("-")
        elif name.endswith("_quick_brown_fox.mp4"): [_, ID, _, _, _] = name.split("_")
        else: [*_, ID, _, _, _] = name.split("_")
        return ID
    
    dataframes = []
    for FEATURES_FILE in feature_files:
        df_temp = pd.read_csv(FEATURES_FILE)
        dataframes.append(df_temp)

    assert (len(dataframes)>=1) and (len(dataframes)<=2)
    df = dataframes[0]
    #print(df.columns[:20]) #'Filename', 'Participant_ID', 'gender', 'age', 'race', 'pd', f'wavlm_feature{x}'
    for i in range(1,len(feature_files)):
        df = pd.merge(left=df, right=dataframes[i], how='inner', on='Filename')

    if len(dataframes)==2:
        df = df.drop(columns=['Participant_ID_y', 'gender_y', 'age_y', 'race_y', 'pd_y'])
        df = df.rename(columns={'Participant_ID_x':'Participant_ID', 'gender_x':'gender', 'age_x':'age', 'race_x':'race', 'pd_x':'pd'})

    '''
    Drop data point if any of the feature is null
    '''
    df = df.dropna(subset = df.columns.difference(['Filename','Participant_ID', 'gender','age','race']), how='any')
 
    #Drop metadata columns to focus on features
    df_features = df.drop(columns=['Filename','Participant_ID', 'gender','age','race','pd'])
 
    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []
 
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)
 
                if val >= corr_thr:
                    drop_cols.append(col.values[0])
 
        drops = set(drop_cols)
 
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
        
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = features.columns
    features = features.to_numpy()
    
    df["id"] = df.Filename.apply(parse_patient_id)
    df["date"] = df.Filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = df["pd"]
    return features, df["label"], df["id"], columns, df["id_date"]

def load_finger_data(hand="left",drop_correlated = False, corr_thr = 0.85):
    '''
    Parse ID from filenames. 
    Some examples:
        2022-03-24T13%3A32%3A36.977Z_NIHNT179KNNF4_finger_tapping_left.mp4 -- NIHNT179KNNF4
        2021-08-30T20%3A00%3A03.162Z_ZTi20lXEMSdqXLxtnTotwoyADq03_finger_tapping_left.mp4 -- ZTi20lXEMSdqXLxtnTotwoyADq03
        NIHYM875FLXFF-finger_tapping-2021-03-17T18-13-01-902Z-.mp4 -- NIHYM875FLXFF
        2019-10-21T22-16-00-772Z35-finger_tapping.mp4 -- 772Z35
    '''
    def parse_patient_id(name:str):
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        elif name.endswith("finger_tapping.mp4"): [*_, ID, _] = name.split("-")
        else: [*_, ID, _, _, _] = name.split("_")
        return ID

    df = pd.read_csv(FINGER_FEATURES_FILE)

    #Drop data point if any of the feature is null
    df = df.dropna(subset = df.columns.difference(['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity','dob','time_mdsupdrs']), how='any')
    
    '''
    Restrict only to one hand (if specified)
    '''    
    if hand!="both" and hand in ["left","right"]:
        df = df[df["hand"]==hand]

    #Drop metadata columns to focus on features
    df_features = df.drop(columns=['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity','pd','dob','time_mdsupdrs','hand'])
    '''
    Drop columns (if set true) if it is correlated with another one with PCC>thr
    '''
    if drop_correlated:
        corr_matrix = df_features.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                if val >= corr_thr:
                    drop_cols.append(col.values[0])

        drops = set(drop_cols)
        
        # Drop features from both the main and the feature dataframe
        df.drop(drops, axis=1, inplace=True)
        df_features.drop(drops, axis=1, inplace=True)
    # end of drop correlated columns implementation
    
    features = df.loc[:, df_features.columns[0]:df_features.columns[-1]]
    columns = features.columns
    features = features.to_numpy()
    
    df["id"] = df.filename.apply(parse_patient_id)
    df["date"] = df.filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    df["label"] = 1.0*(df["pd"]!="no")

    return features, df["label"], df["id"], columns, df["id_date"]

'''
Split the dataframe into train(+dev) and test sets
'''
def train_test_split(df):
    train_df = df[~df["id"].isin(test_ids)]
    test_df = df[df["id"].isin(test_ids)]
    return train_df, test_df

'''
Randomly split the train set into train and validation
dev_size is a ratio, i.e., 0.20 would mean 80% train - 20% dev
'''
def train_dev_split(train_df, dev_size=0.20):
    dev_df = train_df[train_df["id"].isin(dev_ids)]
    train_df = train_df[~train_df["id"].isin(dev_ids)]
    return train_df, dev_df

'''
Calibration Set Separation
'''
def train_calibration_split(train_df, dev_size=0.20):
    calib_df = train_df[train_df["id"].isin(calib_ids)]
    train_df = train_df[~train_df["id"].isin(calib_ids)]
    return train_df, calib_df

'''
Given a dataframe, perform oversampling
input_df: must contain columns features_0, features_1, ..., features_(N-1), and label
output_df: oversamples the minority class and returns in similar format
other columns (i.e., id, filename, etc.) will be removed.
'''
def concat_features(row):
    return np.concatenate([row[f"features_{i}"] for i in range(NUM_MODELS)])

def concat_finger_features(row):
    return np.concatenate([row[f"features_right"], row[f"features_left"]])

def oversample(input_df, sampler):
    feature_shapes = [input_df.iloc[0][f"features_{i}"].shape[0] for i in range(NUM_MODELS)]
    input_df["concat_features"] = input_df.apply(concat_features, axis=1)
    features = input_df.loc[:, "concat_features"]
    labels = input_df.loc[:,"label"]

    X = np.asarray([features.iloc[i] for i in range(len(features))])
    Y = np.asarray([labels.iloc[i] for i in range(len(labels))])

    X, Y = sampler.fit_resample(X, Y)
    output_data = []
    for (x,y) in zip(X,Y):
        data = {}
        start_index = 0
        for i in range(len(feature_shapes)):
            end_index = start_index + feature_shapes[i]
            data[f"features_{i}"] = x[start_index:end_index]
            start_index = end_index

        data["label"] = y
        output_data.append(data)

    output_df = pd.DataFrame.from_dict(output_data)
    return output_df

NUM_MODELS = 0 #just initiate here, later updated based on config

'''
Pytorch Dataset class
'''
class TensorDataset(Dataset):
    def __init__(self,df):
        '''
        df.columns: features_0, features_1, features_2, label, ...
        '''
        self.features = []
        for i in range(NUM_MODELS):
            f = torch.Tensor(np.asarray(df[f"features_{i}"].tolist()))
            self.features.append(f)

        self.labels = torch.Tensor(np.asarray(df["label"]))

    def __getitem__(self, index):
        features = []
        for i in range(NUM_MODELS):
            features.append(self.features[i][index])
        return features, self.labels[index]

    def __len__(self):
        return len(self.labels)

'''
ML baselines using pytorch + BAAL
'''
class ANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.drop1 = mcdropout.Dropout(p = drop_prob)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1,bias=True)
        self.drop2 = mcdropout.Dropout(p = drop_prob)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop1(x1)
        y = self.fc2(x1)
        y = self.drop2(y)
        y = self.sig(y)
        return y

'''
ML baselines using pytorch + BAAL
'''
class ShallowANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1,bias=True)
        self.drop = mcdropout.Dropout(p = drop_prob)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y

'''
Final predictor
Contains two modules:
    1. a custom cross-attention module
    2. prediction network
'''
class CrossAttention(nn.Module):
    def __init__(self, input_dim, query_dim, drop_prob, uncertainty_weight):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.drop_prob = drop_prob
        self.uncertainty_weight = uncertainty_weight
        
        self.form_query = torch.nn.Linear(input_dim, query_dim)
        self.form_key = torch.nn.Linear(input_dim, query_dim)
        self.form_value = torch.nn.Linear(input_dim, query_dim)

        self.drop = mcdropout.Dropout(p = drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.final_layer = torch.nn.Linear((NUM_MODELS-1) * self.input_dim, self.input_dim)

    def forward(self, features, prediction_variances):
        prediction_variances = torch.stack(prediction_variances).transpose(0,1) #(n, N)
        
        queries = []
        keys = []
        values = []
        for i in range(NUM_MODELS):
            q = self.form_query(features[i])
            q = self.drop(q)

            key = self.form_key(features[i])
            key = self.drop(q)

            val = self.form_value(features[i])
            val = self.drop(val)

            queries.append(q)
            keys.append(key)
            values.append(val)

        queries = torch.stack(queries) #(N, n, d)
        queries = queries.transpose(0,1) #(n, N, d)
        keys = torch.stack(keys) #(N, n, d)
        keys_T = keys.transpose(0,1).transpose(-1,-2) #(n, d, N)
        values = torch.stack(values) #(N, n, d)
        values = values.transpose(0,1) #(n, N, d)

        scores = torch.matmul(queries, keys_T) #(n, N, N)
        #scores = self.softmax(scores) #the mid dimension sums up to 1, e.g., rows of scores[0]

        vars = prediction_variances.repeat(1, NUM_MODELS).reshape(-1, NUM_MODELS, NUM_MODELS) #(n, N, N)
        vars = vars + prediction_variances.unsqueeze(dim=-1) #(n, N, N)
        scores = scores - self.uncertainty_weight*vars #(n, N, N)
        scores = self.softmax(scores)
        
        zs = torch.matmul(scores, values) #(n, N, d)
        z = zs.reshape((-1, NUM_MODELS*self.query_dim)) #(n, N*d)
        return z

class HybridFusionNetworkWithUncertainty(nn.Module):
    def __init__(self, feature_shapes, config):
        super(HybridFusionNetworkWithUncertainty, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.query_dim = config["query_dim"]
        self.last_hidden_dim = config["last_hidden_dim"]
        self.drop_prob = config["dropout_prob"]
        self.uncertainty_weight = config["uncertainty_weight"]
        '''
        input: features_i is of shape (feature_shapes[i]); y_pred_score_i
        total input size: feature_shapes[i]+1
        '''
        self.intra_linear = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = CrossAttention(self.hidden_dim, self.query_dim, self.drop_prob, self.uncertainty_weight) #shared weights

        for i in range(NUM_MODELS):
            linear_layer = nn.Linear(in_features=feature_shapes[i], out_features=self.hidden_dim, bias=True)
            self.intra_linear.append(linear_layer)

        self.lin1 = nn.Linear(in_features=((NUM_MODELS*self.query_dim)+NUM_MODELS), out_features=self.last_hidden_dim)
        self.fc = nn.Linear(in_features=self.last_hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.intra_linear_dropout = mcdropout.Dropout(p = self.drop_prob)
        self.lin1_dropout = mcdropout.Dropout(p = self.drop_prob)
    
    def forward(self, inputs):
        (features, predicted_scores, prediction_variances) = inputs
        # print([features[i].shape for i in range(len(features))]) #(n, 232), (n, 1024), (n, 42)
        
        hiddens = []
        for i in range(NUM_MODELS):
            hidden_representation = self.relu(self.intra_linear[i](features[i])) #projection: (n, d_{x_i}) -> (n,d)
            hidden_representation = self.intra_linear_dropout(hidden_representation)
            hidden_representation = self.relu(hidden_representation)
            hidden_representation = self.layer_norm(hidden_representation)
            hiddens.append(hidden_representation)

        pred_scores = torch.stack(predicted_scores).transpose(0,1) #(n, N)
        context = self.cross_attention(torch.cat([torch.unsqueeze(hiddens[k],0) for k in range(NUM_MODELS)]), prediction_variances) #(n, d_q)
        outputs = torch.cat((context, pred_scores),dim=-1) #(n, N+d_q)
        outputs = self.lin1(outputs) #(n, last_hidden_dim)
        outputs = self.lin1_dropout(outputs) 
        logits = self.fc(outputs) #(n,1)
        probs = self.sigmoid(logits) #(n,1)
        return probs

'''
Evaluate performance on validation/test set.
Returns all the metrics defined above and the loss.
'''
def expected_calibration_error(y, y_pred_scores, num_buckets=20):
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

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

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
    metrics['PPV'] = metrics['precision'] = precision_score(labels, preds)
    
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

    '''
    Brier score
    '''
    metrics['BS'] = brier_score_loss(labels, pred_scores)

    '''
    Expected Calibration Error
    '''
    metrics['ECE'] = expected_calibration_error(labels, pred_scores)
    
    return metrics

def fit_platt_scaling(calibration_logits, calibration_labels):
    """
    Fits a logistic regression model for Platt scaling.
    """
    platt_model = LogisticRegression()
    calibration_logits = calibration_logits.reshape(-1, 1)
    platt_model.fit(calibration_logits, calibration_labels)
    return platt_model

def apply_platt_scaling(platt_model, logits):
    """
    Applies Platt scaling to logits.
    """
    logits = logits.reshape(-1, 1)
    calibrated_probs = platt_model.predict_proba(logits)[:, 1]  # Get probability of class 1
    return calibrated_probs



'''
Main evaluation loop to test the fusion model
'''
def evaluate_fusion_model(fusion_model, dataloader, prediction_models, config, split="dev", conformity_threshold=0.5, platt_model=None):
    fusion_model.eval()
    z_critical = scipy.stats.t.ppf(q=0.975, df = config["num_trials"]-1)

    all_labels = [] #true labels
    all_pred_scores = [[] for i in range(NUM_MODELS)] #unimodal prediction scores
    all_final_predictions = [] #fusion predictions
    uncertain_indices = [] #indices where the 95% CI contains 0.5
    loss = 0 #average loss
    n_samples = 0 #number of examples in the dataloader

    criterion = torch.nn.BCELoss() #loss function
    wrapped_prediction_models = [ModelWrapper(prediction_models[i],criterion) for i in range(NUM_MODELS)]
    fusion_model.eval()
    wrapped_fusion_model = ModelWrapper(fusion_model, criterion)
    
    for idx, batch in enumerate(dataloader):
        x = [[] for i in range(NUM_MODELS)] #[x0, x1, ..., xn]
        y_pred_scores = [[] for i in range(NUM_MODELS)] #probs[y0, y1, ..., yn]
        y_preds = [[] for i in range(NUM_MODELS)] #binary[y0, y1, ..., yn]
        y_vars = [[] for i in range(NUM_MODELS)]
        
        (x, y) = batch
        y = y.to(device)

        for i in range(NUM_MODELS):
            x[i] = x[i].to(device)

            if (split!="test") and (config["validation_random_noise"]=="yes"):
                noise = torch.randn(x[i].shape).to(device)
                adjusted_noise = noise*config["noise_variance"]
                x[i] += adjusted_noise

            y_multi_preds = wrapped_prediction_models[i].predict_on_batch(x[i], iterations=config["num_trials"])
            y_pred_scores[i] = y_multi_preds.mean(dim=-1).reshape(-1)
            y_vars[i] = y_multi_preds.std(dim=-1).reshape(-1)
            y_preds[i] = (y_pred_scores[i]>=0.5)
            all_pred_scores[i].extend(y_pred_scores[i].to('cpu').numpy())

        all_labels.extend(y.to('cpu').numpy())
        
        #forward pass
        with torch.no_grad():
            
            final_pred_scores = wrapped_fusion_model.predict_on_batch((x, y_pred_scores, y_vars), iterations=1)
            standard_error = (z_critical*final_pred_scores.std(dim=-1).reshape(-1))/math.sqrt(len(final_pred_scores))
            final_pred_scores = final_pred_scores.mean(dim=-1).reshape(-1)
            index_mask = (final_pred_scores-standard_error<=0.50) & (final_pred_scores+standard_error>=0.50)
            n = final_pred_scores.shape[0]
            loss += criterion(final_pred_scores.reshape(-1), y)*n
            n_samples+=n
        
        if split == "test" and platt_model is not None:
            #print("Before Platt Scaling")
            #print(final_pred_scores)
            final_pred_scores = torch.tensor(apply_platt_scaling(platt_model, final_pred_scores.cpu().numpy())).to(device)
            #print("After Platt Scaling")
            #print(final_pred_scores)
        all_final_predictions.extend(final_pred_scores.cpu().numpy())
        uncertain_indices.extend(index_mask.cpu().numpy())

    #evaluate
    uncertain_indices = np.asarray(uncertain_indices).flatten()
    all_labels = np.asarray(all_labels).flatten()
    all_final_predictions = np.asarray(all_final_predictions).flatten()
    
    # if split=="test":
    #     coverage = (len(all_labels) - uncertain_indices.sum())/len(all_labels)
    #     all_labels = all_labels[~uncertain_indices]
    #     all_final_predictions = all_final_predictions[~uncertain_indices]

    # now for test set, we will apply conformal prediction 
    
    # print(all_final_predictions)
    
    if split=="test":
        count_single = 0
        index_mask_conformal = []
        all_final_predictions_conformal = []
        for i in range(len(all_final_predictions)):
            prediction_set = []
            if all_final_predictions[i] < 1 - conformity_threshold:
                prediction_set = [0]
                count_single += 1
                index_mask_conformal.append(False)
            elif all_final_predictions[i] >= conformity_threshold:
                prediction_set = [1]
                count_single += 1
                index_mask_conformal.append(False)
            else:
                prediction_set = [0,1]
                index_mask_conformal.append(True)
                
            all_final_predictions_conformal.append(prediction_set)
            
        index_mask_conformal = np.asarray(index_mask_conformal).flatten()
        
        all_labels = all_labels[~index_mask_conformal]
        all_final_predictions = all_final_predictions[~index_mask_conformal]
                 
        print(f"Number of single predictions: {count_single}")
        print(f"Number of conformal predictions: {len(all_final_predictions_conformal)}")
        # print the coverage percentage
        print(f"Coverage: {round(count_single/len(all_final_predictions_conformal), 2)}")
        # wandb.log({"conformal_coverage": round(count_single/len(all_final_predictions_conformal), 2)})
        
         
    
    
    
    metrics = compute_metrics(all_labels, all_final_predictions)
    metrics["loss"] = loss.to('cpu').item() / n_samples
    
    # if split=="test":
    #     metrics['coverage'] = coverage

    return metrics


@click.command()
@click.option("--learning_rate", default=0.001, help="Learning rate for classifier")
@click.option("--dropout_prob", default=0.25)
@click.option("--num_buckets", default=20, help="Options: 5, 10, 20, 50, 100")
@click.option("--num_trials", default=30, help="Options: 100-1000")
@click.option("--uncertainty_weight", default=0.01)
@click.option("--minority_oversample",default='no',help="Options: 'yes', 'no'")
@click.option("--sampler", default='SMOTE', help="Options:SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, RandomOverSampler, SMOTEENN, SMOTETomek")
@click.option("--train_random_noise", default="no", help="Options: yes, no")
@click.option("--validation_random_noise", default="no", help="Options: yes, no")
@click.option("--increase_variance",default="no", help="Options: yes, no")
@click.option("--temperature", default=0.05, help="Float between 0 and 1")
@click.option("--noise_variance",default=0.01,help="Float between 0 and 1")
@click.option("--random_state", default=171, help="Random state for classifier")
@click.option("--model_subset_choice", default=0, help="4 possible choices. See Constants.py")
@click.option("--seed", default=113, help="Seed for random")
@click.option("--batch_size",default=64)
@click.option("--num_epochs",default=244)
@click.option("--hidden_dim", default=128)
@click.option("--query_dim", default=64)
@click.option("--last_hidden_dim", default=8)
@click.option("--optimizer",default="AdamW",help="Options: SGD, AdamW, RMSprop")
@click.option("--beta1",default=0.9)
@click.option("--beta2",default=0.999)
@click.option("--weight_decay",default=0.0001)
@click.option("--momentum",default=0.5317318147195794)
@click.option("--use_scheduler",default='yes',help="Options: yes, no")
@click.option("--scheduler",default='reduce',help="Options: step, reduce")
@click.option("--step_size",default=21)
@click.option("--gamma",default=0.34188571201807494)
@click.option("--patience",default=6)
def main(**cfg):
    global NUM_MODELS

    ENABLE_WANDB = True
    if ENABLE_WANDB:
        wandb.init(project="park_final_experiments", config=cfg)

    #reproducibility control
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    selected_models = MODEL_SUBSETS[cfg["model_subset_choice"]]
    
    NUM_MODELS = len(selected_models)
    '''
    Load the paths of the saved models that will be fused together
    '''
    model_paths = []
    
    for i in range(NUM_MODELS):
        path = {}
        MODEL_TAG = selected_models[i]
        path["PREDICTOR_CONFIG"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"predictive_model/model_config.json")
        path["PREDICTOR_MODEL"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"predictive_model/model.pth")
        path["SCALER"] = os.path.join(MODEL_BASE_PATH, MODEL_TAG,"scaler/scaler.pth")
        model_paths.append(path)

    for i in range(NUM_MODELS):
        for key, path in model_paths[i].items():
            assert os.path.exists(path)

    # print("All scaler, and predictive model paths are set and loaded.")
    
    processed_datasets = []
    for i in range(NUM_MODELS):
        predictor_config = {}
        with open(model_paths[i]["PREDICTOR_CONFIG"]) as json_file:
            predictor_config = json.load(json_file)

        drop_correlated = (predictor_config["drop_correlated"]=='yes')

        model_name = selected_models[i]

        if "finger_model" in model_name:
            features_right, labels_right, ids_right, columns, id_dates_right = load_finger_data(drop_correlated=drop_correlated, corr_thr=predictor_config["corr_thr"], hand="right")
            features_left, labels_left, ids_left, columns, id_dates_left = load_finger_data(drop_correlated=drop_correlated, corr_thr=predictor_config["corr_thr"], hand="left")
            
            df_right = pd.DataFrame.from_dict({"features_right":list(features_right), "id_right":ids_right, "row_id":id_dates_right, "label_right":labels_right})
            df_left = pd.DataFrame.from_dict({"features_left":list(features_left), "id_left":ids_left, "row_id":id_dates_left, "label_left":labels_left})
            
            df_both = pd.merge(df_right, df_left, how="inner", on="row_id")
            df_both = df_both.drop(columns=['label_left', 'id_left'])
            df_both = df_both.rename(columns={"label_right":"label", "id_right":"id"})
            df_both["features"] = df_both.apply(concat_finger_features, axis=1)
            
            features = np.stack(df_both.loc[:, "features"])
            labels = df_both.loc[:, "label"]
            ids = df_both.loc[:, "id"]
            row_ids = df_both.loc[:, "row_id"]
        elif "fox_model" in model_name:
            features, labels, ids, columns, row_ids = load_qbf_data(drop_correlated = drop_correlated, corr_thr = predictor_config["corr_thr"])
        elif "facial_expression_smile" in model_name:
            features, labels, ids, columns, row_ids = load_smile_data(drop_correlated = drop_correlated, corr_thr = predictor_config["corr_thr"])
        else:
            assert(False, "Unfamiliar model name.")

        # scale if needed
        if predictor_config["use_feature_scaling"]=="yes":
            scaler = pickle.load(open(model_paths[i]['SCALER'],'rb'))
            features = scaler.transform(features)

        all_data = pd.DataFrame.from_dict({f"features_{i}":(list)(features), f"label_{i}":labels, f"id_{i}":ids, "row_id":row_ids})
        processed_datasets.append(all_data)

        if i>0:
            all_data = pd.merge(processed_datasets[i-1], processed_datasets[i], on="row_id")
            all_data = all_data.drop(columns=[f'label_{i}', f'id_{i}'])
            processed_datasets[i] = all_data

    df = processed_datasets[NUM_MODELS-1]
    df = df.rename(columns={"label_0":"label", "id_0":"id"})
    
    print("Data of finger tapping, audio, and smile is combined and loaded.")
    
    train_df, test_df = train_test_split(df)
    train_df, dev_df = train_dev_split(train_df)
    train_df, calib_df = train_calibration_split(train_df)
    
    print(f"Number of training samples: {len(train_df)}. Positive class: {len(train_df[train_df['label']==1.0])}, Negative class: {len(train_df[train_df['label']==0.0])}.")
    print(f"Number of validation samples: {len(dev_df)}. Positive class: {len(dev_df[dev_df['label']==1.0])}, Negative class: {len(dev_df[dev_df['label']==0.0])}.")
    print(f"Number of test samples: {len(test_df)}. Positive class: {len(test_df[test_df['label']==1.0])}, Negative class: {len(test_df[test_df['label']==0.0])}.")
    
    if cfg["minority_oversample"]=="yes":
        if cfg["sampler"] == "SMOTE":
            sampler = SMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTENC":
            sampler = SMOTENC(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SVMSMOTE":
            sampler = SVMSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "ADASYN":
            sampler = ADASYN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "KMeansSMOTE":
            sampler = KMeansSMOTE(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTEN":
            sampler = SMOTEN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "RandomOverSampler":
            sampler = RandomOverSampler(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTEENN":
            sampler = SMOTEENN(random_state = cfg['random_state'])
        elif cfg["sampler"] == "SMOTETomek":
            sampler = SMOTETomek(random_state = cfg['random_state'])
        else:
            raise ValueError("Invalid sampler")

        train_df = oversample(train_df, sampler)

    train_dataset = TensorDataset(train_df)
    dev_dataset = TensorDataset(dev_df)
    test_dataset = TensorDataset(test_df)
    calib_dataset = TensorDataset(calib_df)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])
    calib_loader = DataLoader(calib_dataset, batch_size = cfg['batch_size'])

    features, label = train_dataset[0]
    feature_shapes = [features[i].shape[0] for i in range(NUM_MODELS)]
    
    prediction_models = []
    for i in range(NUM_MODELS):
        predictor_config = {}
        with open(model_paths[i]["PREDICTOR_CONFIG"]) as json_file:
            predictor_config = json.load(json_file)

        if predictor_config["model"]=="ShallowANN":    
            frozen_model = ShallowANN(feature_shapes[i], drop_prob = predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"]))
            frozen_model = frozen_model.to(device)
            frozen_model.fc.weight.requires_grad = False
            frozen_model.fc.bias.requires_grad = False
        else:
            frozen_model = ANN(feature_shapes[i], drop_prob = predictor_config["dropout_prob"])
            frozen_model.load_state_dict(torch.load(model_paths[i]["PREDICTOR_MODEL"]))
            frozen_model = frozen_model.to(device)
            frozen_model.fc1.weight.requires_grad = False
            frozen_model.fc1.bias.requires_grad = False
            frozen_model.fc2.weight.requires_grad = False
            frozen_model.fc2.bias.requires_grad = False

        prediction_models.append(frozen_model)

    print("All the prediction models are loaded as frozen.")

    fusion_model = HybridFusionNetworkWithUncertainty(feature_shapes, cfg)
    fusion_model = fusion_model.to(device)
    
    criterion = nn.BCELoss()
    if cfg["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(fusion_model.parameters(),lr=cfg['learning_rate'],betas=(cfg['beta1'],cfg['beta2']),weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(fusion_model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="RMSprop":
        optimizer = torch.optim.RMSprop(fusion_model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Invalid optimizer")

    if cfg["use_scheduler"]=="yes":
        if cfg['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        elif cfg['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['gamma'], patience = cfg['patience'])
        else:
            raise ValueError("Invalid scheduler")

    best_model = copy.deepcopy(fusion_model)
    best_dev_loss = np.finfo('float32').max
    best_dev_accuracy = 0
    best_dev_balanced_accuracy = 0
    best_dev_auroc = 0
    best_dev_f1 = 0
    wrapped_prediction_models = [ModelWrapper(prediction_models[i],criterion) for i in range(NUM_MODELS)]

    epoch_no = 0

    for epoch in tqdm(range(cfg['num_epochs'])):
        noise_variance = 0.0
        if (cfg["train_random_noise"]=="yes") and (cfg["increase_variance"]=="yes"):
            noise_variance = cfg["noise_variance"]*(1-math.exp(-epoch_no*cfg["temperature"]))
            epoch_no +=1

        all_labels = []
        all_pred_scores = [[] for i in range(NUM_MODELS)]
        
        for idx, batch in enumerate(train_loader):
            x = [[] for i in range(NUM_MODELS)]
            (x, y) = batch
            y = y.to(device)
            y_pred_scores = [[] for i in range(NUM_MODELS)]
            y_preds = [[] for i in range(NUM_MODELS)]
            y_vars = [[] for i in range(NUM_MODELS)]
            
            for i in range(NUM_MODELS):
                x[i] = x[i].to(device)
                if cfg["train_random_noise"]=="yes":
                    noise = torch.randn(x[i].shape).to(device)
                    adjusted_noise = noise*noise_variance
                    x[i] += adjusted_noise
                
                y_multi_preds = wrapped_prediction_models[i].predict_on_batch(x[i], iterations=cfg["num_trials"])
                y_pred_scores[i] = y_multi_preds.mean(dim=-1).reshape(-1)
                y_preds[i] = (y_pred_scores[i]>=0.5)
                y_vars[i] = y_multi_preds.std(dim=-1).reshape(-1)
                
                all_pred_scores[i].extend(y_pred_scores[i].to('cpu').numpy())

            all_labels.extend(y.to('cpu').numpy())
            
            #forward pass
            optimizer.zero_grad()
            final_predictions = fusion_model((x,y_pred_scores, y_vars))
            l = criterion(final_predictions.reshape(-1),y)
            l.backward()
            optimizer.step()

            if ENABLE_WANDB:
                wandb.log({"train_loss": l.to('cpu').item()})

        #eval on dev set
        dev_metrics = evaluate_fusion_model(fusion_model, dev_loader, prediction_models, cfg)
        dev_loss = dev_metrics["loss"]
        dev_accuracy = dev_metrics["accuracy"]
        dev_balanced_accuracy = dev_metrics["weighted_accuracy"]
        dev_auroc = dev_metrics["auroc"]
        dev_f1 = dev_metrics["f1_score"]
        dev_ece = dev_metrics["ECE"]
        #print(f"Epoch {epoch}: dev accuracy: {dev_metrics['accuracy']}")

        if cfg['use_scheduler']=="yes":
            if cfg['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_loss)

        if dev_loss<best_dev_loss:
            best_model = copy.deepcopy(fusion_model)

            best_dev_loss = dev_loss
            best_dev_accuracy = dev_accuracy
            best_dev_balanced_accuracy = dev_balanced_accuracy
            best_dev_auroc = dev_auroc
            best_dev_f1 = dev_f1
            best_dev_ece = dev_ece
    
    # After training, collect logits and labels from calibration set
    logits, labels = [], []

    fusion_model.eval()
    with torch.no_grad():
        for batch in calib_loader:
            x = [[] for i in range(NUM_MODELS)]
            (x, y) = batch
            y = y.to(device)
            y_pred_scores = [[] for i in range(NUM_MODELS)]
            y_vars = [[] for i in range(NUM_MODELS)]

            for i in range(NUM_MODELS):
                x[i] = x[i].to(device)
                y_multi_preds = wrapped_prediction_models[i].predict_on_batch(x[i], iterations=cfg["num_trials"])
                y_pred_scores[i] = y_multi_preds.mean(dim=-1).reshape(-1)
                y_vars[i] = y_multi_preds.std(dim=-1).reshape(-1)

            final_predictions = fusion_model((x, y_pred_scores, y_vars))
            logits.extend(final_predictions.reshape(-1).cpu().numpy())
            labels.extend(y.reshape(-1).cpu().numpy())

    # Fit Platt scaling model
    platt_model = fit_platt_scaling(np.array(logits), np.array(labels))

    # After the training loop ends, use the calibration set to determine conformity scores

    calibration_scores = []

    # Loop through the calibration set to compute conformity scores
    with torch.no_grad():
        for batch in calib_loader:
            x_calib = [[] for i in range(NUM_MODELS)]
            (x_calib, y_calib) = batch
            y_calib = y_calib.to(device)
            y_pred_scores_calib = [[] for i in range(NUM_MODELS)]
            y_vars_calib = [[] for i in range(NUM_MODELS)]
            
            for i in range(NUM_MODELS):
                x_calib[i] = x_calib[i].to(device)
                
                # Get multiple stochastic predictions for calibration
                y_multi_preds_calib = wrapped_prediction_models[i].predict_on_batch(x_calib[i], iterations=cfg["num_trials"])
                y_pred_scores_calib[i] = y_multi_preds_calib.mean(dim=-1).reshape(-1)
                y_vars_calib[i] = y_multi_preds_calib.std(dim=-1).reshape(-1)
            
            # Fuse the predictions using the fusion model
            final_calib_predictions = best_model((x_calib, y_pred_scores_calib, y_vars_calib))
            
            # get the calibrated probabilities
            #print("Before Platt Scaling")
            #print(final_calib_predictions)
            final_calib_predictions = torch.tensor(apply_platt_scaling(platt_model, final_calib_predictions.cpu().numpy())).to(device)
            #print("After Platt Scaling")
            #print(final_calib_predictions)
            
            # Compute conformity scores based on residuals |y - f(x)|
            conformity_scores = torch.abs(final_calib_predictions.reshape(-1) - y_calib)
            calibration_scores.extend(conformity_scores.cpu().numpy())

    # Determine the conformity threshold for 95% confidence
    alpha = 0.05
    threshold = np.percentile(calibration_scores, 100 * (1 - alpha))

    # # Save the threshold and best model for inference
    # torch.save({
    #     'model_state_dict': best_model.state_dict(),
    #     'calibration_threshold': threshold
    # }, 'calibrated_model.pth')

    print(f"Calibration completed. Threshold for 95% confidence: {threshold}")


    test_metrics = evaluate_fusion_model(best_model, test_loader, prediction_models, cfg, split="test", conformity_threshold=threshold, platt_model=platt_model)
    if ENABLE_WANDB:
        wandb.log(test_metrics)
        wandb.log({"dev_accuracy":best_dev_accuracy, "dev_balanced_accuracy":best_dev_balanced_accuracy, "dev_loss":best_dev_loss, "dev_auroc":best_dev_auroc, "dev_f1":best_dev_f1, "dev_ece":best_dev_ece})
    print(test_metrics)

if __name__ == "__main__":
    main()
