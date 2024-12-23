import torch
import os
import copy
import pickle
import pandas as pd
from pandas import DataFrame

from tqdm import tqdm
import baal.bayesian.dropout as mcdropout
from baal.modelwrapper import ModelWrapper

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlxtend.plotting import plot_confusion_matrix

import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import wandb
import random
import click
import json
import re
import imblearn
from imblearn.over_sampling import SMOTE

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from constants_baal import *

'''
Find the GPU that has max free space
'''
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
results = get_gpu_memory()
gpu_id = np.argmax(results)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

'''
set-up device (for gpu support)
'''
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Running on {device} ...")

if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)
    os.mkdir(os.path.join(MODEL_BASE_PATH,"predictive_model"))
    os.mkdir(os.path.join(MODEL_BASE_PATH,"residual_model"))
    os.mkdir(os.path.join(MODEL_BASE_PATH,"scaler"))

if not os.path.exists(os.path.join(MODEL_BASE_PATH,"predictive_model")):
    os.mkdir(os.path.join(MODEL_BASE_PATH,"predictive_model"))

if not os.path.exists(os.path.join(MODEL_BASE_PATH,"residual_model")):
    os.mkdir(os.path.join(MODEL_BASE_PATH,"residual_model"))

if not os.path.exists(os.path.join(MODEL_BASE_PATH,"scaler")):
    os.mkdir(os.path.join(MODEL_BASE_PATH,"scaler"))

'''
Load dev and test sets for this task
'''
with open(os.path.join(BASE_DIR,"data/dev_set_participants.txt")) as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])

with open(os.path.join(BASE_DIR,"data/test_set_participants.txt")) as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])

assert len(dev_ids.intersection(test_ids))==0

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

def load(drop_correlated = True, corr_thr = 0.85, hand="both"):
    df = pd.read_csv(FEATURES_FILE)

    '''
    Drop data point if any of the feature is null
    '''
    df = df.dropna(subset = df.columns.difference(['Unnamed: 0','filename','Protocol','Participant_ID','Task',
                'Duration','FPS','Frame_Height','Frame_Width','gender','age','race',
                'ethnicity','dob','time_mdsupdrs']), how='any')
    
    '''
    Restrict only to one hand (if specified)
    '''    
    if hand!="both" and hand in ["left","right"]:
        df = df[df["hand"]==hand]

    '''
    Drop metadata columns to focus on features
    '''
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

    '''
    Labels are not often consistent across datasets.
    Typically, "yes", "Possible", "Probable", "maybe" are considered positive.
    "no", and "Unlikely" are considered negative.
    '''
    #print(df["pd"].unique()) #['no' 'yes' 'maybe']: 'no' -> negative; 'yes', 'maybe' -> positive
    labels = 1.0*(df["pd"]!="no")
    df["id"] = df.filename.apply(parse_patient_id)
    df["date"] = df.filename.apply(parse_date)
    df["id_date"] = df["id"]+"#"+df["date"]
    return features, labels, df["id"], df["id_date"], columns
'''
Based on the predefined test split, split the dataframe into train+dev and test sets
'''
def train_test_split(features, labels, ids):
    features_train = []
    labels_train = []
    ids_train = []

    features_test = []
    labels_test = []
    ids_test = []

    for x, l, pid in zip(features, labels, ids):
        if pid not in test_ids:
            ids_train.append(pid)
            features_train.append(x)
            labels_train.append(l)
        elif pid in test_ids:
            ids_test.append(pid)
            features_test.append(x)
            labels_test.append(l)

    # print("IDs that are in the full test set, but not in this test dataset")
    # absent_ids = set(test_ids).difference(set(ids_test))
    # absent_ids = [line + '\n' for line in absent_ids]
    # with open("missing_test_ids.txt","w") as f:
    #     f.writelines(absent_ids)

    return features_train, labels_train, ids_train, features_test, labels_test, ids_test

'''
Based on the predefined dev split, split the dataframe into train and dev sets
'''
def train_dev_split(features, labels, ids):
    features_train = []
    labels_train = []
    ids_train = []

    features_dev = []
    labels_dev = []
    ids_dev = []

    for (x, l, pid) in zip(features, labels, ids):
        if pid not in dev_ids:
            ids_train.append(pid)
            features_train.append(x)
            labels_train.append(l)
        else:
            ids_dev.append(pid)
            features_dev.append(x)
            labels_dev.append(l)

    print(f"Size of the train set: {len(labels_train)}, dev set: {len(labels_dev)}")

    return features_train, labels_train, ids_train, features_dev, labels_dev, ids_dev

'''
Pytorch Dataset class
'''
class TensorDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.Tensor(np.asarray(features))
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

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

def evaluate(model, dataloader, num_trials, num_buckets):
    model.eval()

    all_preds = []
    all_labels = []
    results = {}
    loss = 0
    criterion = torch.nn.BCELoss()
    wrapped_model = ModelWrapper(model,criterion)

    n_samples = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_multi_preds = wrapped_model.predict_on_batch(x, iterations=num_trials)
            y_preds = y_multi_preds.mean(dim=-1)
            y_errors = y_multi_preds.std(dim=-1)
            n = y.shape[0]
            loss += criterion(y_preds.reshape(-1), y)*n
            n_samples+=n
            all_preds.extend(y_preds.to('cpu').numpy())
            all_labels.extend(y.to('cpu').numpy())

    results = compute_metrics(all_labels, all_preds)
    results["loss"] = loss.to('cpu').item() / n_samples
    return results

def concat_features(row):
    return np.concatenate([row[f"features_right"], row[f"features_left"]])

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params

'''
/localdisk1/PARK/colearning/code/unimodal_models/finger_tapping/unimodal_finger_baal.py 
--batch_size=256 --corr_thr=0.95 --drop_correlated=no --dropout_prob=0.24180259124462203 
--gamma=0.683941938387959 --hand=both --learning_rate=0.3081959128766984 --minority_oversample=no 
--model=ShallowANN --momentum=0.9206317439937552 --num_buckets=10 --num_epochs=87 --num_trials=1000 
--optimizer=SGD --patience=13 --random_state=526 --scaling_method=StandardScaler --scheduler=reduce 
--seed=790 --step_size=18 --use_feature_scaling=yes --use_scheduler=yes
'''
@click.command()
@click.option("--model", default="ShallowANN", help="Options: ANN, ShallowANN")
@click.option("--dropout_prob", default=0.24180259124462203)
@click.option("--num_trials", default=1000, help="Options: 100, 500, 1000, 5000, 10000, 50000")
@click.option("--num_buckets", default=20, help="Options: 5, 10, 20, 50, 100")
@click.option("--hand", default="both", help="Options: left, right, both")
@click.option("--learning_rate", default=0.3081959128766984, help="Learning rate for classifier")
@click.option("--random_state", default=526, help="Random state for classifier")
@click.option("--seed", default=790, help="Seed for random")
@click.option("--use_feature_scaling",default='yes',help="yes if you want to scale the features, no otherwise")
@click.option("--scaling_method",default='StandardScaler',help="Options: StandardScaler, MinMaxScaler")
@click.option("--minority_oversample",default='no',help="Options: 'yes', 'no'")
@click.option("--batch_size",default=256)
@click.option("--num_epochs",default=87)
@click.option("--drop_correlated",default='no',help="Options: yes, no")
@click.option("--corr_thr",default=0.95)
@click.option("--optimizer",default="SGD",help="Options: SGD, AdamW")
@click.option("--beta1",default=0.9)
@click.option("--beta2",default=0.999)
@click.option("--weight_decay",default=0.0001)
@click.option("--momentum",default=0.9206317439937552)
@click.option("--use_scheduler",default='yes',help="Options: yes, no")
@click.option("--scheduler",default='reduce',help="Options: step, reduce")
@click.option("--step_size",default=18)
@click.option("--gamma",default=0.683941938387959)
@click.option("--patience",default=13)
def main(**cfg):
    ENABLE_WANDB = False
    if ENABLE_WANDB:
        wandb.init(project="park_final_experiments", config=cfg)

    '''
    Ensure reproducibility of randomness
    '''
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    oversample = SMOTE(random_state = cfg['random_state'])
    criterion = torch.nn.BCELoss()

    if cfg["drop_correlated"]=='no':
        drop_correlated = False
    else:
        drop_correlated = True

    if cfg["hand"] in ['left', 'right']:
        features, labels, ids, _, columns = load(drop_correlated=drop_correlated, corr_thr=cfg["corr_thr"], hand=cfg["hand"])
    else:
        features_right, labels_right, ids_right, id_dates_right, columns = load(drop_correlated=drop_correlated, corr_thr=cfg["corr_thr"], hand="right")
        features_left, labels_left, ids_left, id_dates_left, columns = load(drop_correlated=drop_correlated, corr_thr=cfg["corr_thr"], hand="left")
        #print(len(id_dates_left), len(set(id_dates_left)), len(id_dates_right), len(set(id_dates_right)), len(set(id_dates_left).intersection(set(id_dates_right))))
        
        df_right = pd.DataFrame.from_dict({"features_right":list(features_right), "id_right":ids_right, "row_id":id_dates_right, "label_right":labels_right})
        df_left = pd.DataFrame.from_dict({"features_left":list(features_left), "id_left":ids_left, "row_id":id_dates_left, "label_left":labels_left})
        
        df_both = pd.merge(df_right, df_left, how="inner", on="row_id")
        df_both = df_both.drop(columns=['label_left', 'id_left'])
        df_both = df_both.rename(columns={"label_right":"label", "id_right":"id"})
        df_both["features"] = df_both.apply(concat_features, axis=1)
        
        features = df_both.loc[:, "features"]
        labels = df_both.loc[:, "label"]
        ids = df_both.loc[:, "id"]

    '''
    Train+dev and test split
    '''
    features_train, labels_train, ids_train, features_test, labels_test, ids_test = train_test_split(features, labels, ids)

    print(f"Number of unique ids in training and test sets: {len(set(ids_train))}, {len(set(ids_test))}")
    assert len(set(ids_train).intersection(set(ids_test))) == 0

    '''
    Train-dev split (random)
    '''
    features_train, labels_train, ids_train, features_dev, labels_dev, ids_dev = train_dev_split(features_train, labels_train, ids_train)

    X_train, X_dev, X_test = features_train, features_dev, features_test
    y_train, y_dev, y_test = labels_train, labels_dev, labels_test
    
    used_scaler = None

    if cfg['use_feature_scaling']=='yes':
        if cfg['scaling_method'] == 'StandardScaler':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)
        pickle.dump(scaler, open(SCALER_PATH,"wb"))
        used_scaler = pickle.load(open(SCALER_PATH,'rb'))
        
    if cfg['minority_oversample']=='yes':
        (X_train, y_train) = oversample.fit_resample(X_train, y_train)

    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    y_test = np.asarray(y_test)
        
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"])
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])
        
    model = None
    if cfg['model']=="ANN":
        model = ANN(features[0].shape[0], drop_prob=cfg["dropout_prob"])
    elif cfg['model']=="ShallowANN":
        model = ShallowANN(features[0].shape[0], drop_prob=cfg["dropout_prob"])
    else:
        raise ValueError("Invalid model")

    model = model.to(device)
    num_params = count_parameters(model)
    print(f'Number of parameters: {num_params}')
 
if __name__ == "__main__":
    main()