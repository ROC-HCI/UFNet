import os
import copy
import pickle
import torch
import pandas as pd
from pandas import DataFrame
 
from tqdm import tqdm
 
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
 
from mlxtend.plotting import plot_confusion_matrix
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import subprocess as sp
 
import wandb
import random
import click
import json
import imblearn
import re
from imblearn.over_sampling import SMOTE
 
from torch import nn
from torch.utils.data import Dataset, DataLoader
 
from constants import *

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
Load patient ids in dev and test sets
'''
with open(os.path.join(BASE_DIR,"data/dev_set_participants.txt")) as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])

with open(os.path.join(BASE_DIR,"data/test_set_participants.txt")) as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])

'''
Features, Labels: CSV to Dataframe
'''
def load(video_model):
    df = pd.read_csv(FEATURES_FILE)
    print(f"length of dataset: {len(df)}")

    if video_model=="VideoMAE":
        video_embedding_file = "/localdisk1/PARK/park_vlm/VideoEmbeddings/VideoMAE/VideoMAE_Features.pkl"
        feature_len = 768
    elif video_model=="TimeSformer":
        video_embedding_file = "/localdisk1/PARK/park_vlm/VideoEmbeddings/TimeSformer/TimeSformer_Features.pkl"
        feature_len = 768
    elif video_model=="ViViT":
        video_embedding_file = "/localdisk1/PARK/park_vlm/VideoEmbeddings/ViViT/ViViT_Features.pkl"
        feature_len = 768
    elif video_model == "Uniformer":
        video_embedding_file = "/localdisk1/PARK/park_vlm/VideoEmbeddings/Uniformer/Uniformer_Features.pkl"
        feature_len = 512
    else:
        assert False, "Unsupported video model"

    with open(video_embedding_file, 'rb') as f:
        loaded_data = pickle.load(f)

    embedding_vectors = {}
    for item in loaded_data:
        key = item["filename"]
        embedding = item["mean_pooled_embedding"]
        embedding_vectors[key] = embedding.cpu()
    
    filenames = df["Filename"]
    '''
    Labels are not often consistent across datasets.
    Typically, "yes", "Possible", "Probable" are considered positive.
    "no", and "Unlikely" are considered negative.
    '''
    labels = df['pd'].apply(lambda x: 0 if str(x) in ['no','0'] else 1).to_numpy()
    '''
    IDs should be just the patient ids excluding dates and filenames/extensions.
    Task-wise feature files are processed differently. Please cautiously check.
    '''
    IDs = df['ID']

    dataset = []
    
    for filename, label, pid in zip(filenames, labels, IDs):
        if filename in embedding_vectors.keys():
            features = np.asarray(embedding_vectors[filename])
            dataset.append({"filename": filename, "label": label, "pid": pid, "features": features})

    dataset_df = pd.DataFrame.from_dict(dataset)
    print(f"Length of datasets with embeddings: {len(dataset_df)}")
    feature_columns = [f'x{i}' for i in range(feature_len)]

    features = np.asarray(list(dataset_df["features"]))
    
    return features, dataset_df["label"], dataset_df["pid"], feature_columns

#print(len(set(test_ids)), len(set(dev_ids)))

'''
Based on the predefined test split, split the dataframe into train+dev and test sets
Note: this train set will further be split into train/dev
So, this is basically train+dev; and test splits
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
Randomly split the train set into train and validation
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
ML baselines using pytorch
'''
class ANN(nn.Module):
    def __init__(self, n_features):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1,bias=True)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.hidden_activation(self.fc1(x))
        y = self.fc2(x1)
        y = self.sig(y)
        return y

'''
ML baselines using pytorch
'''
class ShallowANN(nn.Module):
    def __init__(self, n_features):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1,bias=True)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y = self.fc(x)
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

'''
Evaluate performance on validation/test set.
Returns all the metrics defined above and the loss.
'''
def evaluate(model, dataloader):
    all_preds = []
    all_labels = []
    results = {}

    loss = 0
    criterion = torch.nn.BCELoss()

    n_samples = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_preds = model(x)
            n = y.shape[0]
            loss += criterion(y_preds.reshape(-1), y)*n
            n_samples+=n
            all_preds.extend(y_preds.to('cpu').numpy())
            all_labels.extend(y.to('cpu').numpy())

    results = compute_metrics(all_labels, all_preds)
    results["loss"] = loss.to('cpu').item() / n_samples

    return results

@click.command()
@click.option("--model", default="ShallowANN", help="Options: ANN, ShallowANN")
@click.option("--learning_rate", default=0.8365099039036598, help="Learning rate for classifier")
@click.option("--random_state", default=201, help="Random state for classifier")
@click.option("--seed", default=488, help="Seed for random")
@click.option("--batch_size",default=1024)
@click.option("--num_epochs",default=74)
@click.option("--optimizer",default="SGD",help="Options: SGD, AdamW")
@click.option("--beta1",default=0.9)
@click.option("--beta2",default=0.999)
@click.option("--weight_decay",default=0.0001)
@click.option("--momentum",default=0.615229008837764)
@click.option("--use_scheduler",default='yes',help="Options: yes, no")
@click.option("--scheduler",default='reduce',help="Options: step, reduce")
@click.option("--step_size",default=7)
@click.option("--gamma",default=0.7908197575913161 )
@click.option("--patience",default=4)
@click.option("--video_model", default='TimeSformer', help="Options: VideoMAE, TimeSformer, Uniformer, ViViT")
def main(**cfg):
    ENABLE_WANDB = False
    if ENABLE_WANDB:
        wandb.init(project="park_rebuttal_experiments", config=cfg)

    '''
    save the configurations obtained from wandb (or command line) into the model config file
    '''
    # with open(MODEL_CONFIG_PATH,"w") as f:
    #     f.write(json.dumps(cfg))
        
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
        
    features, labels, ids, columns = load(cfg["video_model"])
    print("All folds: ")
    print(f"Number of data: {len(labels)}, PD: {np.sum(labels==1)} ({(np.sum(labels==1)*100)/len(labels)}%)")
    print(f"Number of unique patients: {len(set(ids))}, PD: {len(set(ids[labels==1]))} ({(len(set(ids[labels==1]))*100)/len(set(ids))}%)")
    '''
    Train+dev and test splits
    '''
    features_train, labels_train, ids_train, features_test, labels_test, ids_test = train_test_split(features, labels, ids)
    
    print(f"Number of unique ids in training and test sets: {len(set(ids_train))}, {len(set(ids_test))}")
    assert len(set(ids_train).intersection(set(ids_test))) == 0
    print("There is no overlap between train and test sets")
    
    '''
    Train-dev split (with randomly predefined dev set ids)
    '''
    features_train, labels_train, ids_train, features_dev, labels_dev, ids_dev = train_dev_split(features_train, labels_train, ids_train)

    print("Train set: ")
    labels = np.asarray(labels_train)
    ids = np.asarray(ids_train)
    print(f"Number of data: {len(labels)}, PD: {np.sum(labels==1)} ({(np.sum(labels==1)*100)/len(labels)}%)")
    print(f"Number of unique patients: {len(set(ids))}, PD: {len(set(ids[labels==1]))} ({(len(set(ids[labels==1]))*100)/len(set(ids))}%)")

    print("Validation set: ")
    labels = np.asarray(labels_dev)
    ids = np.asarray(ids_dev)
    print(f"Number of data: {len(labels)}, PD: {np.sum(labels==1)} ({(np.sum(labels==1)*100)/len(labels)}%)")
    print(f"Number of unique patients: {len(set(ids))}, PD: {len(set(ids[labels==1]))} ({(len(set(ids[labels==1]))*100)/len(set(ids))}%)")
    
    print("Test set: ")
    labels = np.asarray(labels_test)
    ids = np.asarray(ids_test)
    print(f"Number of data: {len(labels)}, PD: {np.sum(labels==1)} ({(np.sum(labels==1)*100)/len(labels)}%)")
    print(f"Number of unique patients: {len(set(ids))}, PD: {len(set(ids[labels==1]))} ({(len(set(ids[labels==1]))*100)/len(set(ids))}%)")
    
    X_train, X_dev, X_test = features_train, features_dev, features_test
    y_train, y_dev, y_test = labels_train, labels_dev, labels_test

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
        model = ANN(features.shape[1])
    elif cfg['model']=="ShallowANN":
        model = ShallowANN(features.shape[1])
    else:
        raise ValueError("Invalid model")

    model = model.to(device)
    
    if cfg["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(cfg['beta1'],cfg['beta2']),weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Invalid optimizer")    
    
    if cfg["use_scheduler"]=="yes":
        if cfg['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        elif cfg['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['gamma'], patience = cfg['patience'])
        else:
            raise ValueError("Invalid scheduler")
        
    best_dev_loss = np.finfo(features.dtype).max
    best_dev_accuracy = 0.0
    best_dev_balanced_accuracy = 0.0
    best_dev_auroc = 0.0
    best_dev_f1 = 0.0
    best_model = copy.deepcopy(model)
    
    for epoch in tqdm(range(cfg['num_epochs'])):
        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_preds = model(x)
            l = criterion(y_preds.reshape(-1),y)
            l.backward()
            optimizer.step()
            
            if ENABLE_WANDB:
                wandb.log({"train_loss": l.to('cpu').item()})
        
        dev_metrics = evaluate(model, dev_loader)
        dev_loss = dev_metrics['loss']
        dev_accuracy = dev_metrics['accuracy']
        dev_balanced_accuracy = dev_metrics['weighted_accuracy']
        dev_auroc = dev_metrics['auroc']
        dev_f1 = dev_metrics['f1_score']

        if cfg['use_scheduler']=="yes":
            if cfg['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_loss)
            
        if dev_loss < best_dev_loss:
             best_model = copy.deepcopy(model)
             best_dev_loss = dev_loss
             best_dev_accuracy = dev_accuracy
             best_dev_balanced_accuracy = dev_balanced_accuracy
             best_dev_auroc = dev_auroc
             best_dev_f1 = dev_f1
             

    results = evaluate(best_model, test_loader)
    if ENABLE_WANDB:
        wandb.log(results)
        wandb.log({"dev_accuracy":best_dev_accuracy, "dev_balanced_accuracy":best_dev_balanced_accuracy, "dev_loss":best_dev_loss, "dev_auroc":best_dev_auroc, "dev_f1":best_dev_f1})

    print(results)


    # '''
    # Save best model
    # '''
    # torch.save(best_model.to('cpu').state_dict(),MODEL_PATH)

    # '''
    # Test whether the model can be loaded successfully
    # '''
    # if cfg['model']=="ShallowANN":
    #     loaded_model = ShallowANN(features.shape[1])
    # elif cfg['model']=="ANN":
    #     loaded_model = ANN(features.shape[1])
    # loaded_model.load_state_dict(torch.load(MODEL_PATH))
    # loaded_model = loaded_model.to(device)
    # print(evaluate(loaded_model,test_loader))
    # print(cfg)
    # print(loaded_model)
    
    
if __name__ == "__main__":
    main()