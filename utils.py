"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import os, random
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import torch, dgl, pickle
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from LM_Mol.tokenizer import MolTranBertTokenizer



def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    #if seed is None:
    #    seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dict(path, obj):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir


def dgl_collate_func_ds(x):
    tokenizer = MolTranBertTokenizer('./LM_Mol/bert_vocab.txt')
    d, ds, p, e, y = zip(*x)
    #print(d)
    d = dgl.batch(d)
    #breakpoint()
    ds = tokenizer.batch_encode_plus([''.join(smile) for smile in ds], padding=True, add_special_tokens=True)
    p = np.asarray(p)
    e = tailor_batch(e)
    y = np.asarray(y)
    return d, ds, torch.tensor(p), e, torch.tensor(y)


def prepend_pad(input_np, target_len, cls_tk, pad_tk):
    np_final = np.full(target_len, pad_tk)
    np_final[0] = cls_tk
    sz = len(input_np)
    np_final[1:sz+1] = input_np
    # if sz + 1 == target_len:
    #     return np_final
    # elif sz + 1 < target_len:
    #     return np.pad(np_final, (0, target_len - sz - 1), mode='constant', constant_values=pad_tk)
    # elif sz + 1 > target_len:
    #     print('len cuowu')
    return np_final

# dataloader collate_fn, tailor the data and prepare the input dict

def tailor_batch(batch_data):
    dict_base_id = {
    'A':0,
    'U':1,
    'T':1,
    'C':2,
    'G':3,
    'N':4,
    'Y':5,
    'R':6,
    'S':7,
    'K':8,
    'W':9,
    'M':10,
    'D':11,
    'H':12,
    'V':13,
    'B':14,
    'I':15,
    '-':16,
    'MASK':17,
    'CLS':18,
    'PAD':19
    }
    batch_data_new = {}
    sz_list = []
    atten_list = []
    unmasked_id_list = []
    for x in batch_data:
        seq = x.strip()
        sz = len(seq)
        sz_list.append(sz+1)
    max_seq_len = max(sz_list)
    for x in batch_data:
        seq = x.strip()
        sz = len(seq)
        atten_list.append(torch.from_numpy(prepend_pad(np.array([True]*sz), max_seq_len, True, False)))
        unmasked_id_list.append(torch.from_numpy(prepend_pad(np.array(list(map(lambda x:dict_base_id[x], seq))), max_seq_len, dict_base_id['CLS'], dict_base_id['PAD'])))

    batch_data_new['input_ids'] = torch.cat([x1.unsqueeze(0)  for x1 in unmasked_id_list], 0)
    batch_data_new['seq_len'] = torch.cat([torch.tensor(x1).unsqueeze(0)  for x1 in sz_list], 0)
    batch_data_new['attention_mask'] = (torch.cat([x1.unsqueeze(0)  for x1 in atten_list], 0)).float()
    return batch_data_new


__all__ = ["accuracy", "precision", "recall", "roc", "pr", "calculate_metrics"]

from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


class MLMetrics(object):
    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst)>0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg()

    def compute_avg(self):
        if len(self.metrics)>1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.pre = self.avg[1]
        self.rec = self.avg[2]
        self.f1  = self.avg[3]
        self.auc = self.avg[4]
        self.prc = self.avg[5]
        self.tp  = int(self.sum[6])
        self.tn  = int(self.sum[7])
        self.fp  = int(self.sum[8])
        self.fn  = int(self.sum[9])
        if len(self.avg)>9:
            self.other = self.avg[10:]


def accuracy_cal(label, prediction):
    metric = np.array(accuracy_score(label, np.round(prediction)))
    return metric

def precision_cal(label, prediction):

    metric = np.array(precision_score(label, np.round(prediction)))
    #print("precision")
    #print(metric)
    return metric

def recall_cal(label, prediction):

    metric = np.array(recall_score(label, np.round(prediction)))
    #print("recall")
    #print(metric)
    return metric

def f1_cal(label, prediction):
    metric = np.array(accuracy_score(label, np.round(prediction)))
    return metric

def roc_cal(label, prediction):
    fpr, tpr, thresholds = roc_curve(label, prediction)
    score = auc(fpr, tpr)
    metric = np.array(score)
    curves = [(fpr, tpr)]
    return metric, curves


def pr_cal(label, prediction):
    precision, recall, thresholds = precision_recall_curve(label, prediction)
    score = auc(recall, precision)
    metric = np.array(score)
    curves = [(precision, recall)]
    return metric, curves

def tfnp_cal(label, prediction):
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn =0,0,0,0
    return tp, tn, fp, fn


def calculate_metrics(label, prediction, objective):
    """calculate metrics for classification"""
    if (objective == "binary") :

        accuracy = accuracy_cal(label, prediction)
        precision = precision_cal(label, prediction)
        recall = recall_cal(label, prediction)
        f1 = f1_cal(label, prediction)
        auc_roc, roc_curves = roc_cal(label, prediction)
        auc_pr, pr_curves = pr_cal(label, prediction)

        pred_class = prediction>0.5
        tp, tn, fp, fn = tfnp_cal(label, pred_class)

        mean = [np.nanmean(accuracy), np.nanmean(precision), np.nanmean(recall), np.nanmean(f1), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn]
        std =  [np.nanstd(accuracy),  np.nanstd(precision),  np.nanstd(recall),  np.nanstd(f1),  np.nanstd(auc_roc),  np.nanstd(auc_pr)]

    return [mean, std]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)




def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    dataset = shuffle(dataset)
    return dataset


def get_kfold_data(i, datasets, k=5, v=1,random_state=42):
    random.seed(random_state)
    #np.random.seed(42)
    v = v * 10
    fold_size = len(datasets) // k
    #print(len(datasets), i, k, fold_size)

    test_start = i * fold_size
    #print(test_start)
    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        TestSet = datasets[test_start:test_end]
        TVset = pd.concat([datasets[0:test_start], datasets[test_end:]])
        Val_size = len(TVset) // v
        ValSet = TVset.sample(n=Val_size)
        TVset_all = TVset.append(ValSet)
        TrainSet =TVset_all.drop_duplicates(keep=False)


    elif i == 0:
        test_end = fold_size
        TestSet = datasets[test_start:test_end]
        TVset = datasets[test_end:]
        Val_size=len(TVset) // v
        ValSet = TVset.sample(n=Val_size)
        TVset_all = TVset.append(ValSet)
        TrainSet =TVset_all.drop_duplicates(keep=False)


    else:
        TestSet = datasets[test_start:]
        TVset = datasets[0:test_start]
        Val_size = len(TVset) // v
        ValSet = TVset.sample(n=Val_size)
        TVset_all = TVset.append(ValSet)
        TrainSet =TVset_all.drop_duplicates(keep=False)

    return TrainSet.reset_index(drop=True), ValSet.reset_index(drop=True), TestSet.reset_index(drop=True)



def get_kfold_data_target(i, df, k=5, v=1,random_state=42):
    random.seed(random_state)

    v = v * 10

    df['Sequence_structure']=df['Sequence']+df['Structure']
    #print(df)
    Sequence_structure = shuffle_dataset(df['Sequence_structure'].unique(),random_state)
    #print(len(df['Sequence_structure'].unique()))


    fold_size = len(Sequence_structure) // k

    test_start = i * fold_size

    if i != k - 1 and i != 0:

        test_end = (i + 1) * fold_size


        TestSet = df.loc[df['Sequence_structure'].isin(Sequence_structure[test_start:test_end])]
        TVSet = df.loc[~df['Sequence_structure'].isin(Sequence_structure[test_start:test_end])]

        TVSet_SMILES = TVSet['Sequence_structure'].unique()
        Val_size = len(TVSet_SMILES)//v
        #print(list(TVSet_SMILES))
        Val_SMILES= sample(list(TVSet_SMILES),Val_size)
        Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(Val_SMILES)))
        ValSet = TVSet.loc[TVSet['Sequence_structure'].isin(Val_SMILES)]
        #print(len(ValSet))
        TrainSet = TVSet.loc[TVSet['Sequence_structure'].isin(Train_SMILES)]

    elif i == 0:
        test_end = fold_size

        TestSet  = df.loc[df['Sequence_structure'].isin(Sequence_structure[test_start:test_end])]
        TVSet = df.loc[~df['Sequence_structure'].isin(Sequence_structure[test_start:test_end])]
        TVSet_SMILES = TVSet['Sequence_structure'].unique()
        Val_size = len(TVSet_SMILES)//v
        Val_SMILES= sample(list(TVSet_SMILES),Val_size)
        #print(Val_SMILES)
        Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(Val_SMILES)))
        ValSet = TVSet.loc[TVSet['Sequence_structure'].isin(Val_SMILES)]
        TrainSet = TVSet.loc[TVSet['Sequence_structure'].isin(Train_SMILES)]

    else:

        TestSet  = df.loc[df['Sequence_structure'].isin(Sequence_structure[test_start:])]
        TVSet = df.loc[~df['Sequence_structure'].isin(Sequence_structure[test_start:])]
        TVSet_SMILES = TVSet['Sequence_structure'].unique()
        Val_size = len(TVSet_SMILES)//v
        Val_SMILES= sample(list(TVSet_SMILES),Val_size)
        Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(Val_SMILES)))
        ValSet = TVSet.loc[TVSet['Sequence_structure'].isin(Val_SMILES)]
        TrainSet = TVSet.loc[TVSet['Sequence_structure'].isin(Train_SMILES)]

    return TrainSet.reset_index(drop=True), ValSet.reset_index(drop=True), TestSet.reset_index(drop=True)



def get_kfold_data_drug_fix(i, df, k=5, v=1,random_state=42):
    random.seed(random_state)
    test_cutoff=0.15
    valid_cutoff=0.09
    shift=0.01

    v = v * 10
    #print(len(df))
    SMILES = shuffle_dataset(df['SMILES'].unique(),random_state)


    fold_size = len(SMILES) // k

    test_start = i * fold_size

    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size

        TestSet = df.loc[df['SMILES'].isin(SMILES[test_start:test_end])]

        if (len(TestSet)<len(df)*(test_cutoff-shift)):
            i=1
            #print( len(TestSet),len(df)*(test_cutoff-shift))
            while len(TestSet)<len(df)*(test_cutoff-shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start:test_end+i])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end+i])]
                i += 1
                if(i>=100):
                    break
        elif (len(TestSet)>len(df)*(test_cutoff+shift)):
            i=1

            while len(TestSet)>len(df)*(test_cutoff+shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start:test_end - i])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end - i])]
                i += 1
        else:
            TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end])]

        TVSet_SMILES = shuffle_dataset(TVSet['SMILES'].unique(), random_state)
        Val_size = len(TVSet_SMILES)//v
        ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size])]


        if (len(ValSet) < len(df) * (valid_cutoff-shift)):
            j=1
            while len(ValSet) < len(df) * (valid_cutoff-shift):
                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size+j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size+j-1])))
        elif (len(ValSet) > len(df) * (valid_cutoff+shift)):
            j=1
            while len(ValSet) > len(df) * (valid_cutoff+shift):
                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size-j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size-j+1])))
        else:
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size])))

        TrainSet = TVSet.loc[TVSet['SMILES'].isin(Train_SMILES)]

    elif i == 0:
        test_end = fold_size
        TestSet  = df.loc[df['SMILES'].isin(SMILES[test_start:test_end])]

        if (len(TestSet)<len(df)*(test_cutoff-shift)):
            i=1
            while len(TestSet)<len(df)*(test_cutoff-shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start:test_end+i])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end+i])]
                i += 1
        elif (len(TestSet)>len(df)*(test_cutoff+shift)):
            i=1
            while len(TestSet)>len(df)*(test_cutoff+shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start:test_end - i])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end - i])]
                i += 1
        else:
            TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:test_end])]

        TVSet_SMILES = shuffle_dataset(TVSet['SMILES'].unique(), random_state)
        Val_size = len(TVSet_SMILES)//v
        ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size])]


        if (len(ValSet) < len(df) * (valid_cutoff-shift)):
            j=1
            while len(ValSet) < len(df) * (valid_cutoff-shift):

                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size+j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size+j-1])))
        elif (len(ValSet) > len(df) * (valid_cutoff+shift)):
            j=1
            while len(ValSet) > len(df) * (valid_cutoff+shift):
                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size-j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size-j+1])))
        else:
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size])))

        TrainSet = TVSet.loc[TVSet['SMILES'].isin(Train_SMILES)]


    else:
        TestSet  = df.loc[df['SMILES'].isin(SMILES[test_start:])]
        if (len(TestSet)<len(df)*(test_cutoff-shift)):
            i=1
            while len(TestSet)<len(df)*(test_cutoff-shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start-i:])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start-i:])]
                i+=1
        elif (len(TestSet)>len(df)*(test_cutoff+shift)):
            i=1
            while len(TestSet)>len(df)*(test_cutoff+shift):
                TestSet = df.loc[df['SMILES'].isin(SMILES[test_start+i:])]
                TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start+i:])]
                i += 1
        else:
            TVSet = df.loc[~df['SMILES'].isin(SMILES[test_start:])]

        TVSet_SMILES = shuffle_dataset(TVSet['SMILES'].unique(), random_state)
        Val_size = len(TVSet_SMILES)//v
        ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size])]


        if (len(ValSet) < len(df) * (valid_cutoff-shift)):
            j=1
            while len(ValSet) < len(df) * (valid_cutoff-shift):
                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size+j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size+j-1])))
        elif (len(ValSet) > len(df) * (valid_cutoff+shift)):
            j=1
            while len(ValSet) > len(df) * (valid_cutoff+shift):
                ValSet = TVSet.loc[TVSet['SMILES'].isin(TVSet_SMILES[0:Val_size-j])]
                j+=1
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size-j+1])))
        else:
            Train_SMILES = list(set(list(TVSet_SMILES)).difference(set(TVSet_SMILES[0:Val_size])))

        TrainSet = TVSet.loc[TVSet['SMILES'].isin(Train_SMILES)]



    return TrainSet.reset_index(drop=True), ValSet.reset_index(drop=True), TestSet.reset_index(drop=True)


def get_kfold_data_drug_fix_best_final(i, df, k=5, v=1,random_state=42):
    with open('./dataset_cv_best/smi_train'+str(i+1)+'.txt', 'r') as f:
        Train_SMILES = f.read().splitlines()
    with open('./dataset_cv_best/smi_valid'+str(i+1)+'.txt', 'r') as f:
        Valid_SMILES = f.read().splitlines()
    with open('./dataset_cv_best/smi_test'+str(i+1)+'.txt', 'r') as f:
        Test_SMILES = f.read().splitlines()

    if(i==0):
        TrainSet = df.loc[df['SMILES'].isin(Train_SMILES)]
        ValSet = df.loc[df['SMILES'].isin(Valid_SMILES)]
        TestSet = df.loc[df['SMILES'].isin(Test_SMILES)]
    elif(i==1):
        TrainSet = df.loc[df['SMILES'].isin(Train_SMILES)]
        ValSet = df.loc[df['SMILES'].isin(Valid_SMILES)]
        TestSet = df.loc[df['SMILES'].isin(Test_SMILES)]
    elif(i == 2):
        TrainSet = df.loc[df['SMILES'].isin(Train_SMILES)]
        ValSet = df.loc[df['SMILES'].isin(Valid_SMILES)]
        TestSet = df.loc[df['SMILES'].isin(Test_SMILES)]
    elif(i == 3):
        TrainSet = df.loc[df['SMILES'].isin(Train_SMILES)]
        ValSet = df.loc[df['SMILES'].isin(Valid_SMILES)]
        TestSet = df.loc[df['SMILES'].isin(Test_SMILES)]
    elif(i == 4):
        TrainSet = df.loc[df['SMILES'].isin(Train_SMILES)]
        ValSet = df.loc[df['SMILES'].isin(Valid_SMILES)]
        TestSet = df.loc[df['SMILES'].isin(Test_SMILES)]

    return TrainSet.reset_index(drop=True), ValSet.reset_index(drop=True), TestSet.reset_index(drop=True)