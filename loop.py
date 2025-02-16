"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from utils import MLMetrics
from tqdm import tqdm
from model import SmrtNet
from utils import fix_seed

import warnings
warnings.filterwarnings("ignore")

def model_trained(path_dir=None, args=None):
    model = SmrtNet(args).cuda()
    model.load_state_dict(torch.load(path_dir))
    return model

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    model.train()
    trian_pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}')
    t_met = MLMetrics(objective='binary')
    train_losses_in_epoch=[]
    for train_batch_idx, train_data in enumerate(trian_pbar):
        train_ds, train_de, train_rs, train_re, train_y = train_data[1]
        train_ds, train_rs= train_ds.to(device), train_rs.float().to(device)
        train_y = Variable(torch.from_numpy(np.array(train_y)).float()).to(device)
        optimizer.zero_grad()
        de_idx = torch.tensor(train_de['input_ids']).to(device)
        de_mask = torch.tensor(train_de['attention_mask']).to(device)
        train_re = {k: v.to(device) for k, v in train_re.items() if k in ['input_ids', 'attention_mask']}
        re_atten_mask = train_re['attention_mask']
        re_input_ids = train_re['input_ids']
        train_output = model(train_ds, de_idx, de_mask, train_rs, re_input_ids, re_atten_mask)
        train_prob = torch.sigmoid(train_output)
        train_output = torch.squeeze(train_output, 1)
        train_loss = criterion(train_output, train_y)
        label = train_y.to(device='cpu', dtype=torch.long).detach().numpy()
        prob = train_prob.to(device='cpu').detach().numpy()
        t_met.update(label, prob, [train_loss.item()])
        train_losses_in_epoch.append(train_loss.item())
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        trian_pbar.set_description(f'Epoch [{epoch}/{args.epoch}]')
        trian_pbar.set_postfix(acc=t_met.acc, pre=t_met.pre, rec=t_met.rec, f1=t_met.f1, los=train_loss.item(), auc=t_met.auc, prc=t_met.prc)
    train_loss_a_epoch = np.average(train_losses_in_epoch)
    return t_met, train_loss_a_epoch

def valid(args, model, device, valid_loader, criterion, epoch):
    model.eval()
    v_met = MLMetrics(objective='binary')
    y_all = []
    p_all = []
    l_all = []
    valid_losses_in_epoch = []
    with torch.no_grad():
        for valid_batch_idx, (valid_ds, valid_de, valid_rs, valid_re, valid_y) in enumerate(valid_loader):
            '''data preparation '''
            valid_ds, valid_rs= valid_ds.to(device), valid_rs.float().to(device)
            valid_y = Variable(torch.from_numpy(np.array(valid_y)).float()).to(device)
            de_idx = torch.tensor(valid_de['input_ids']).to(device)
            de_mask = torch.tensor(valid_de['attention_mask']).to(device)
            valid_re = {k: v.to(device) for k, v in valid_re.items() if k in ['input_ids', 'attention_mask']}
            re_atten_mask = valid_re['attention_mask']
            re_input_ids = valid_re['input_ids']
            valid_output = model(valid_ds, de_idx, de_mask, valid_rs, re_input_ids, re_atten_mask)
            valid_prob = torch.sigmoid(valid_output)
            valid_output = torch.squeeze(valid_output, 1)
            valid_loss = criterion(valid_output, valid_y)
            label = valid_y.to(device='cpu', dtype=torch.long).detach().numpy()
            prob = valid_prob.to(device='cpu').detach().numpy()
            y_all.append(label)
            p_all.append(prob)
            l_all.append(valid_loss.item())
            valid_losses_in_epoch.append(valid_loss.item())
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)
    v_met.update(y_all, p_all, [l_all.mean()])
    valid_loss_a_epoch = np.average(valid_losses_in_epoch)
    return v_met, valid_loss_a_epoch, y_all, p_all

def test(args, model, device, valid_loader, criterion):
    model.eval()
    v_met = MLMetrics(objective='binary')
    y_all = []
    p_all = []
    l_all = []
    valid_losses_in_epoch = []

    with torch.no_grad():
        for valid_batch_idx, (valid_ds, valid_de, valid_rs, valid_re, valid_y) in enumerate(valid_loader):
            fix_seed(args.seed)
            valid_ds, valid_rs= valid_ds.to(device), valid_rs.float().to(device)
            valid_y = Variable(torch.from_numpy(np.array(valid_y)).float()).to(device)
            de_idx = torch.tensor(valid_de['input_ids']).to(device)
            de_mask = torch.tensor(valid_de['attention_mask']).to(device)
            valid_re = {k: v.to(device) for k, v in valid_re.items() if k in ['input_ids', 'attention_mask']}
            re_atten_mask = valid_re['attention_mask']
            re_input_ids = valid_re['input_ids']
            valid_output = model(valid_ds, de_idx, de_mask, valid_rs, re_input_ids, re_atten_mask)
            valid_prob = torch.sigmoid(valid_output)
            valid_output = torch.squeeze(valid_output, 1)
            valid_loss = criterion(valid_output, valid_y)
            label = valid_y.to(device='cpu', dtype=torch.long).detach().numpy()
            prob = valid_prob.to(device='cpu').detach().numpy()
            y_all.append(label)
            p_all.append(prob)
            l_all.append(valid_loss.item())
            valid_losses_in_epoch.append(valid_loss.item())
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)
    v_met.update(y_all, p_all, [l_all.mean()])
    valid_loss_a_epoch = np.average(valid_losses_in_epoch)
    return v_met, valid_loss_a_epoch, y_all, p_all



def bench(args, model, device, valid_loader):
    valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}')
    model.eval()
    y_all = []
    p_all = []
    with torch.no_grad():
        for valid_batch_idx, valid_data in enumerate(valid_pbar):
            fix_seed(args.seed)
            valid_ds, valid_de, valid_rs, valid_re, valid_y = valid_data[1]
            valid_ds, valid_rs= valid_ds.to(device), valid_rs.float().to(device)
            valid_y = Variable(torch.from_numpy(np.array(valid_y)).float()).to(device)
            de_idx = torch.tensor(valid_de['input_ids']).to(device)
            de_mask = torch.tensor(valid_de['attention_mask']).to(device)
            valid_re = {k: v.to(device) for k, v in valid_re.items() if k in ['input_ids', 'attention_mask']}
            re_atten_mask = valid_re['attention_mask']
            re_input_ids = valid_re['input_ids']
            valid_output = model(valid_ds, de_idx, de_mask, valid_rs, re_input_ids, re_atten_mask)
            valid_prob = torch.sigmoid(valid_output)
            valid_output = torch.squeeze(valid_output, 1)
            label = valid_y.to(device='cpu', dtype=torch.long).detach().numpy()
            prob = valid_prob.to(device='cpu').detach().numpy()
            y_all.append(label)
            p_all.append(prob)
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    return y_all, p_all