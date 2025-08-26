"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import os
import torch
from torch.autograd import Variable
import pandas as pd
from rdkit import Chem
import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem.Draw import rdMolDraw2D,SimilarityMaps
from LM_Mol.tokenizer import MolTranBertTokenizer
from scipy.signal import savgol_filter
import xsmiles
import json

import matplotlib.gridspec as gridspec
from PIL import Image
ACGU_PATH = os.path.join('./img_log','acgu.npz')
CHARS = np.load(ACGU_PATH,allow_pickle=True)['data']
DB_PATH = os.path.join('./img_log','dot_bracket.npz')
DOT_BRACKET = np.uint8(np.load(DB_PATH,allow_pickle=True)['data'])



class SmoothGrad(object):
    def __init__(self, model, device='cpu', only_seq=False, train=False,
                 x_stddev=0.015, y_stddev=0.015, nsamples=20, magnitude=2):
        self.model = model
        self.device = device
        self.train = train
        self.only_seq = only_seq
        self.x_stddev = x_stddev
        self.y_stddev = y_stddev
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.features = model

    def get_gradients(self, v_d, de_idx, de_mask, v_p, re_ids, re_mask, pred_label=None, grad_channel='v_P'):
        self.model.eval()
        self.model.zero_grad()
        v_d = v_d.to(self.device)
        v_p = v_p.to(self.device)
        de_idx = de_idx.long().to(self.device)
        de_mask = de_mask.to(self.device)
        re_ids = re_ids.long().to(self.device)
        re_mask = re_mask.to(self.device)

        if grad_channel=='v_D':
            pass
        elif grad_channel=='v_De':
            de_idx.requires_grad = False
        elif grad_channel=='v_P':
            v_p.requires_grad = True
        elif grad_channel=='v_Pe':
            re_ids.requires_grad = False
        
        output,de_embeddings,re_embeddings,node_feats = self.model(v_d, de_idx, de_mask, v_p, re_ids, re_mask, get_attention=True)
        output = torch.sigmoid(output)
        output.backward()

        if grad_channel=='v_D':
            return torch.abs(node_feats.grad).sum(axis=1)
        elif grad_channel=='v_De':
            return torch.abs(de_embeddings.grad).sum(axis=2)
        elif grad_channel=='v_P':
            return v_p.grad
        elif grad_channel=='v_Pe':
            return torch.abs(re_embeddings.grad).sum(axis=2)
        else:
            exit('no grad!')

    def get_smooth_gradients(self, v_d, de_idx, de_mask, v_p, re_ids, re_mask, z=None, grad_channel='v_P'):
        return self.__call__(v_d, de_idx, de_mask, v_p, re_ids, re_mask, z, grad_channel)

    def __call__(self, v_d, de_idx, de_mask, v_p, re_ids, re_mask, z=None, grad_channel='v_P'):
        """[summary]

        Args:
            z ([type]): [description]
            y ([type]): [description]
            x_stddev (float, optional): [description]. Defaults to 0.15.
            t_stddev (float, optional): [description]. Defaults to 0.15.
            nsamples (int, optional):   [description]. Defaults to 20.
            magnitude (int, optional):  magnitude:0,1,2; 0: original gradient, 1: absolute value of the gradient,
                                        2: square value of the gradient. Defaults to 2.

        Returns:
            [type]: [description]
        """

        if grad_channel=='v_D':
            y = v_d.ndata['h']
            y_stddev = (self.y_stddev * (y.max() - y.min())).to(self.device).item()
            y_noise = torch.zeros(y.shape).to(self.device)
        elif grad_channel=='v_De':
            y = de_idx
            y_stddev = (0.1*self.y_stddev * (y.max() - y.min())).to(self.device).item()
            y_noise = torch.zeros(y.shape).to(self.device)
        elif grad_channel=='v_P':
            y = v_p
            y_stddev = (self.y_stddev * (y[:,:,:,:4].max() - y[:,:,:,:4].min())).to(self.device).item()
            y_noise = torch.zeros(y[:,:,:,:4].shape).to(self.device)
            y_stddev_struct = (self.y_stddev * (y[:,:,:,4].max() - y[:,:,:,4].min())).to(self.device).item()
            y_noise_struct = torch.zeros(y[:,:,:,4].shape).to(self.device).unsqueeze(dim=3)
        elif grad_channel=='v_Pe':
            y = re_ids
            y_stddev = (0.1*self.y_stddev * (y.max() - y.min())).to(self.device).item()
            y_noise = torch.zeros(y.shape).to(self.device)
        
        total_y_grad = torch.zeros(y.shape).to(self.device)
        for i in range(self.nsamples):
            if grad_channel=='v_D':
                total_y_grad = torch.zeros((y.shape[0])).to(self.device)
                y_plus_noise = y + y_noise.zero_().normal_(0, y_stddev)
                v_d.ndata['h'] = y_plus_noise
                y_grad = self.get_gradients(v_d, de_idx, de_mask, v_p, re_ids, re_mask, z, grad_channel)
            elif grad_channel=='v_De':
                #y_plus_noise = torch.abs(y + y_noise.zero_().normal_(0, y_stddev))
                y_plus_noise = y
                y_grad = self.get_gradients(v_d, torch.abs(y_plus_noise), de_mask, v_p, re_ids, re_mask, z, grad_channel)
            elif grad_channel=='v_P':
                y_plus_noise = y + torch.cat((y_noise.zero_().normal_(0, y_stddev),y_noise_struct.zero_().normal_(0, y_stddev_struct)),dim=3)
                y_grad = self.get_gradients(v_d, de_idx, de_mask, y_plus_noise, re_ids, re_mask, z, grad_channel)
            elif grad_channel=='v_Pe':
                #y_plus_noise = torch.abs(y + y_noise.zero_().normal_(0, y_stddev))
                y_plus_noise = y 
                y_grad = self.get_gradients(v_d, de_idx, de_mask, v_p, torch.abs(y_plus_noise), re_mask, z, grad_channel)
            
            if self.magnitude == 1:
                total_y_grad += torch.abs(y_grad)
            elif self.magnitude == 2:
                total_y_grad += y_grad * y_grad

        total_y_grad /= self.nsamples
        return total_y_grad

    def get_batch_gradients(self, v_d, de_idx, de_mask, v_p, re_ids, re_mask, Z=None, grad_channel='v_P'):
        if grad_channel=='v_D':
            Y = v_d.ndata['h']
            Y = Y.view(1,Y.shape[0],Y.shape[1])
        elif grad_channel=='v_De':
            Y = de_idx
        elif grad_channel=='v_P':
            Y = v_p
        elif grad_channel=='v_Pe':
            Y = re_ids
        else:
            Y = None

        if Y is not None:
            assert len(Y) == len(Z), "The size of input {} and target {} are not matched.".format(len(Y), len(Z))
        p_g = torch.zeros_like(Y,dtype=float)
        for i in range(Y.shape[0]):
            y = Y[i:i + 1]
            if Z is not None:
                z = Z[i:i + 1]
            else:
                z = None
        if grad_channel=='v_D':
            p_g = torch.zeros((y.shape[0],y.shape[1]))
            p_g[i:i + 1] = self.get_smooth_gradients(v_d, de_idx, de_mask, v_p, re_ids, re_mask, z, grad_channel)
        elif grad_channel=='v_De':
            p_g[i:i + 1] = self.get_smooth_gradients(v_d, y, de_mask, v_p, re_ids, re_mask, z, grad_channel)
        elif grad_channel=='v_P':
            p_g[i:i + 1] = self.get_smooth_gradients(v_d, de_idx, de_mask, y, re_ids, re_mask, z, grad_channel)
        elif grad_channel=='v_Pe':
            p_g[i:i + 1] = self.get_smooth_gradients(v_d, de_idx, de_mask, v_p, y, re_mask, z, grad_channel)

        return p_g

def generate_saliency(model, v_d, de_idx, de_mask, v_p, re_ids, re_mask, smooth=False, nsamples=20, stddev=0.15, only_seq=False, train=False):
    saliency = SmoothGrad(model, only_seq, train)
    x_grad = saliency.get_smooth_gradients(v_d, de_idx, de_mask, v_p, re_ids, re_mask, nsamples=nsamples, x_stddev=stddev, t_stddev=stddev)
    return x_grad

class GuidedBackpropReLU(torch.autograd.Function):

    def __init__(self, inplace=False):
        super(GuidedBackpropReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            input,
            pos_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors

        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, pos_mask_1),
            pos_mask_2)

        return grad_input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'

class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self, model, device='cpu', only_seq=False, train=False,
                 x_stddev=0.15, t_stddev=0.15, nsamples=20, magnitude=2):
        super(GuidedBackpropSmoothGrad, self).__init__(
            model, device, only_seq, train, x_stddev, t_stddev, nsamples, magnitude)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()

def compute_saliency(model, device, test_loader, grad_channel='v_P', batch_size=1):
    model.eval()

    identity = "results"
    saliency_path = os.path.join("./out", identity+'.sal')

    sgrad = GuidedBackpropSmoothGrad(model, device=device)
    for batch_idx, (interpret_ds, interpret_de, interpret_rs, interpret_re, interpret_y) in enumerate(test_loader):

        interpret_ds, interpret_rs= interpret_ds.to(device), interpret_rs.float().to(device)

        interpret_y = Variable(torch.from_numpy(np.array(interpret_y)).float()).to(device)

        de_idx = torch.tensor(interpret_de['input_ids']).to(device)
        de_mask = torch.tensor(interpret_de['attention_mask']).to(device)

        interpret_re = {k: v.to(device) for k, v in interpret_re.items() if k in ['input_ids', 'attention_mask']}
        re_atten_mask = interpret_re['attention_mask']
        re_input_ids = interpret_re['input_ids']

        guided_saliency_p= sgrad.get_batch_gradients(interpret_ds, de_idx, de_mask, interpret_rs, re_input_ids, re_atten_mask, interpret_y, grad_channel)
        N = guided_saliency_p.shape[0] # (N, 1, 31, 5)
        
        output = model(interpret_ds, de_idx, de_mask, interpret_rs, re_input_ids, re_atten_mask)
        prob = torch.sigmoid(output)
        p_np = prob.to(device='cpu').detach().numpy().squeeze()
        
        str_sal=[] 
        for i in range(N):
            str_sal = mat2str(np.squeeze(guided_saliency_p[i]))
    return str_sal[:-1]

def mat2str(m):
    string=""
    if len(m.shape)==1:
        for j in range(m.shape[0]):
            string+= "%.4E," % m[j]
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                string+= "%.4E," % m[i,j]
    return string

def add_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for i in range( atoms ):
        mol.GetAtomWithIdx(i).SetProp(
            'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
    return mol,atoms


######################################################################################
##  RNA sequence interpretability
######################################################################################

def interpretability_P(testData, model, device, test_loader, resultsF, index=0, smooth_steps=3):
    sal_p  = compute_saliency(model, device, test_loader, 'v_P').split(",")
    sal_pe = compute_saliency(model, device, test_loader, 'v_Pe').split(",")

    with open(resultsF + '/0_salency_P_'+str(index)+'.txt', 'w') as f:
        for n in range(len(sal_p)//31):
            f.write(','.join(sal_p[n*31:(n+1)*31]) + '\n')
    with open(resultsF + '/0_salency_Pe_'+str(index)+'.txt', 'w') as f:
        f.write(','.join(sal_pe[1:]) + '\n')
    df_p = pd.read_table(resultsF + '/0_salency_P_'+str(index)+'.txt',sep=',',header=None)
    df_pe = pd.read_table(resultsF + '/0_salency_Pe_'+str(index)+'.txt',sep=',',header=None)
    df = pd.concat((df_p,df_pe),axis=0, ignore_index=True)
    
    df3 = pd.Series(list(testData['Sequence'][index]+'AUGC'))
    df4=(pd.get_dummies(df3)).T
    df4 = df4.iloc[:,:31]
    #df2=pd.read_table('./dataset/ref_df.txt',sep=',',header=None)
    df2=pd.DataFrame(index=np.arange(2), columns=np.arange(31)).fillna(1)
    merge = pd.concat([df4,df2],axis=0, ignore_index=True)
    df_final=df*merge
    
    if smooth_steps >= 3:
        df_final.iloc[:4,:] = df_final.iloc[:4,:].sum(axis=0)
        df_final = savgol_filter(df_final, smooth_steps, 1, axis=1, mode='nearest')
        df_final=df_final*merge
    else:
        #df_final = savgol_filter(df_final, 1, 0, axis=1, mode='nearest')
        pass
    
    df3 = pd.Series(list(testData['Structure'][index]+'.()'))
    df5=(pd.get_dummies(df3)).T
    df5 = df5.iloc[:,:31]
    merge = pd.concat([df4,df5],axis=0, ignore_index=True)

    _ = plot_saliency(merge.iloc[:7,:].values, df_final.iloc[:5,:].values, nt_width=100, norm_factor=3, str_null=np.zeros(31), outdir=resultsF + '/2_RNA_final_'+str(index)+'.pdf')
    
    seq_norm = (df_final.iloc[:4,:]-df_final.iloc[:4,:].min().min())/(df_final.iloc[:4,:].max().max()-df_final.iloc[:4,:].min().min())
    seq_norm.loc['avg'] = seq_norm.apply(lambda x: x.sum(),axis=0)
    seq_norm = seq_norm.iloc[4:]

    struct_norm = (df_final.iloc[4:5,:]-df_final.iloc[4:5,:].min().min())/(df_final.iloc[4:5,:].max().max()-df_final.iloc[4:5,:].min().min()+1e-32)
    embedding_norm = (df_final.iloc[5:,:]-df_final.iloc[5:,:].min().min())/(df_final.iloc[5:,:].max().max()-df_final.iloc[5:,:].min().min()+1e-32)
    
    seq_norm = np.average(np.concatenate((seq_norm,embedding_norm)),axis=0)
    seq_norm = (seq_norm-np.min(seq_norm))/(np.max(seq_norm)-np.min(seq_norm))
    df_norm=np.concatenate((seq_norm.reshape(1,seq_norm.shape[0]),struct_norm))

    plt.figure(figsize=(30, 2.5))
    plt.rcParams.update({"font.size":20})
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    df_norm = pd.DataFrame(df_norm, index=['sequence','structure'],columns=list(testData['Sequence'][index]))
    h=sns.heatmap(df_norm, cmap='Reds', linewidths=1.8, linecolor='grey',annot=False,cbar=False)
    cb = h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=10)  # 设置colorbar刻度字体大小。
    
    plt.savefig(resultsF + "/1_RNA_pre_"+str(index)+".pdf")

######################################################################################
##  SmrtNet silency map heatmap and motif plot
######################################################################################


def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm

def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = np.array(Image.fromarray(CHARS[index[j]]).resize((nt_width,nt_height[j])))
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)

def dot_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):
    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = np.array(Image.fromarray(DOT_BRACKET[index[j]]).resize((nt_width,nt_height[j])))
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)

def plot_saliency(X, W, nt_width=100, norm_factor=3, str_null=None, outdir="results/"):
    # filter out zero-padding
    # W = savgol_filter(W, 3, 1, axis=1,mode='nearest')
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]
    num_nt = len(plot_index)
    trace_width = num_nt*nt_width
    trace_height = 400
    
    seq_str_mode = False
    if X.shape[0]>4:
        seq_str_mode = True
        assert str_null is not None, "Null region is not provided."

    # sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width*5//4)

    if seq_str_mode:
        # structure line
        str_raw = X[4, plot_index]
        if str_null.sum() > 0:
            str_raw[str_null.T==1] = -0.01

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw.shape[0]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1-str_raw[v])*trace_height 
            # i+=1
        img_str_raw = dot_logo(X[4:, plot_index], height=nt_width, nt_width=nt_width*5//4)

    # sequence saliency logo
    seq_sal = normalize_pwm(W[:4, plot_index], factor=norm_factor)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width*5//4)
    img_seq_sal = np.array(Image.fromarray(W[:4, plot_index].sum(axis=0).reshape(1,len(plot_index))).resize((trace_width*5, trace_height)))

    if seq_str_mode:
        # structure saliency logo
        str_sal = W[4, plot_index].reshape(1,-1)
        img_str_sal = np.array(Image.fromarray(str_sal).resize((trace_width*5, trace_height)))

    # plot    
    #fig = plt.figure(figsize=(10.1,2))
    plt.style.use('default')
    fig = plt.figure(figsize=(11,3))
    gs = gridspec.GridSpec(nrows=5, ncols=1, height_ratios=[2.5, 0.25, 0.5, 0.5, 0.25])
    cmap_reversed = mpl.cm.get_cmap('Spectral_r')             # jet, Spectral_r, coolwarm, RdBu_r

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal_logo)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)

    ax = fig.add_subplot(gs[2, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[4, 0]) 
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)

        ax = fig.add_subplot(gs[3, 0]) 
        ax.axis('off')
        ax.imshow(img_str_raw)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # save figure
    filepath = outdir
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig

def merge_plot(inputPath,resultsF, index, smooth_steps=3):
    '''
    '''
    testData = pd.read_csv(inputPath,sep='\t',header=None,index_col=0,names=['drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    df_p = np.zeros((5,6,31))
    resM = pd.read_table(resultsF + '/merge_results.txt',sep='\t',header=None,names=['ids','drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    shreshold = resM.loc[index,'prob']
    for i in range(5):
        #if i == 2:
        #    continue

        if pd.read_table(resultsF + '/CV_' + str(i+1) + '/results_sort.txt',sep='\t').loc[0,'prob']<shreshold:
            continue
        try:
            df_tmp_p = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_P_'+str(index)+'.txt',sep=',',header=None)
            df_tmp_pe = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_Pe_'+str(index)+'.txt',sep=',',header=None)
            df_tmp = pd.concat([df_tmp_p, df_tmp_pe], axis=0)
            df_tmp = df_tmp.values
            df_tmp = (df_tmp-df_tmp.min(axis=1)[:,np.newaxis])/(df_tmp.max(axis=1)[:,np.newaxis]-df_tmp.min(axis=1)[:,np.newaxis])
        except:
            df_tmp = np.zeros((6,31))

        df_p[i,:,:] = df_tmp.reshape(1,6,31)

    df_p = df_p.sum(axis=0)/3
    #cutoff = df_p.mean(axis=0)
    #df_p = np.where(df_p>cutoff,df_p,0)
    df_p = pd.DataFrame(df_p)
    #df_p = pd.DataFrame(np.median(df_p,axis=0))


    df3 = pd.Series(list(testData['Sequence'][index]+'AUGC'))
    df4=(pd.get_dummies(df3)).T
    df4 = df4.iloc[:,:31]
    #df2=pd.read_table('./dataset/ref_df.txt',sep=',',header=None)
    df2=pd.DataFrame(index=np.arange(2), columns=np.arange(31)).fillna(1)
    merge = pd.concat([df4,df2],axis=0, ignore_index=True)
    df_final=df_p*merge

    if smooth_steps >= 3:
        df_final.iloc[:4,:] = df_final.iloc[:4,:].sum(axis=0)
        df_final = savgol_filter(df_final, smooth_steps, 1, axis=1, mode='nearest')
        df_final=df_final*merge
    else:
        #df_final = savgol_filter(df_final, 1, 0, axis=1, mode='nearest')
        pass
    
    window_size = 8
    max_value = 0
    for i in range(0,df_final.shape[1]-window_size+1,1):
        if df_final.iloc[:4,i:i+window_size].values.mean() >= max_value:
            max_index = i
            max_value = df_final.iloc[:4,i:i+window_size].values.mean()

    max_value = 0
    for j in range(0,df_final.shape[1]-window_size+1,1):
        if df_final.iloc[4:5,j:j+window_size].values.mean() >= max_value:
            max_jndex = j
            max_value = df_final.iloc[4:5,j:j+window_size].values.mean()
    
    if max_jndex <= max_index and max_jndex+window_size >= max_index:
        df_final.iloc[:,:max_jndex] = 0
        df_final.iloc[:,max_index+window_size:] = 0
    elif max_index <= max_jndex and max_index+window_size >= max_jndex:
        df_final.iloc[:,:max_index] = 0
        df_final.iloc[:,max_jndex+window_size:] = 0
    else:
        df_final.iloc[:,:min(max_index,max_jndex)] = 0
        df_final.iloc[:,min(max_index,max_jndex)+window_size:max(max_index,max_jndex)] = 0
        df_final.iloc[:,max(max_index,max_jndex)+window_size:] = 0

    
    #cutoff = df_final.mean(axis=0) * 2
    #df_final = pd.DataFrame(np.where(df_final>cutoff,df_final,0))

    df3 = pd.Series(list(testData['Structure'][index]+'.()'))
    df5=(pd.get_dummies(df3)).T
    df5 = df5.iloc[:,:31]
    merge = pd.concat([df4,df5],axis=0, ignore_index=True)

    plt.scatter(x=[0,1],y=[0,1],c=[0,1], cmap='Spectral_r')
    plt.colorbar(label="saliency map signal", orientation="horizontal")
    plt.savefig(resultsF + "/0_RNA_colorbar_"+str(index)+".pdf",bbox_inches='tight')
    plt.close()
    _ = plot_saliency(merge.iloc[:7,:].values, df_final.iloc[:5,:].values, nt_width=100, norm_factor=3, str_null=np.zeros(31), outdir=resultsF + '/2_RNA_final_'+str(index)+'.pdf')
    
    seq_norm = (df_final.iloc[:4,:]-df_final.iloc[:4,:].min().min())/(df_final.iloc[:4,:].max().max()-df_final.iloc[:4,:].min().min())
    seq_norm.loc['avg'] = seq_norm.apply(lambda x: x.sum(),axis=0)
    seq_norm = seq_norm.iloc[4:]

    struct_norm = (df_final.iloc[4:5,:]-df_final.iloc[4:5,:].min().min())/(df_final.iloc[4:5,:].max().max()-df_final.iloc[4:5,:].min().min()+1e-32)
    embedding_norm = (df_final.iloc[5:,:]-df_final.iloc[5:,:].min().min())/(df_final.iloc[5:,:].max().max()-df_final.iloc[5:,:].min().min()+1e-32)
    
    seq_norm = np.average(np.concatenate((seq_norm,embedding_norm)),axis=0)
    seq_norm = (seq_norm-np.min(seq_norm))/(np.max(seq_norm)-np.min(seq_norm))
    df_norm=np.concatenate((seq_norm.reshape(1,seq_norm.shape[0]),struct_norm))

    plt.figure(figsize=(30, 2.5))
    plt.rcParams.update({"font.size":20})
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    df_norm = pd.DataFrame(df_norm, index=['sequence','structure'],columns=list(testData['Sequence'][index]))
    h=sns.heatmap(df_norm, cmap='Reds', linewidths=1.8, linecolor='grey',annot=False,cbar=False)
    cb = h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=10)  # 设置colorbar刻度字体大小。
    
    plt.savefig(resultsF + "/1_RNA_pre_"+str(index)+".pdf")    
    plt.close()

    return 1

