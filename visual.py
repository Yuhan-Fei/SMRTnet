"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import pandas as pd
import numpy as np 
from rdkit import Chem
from LM_Mol.tokenizer import MolTranBertTokenizer
from scipy.signal import savgol_filter
from explain import plot_saliency
import xsmiles, json

def xsimles(inputPath,resultsF, index, smooth_steps=3, select_cv=0):

    testData = pd.read_csv(inputPath,sep='\t',header=None,index_col=0,names=['drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    resM = pd.read_table(resultsF + '/merge_results.txt',sep='\t',header=None,names=['ids','drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    shreshold = resM.loc[index,'prob']
    smiles = testData['SMILES'][index]
    vocab_reverse = {}
    with open('./LM_Mol/bert_vocab.txt') as read_object:
        for i,line in enumerate(read_object):
            vocab_reverse[i] = line.strip()
    
    tokenizer = MolTranBertTokenizer('./LM_Mol/bert_vocab.txt')
    interpret_de = tokenizer.batch_encode_plus([''.join(smile) for smile in [smiles]], padding=True, add_special_tokens=True)
    de_idx = interpret_de['input_ids'][0]
    de_smiles = [vocab_reverse[x].upper() for x in de_idx][1:-1]
    de_idx_list = []
    
    mol = Chem.MolFromSmiles(smiles)
    i, j = 0, 0  
    while i < mol.GetNumAtoms():
        if mol.GetAtomWithIdx(i).GetSymbol().upper() not in de_smiles[j]:
            j+=1
        elif mol.GetAtomWithIdx(i).GetSymbol().upper() in de_smiles[j]:
            de_idx_list.append(j)
            i+=1
            j+=1
    de_smiles = [de_smiles[x] for x in de_idx_list]

    for i in range(5):
        try:

            n = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_D_'+str(index)+'.txt',sep=',',header=None).shape[1]
            break
        except:
            pass
    attn = np.zeros((5,2,n))

    if select_cv == 0:
        for i in range(5):
            if pd.read_table(resultsF + '/CV_' + str(i+1) + '/results_sort.txt',sep='\t').loc[0,'prob']<shreshold:
                continue
            try:
                df_tmp_d = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_D_'+str(index)+'.txt',sep=',',header=None)
                df_tmp_d = df_tmp_d.values
                df_tmp_d = (df_tmp_d-df_tmp_d.min(axis=1)[:,np.newaxis])/(df_tmp_d.max(axis=1)[:,np.newaxis]-df_tmp_d.min(axis=1)[:,np.newaxis])
            except:
                df_tmp_d = np.zeros((2,n))
            attn[i,:,:] = df_tmp_d.reshape(1,2,n)
        attn = attn.sum(axis=0)/3
    
    elif select_cv >= 6:
        cv_compare_list = []
        for i in range(5):
            cv_compare_list.append(pd.read_table(resultsF + '/CV_' + str(i+1) + '/results_sort.txt',sep='\t').loc[0,'prob'])
        
        shreshold = np.median(cv_compare_list)
        i = cv_compare_list.index(shreshold)
        try:
            df_tmp_d = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_D_'+str(index)+'.txt',sep=',',header=None)
            df_tmp_d = df_tmp_d.values
            df_tmp_d = (df_tmp_d-df_tmp_d.min(axis=1)[:,np.newaxis])/(df_tmp_d.max(axis=1)[:,np.newaxis]-df_tmp_d.min(axis=1)[:,np.newaxis])
        except:
            df_tmp_d = np.zeros((2,n))
        attn[i,:,:] = df_tmp_d.reshape(1,2,n)
        attn = attn.sum(axis=0)
    
    else:
        try:
            df_tmp_d = pd.read_table(resultsF + '/CV_' + str(select_cv) + '/0_salency_D_'+str(index)+'.txt',sep=',',header=None)
            df_tmp_d = df_tmp_d.values
            df_tmp_d = (df_tmp_d-df_tmp_d.min(axis=1)[:,np.newaxis])/(df_tmp_d.max(axis=1)[:,np.newaxis]-df_tmp_d.min(axis=1)[:,np.newaxis])
        except:
            df_tmp_d = np.zeros((2,n))
        attn[select_cv-1,:,:] = df_tmp_d.reshape(1,2,n)        
        attn = attn.sum(axis=0)

    if smooth_steps >= 3:
        attn = savgol_filter(attn, smooth_steps, 1, axis=1, mode='nearest')
    else:
        pass

    attn_smiles = (attn[0,:]-attn[0,:].min())/(attn[0,:].max()-attn[0,:].min())
    attn_struct = (attn[1,:]-attn[1,:].min())/(attn[1,:].max()-attn[1,:].min())
    attn = np.average(np.concatenate((attn_smiles.reshape(-1,attn.shape[1]),attn_struct.reshape(-1,attn.shape[1]))),axis=0)
    attn = (attn-attn.min())/(attn.max()-attn.min())

    atom_cols = {}
    colors = [(25,178,255),(82,208,255),(117,226,255),(163,241,255),(209,251,255),\
              (255,234,211),(255,206,168),(255,173,124),(255,141,92),(255,88,38)]
    for i,v in enumerate(colors):
        colors[i] = (colors[i][0]/255,colors[i][1]/255,colors[i][2]/255)
    for i in range(n):
        if int(attn.tolist()[i]*10//1)==10:
            atom_cols[i] = colors[9]
        else:
            atom_cols[i] = colors[int(attn.tolist()[i]*10//1)]

    small_molecule = {
            'string': smiles,
            'methods': [{'name': 'atoms', 'scores': attn.tolist()}],
            'attributes':{'Pred.':shreshold}
    }
    view_config = {
        'hideBarChart': False,
        'hideAttributesTable': True, 
        'drawerType': 'RDKitDrawer', # Possible values RDKitDrawer (colored) or RDKitDrawer_black (black).    
    }
    gradient_config = {    
        'palette': 'PiYG_5', # default: PiYG_5. 
        'thresholds': [], #default []
        'colorDomain': [-1, 0, 1],  #default: [-1,0,1]
        
        'radius': {'min': 15, 'max': 40},  #default: {'min': 15, 'max': 40}
        'opacity': {'min': 0.6, 'max': 1}, #default: {'min': 0.6, 'max': 1} 
        'blur': 0.7, #default: 0.7
    }
    #w = xsmiles.XSmilesWidget(molecules=json.dumps([small_molecule]), gradient_config=json.dumps(gradient_config))
    w = xsmiles.XSmilesWidget(molecules=json.dumps([small_molecule]))
    return w


def RNA_plot(inputPath, resultsF, index, smooth_steps=3):
    '''
    '''
    testData = pd.read_csv(inputPath,sep='\t',header=None,index_col=0,names=['drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    df_p = np.zeros((5,6,31))
    df_p_median = np.zeros((6,31))
    resM = pd.read_table(resultsF + '/merge_results.txt',sep='\t',header=None,names=['ids','drug', 'rna', 'SMILES', 'Sequence', 'Structure', 'prob'])
    shreshold = resM.loc[index,'prob']
    for i in range(5):
        if pd.read_table(resultsF + '/CV_' + str(i+1) + '/results_sort.txt',sep='\t').loc[0,'prob']<shreshold:
            continue
        try:
            df_tmp_p = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_P_'+str(index)+'.txt',sep=',',header=None)
            df_tmp_pe = pd.read_table(resultsF + '/CV_' + str(i+1) + '/0_salency_Pe_'+str(index)+'.txt',sep=',',header=None)
            df_tmp = pd.concat([df_tmp_p, df_tmp_pe], axis=0)
            df_tmp = df_tmp.values
            if pd.read_table(resultsF + '/CV_' + str(i+1) + '/results_sort.txt',sep='\t').loc[0,'prob']==shreshold:
                df_p_median = df_tmp.copy()
            df_tmp = (df_tmp-df_tmp.min(axis=1)[:,np.newaxis])/(df_tmp.max(axis=1)[:,np.newaxis]-df_tmp.min(axis=1)[:,np.newaxis])
        except:
            df_tmp = np.zeros((6,31))
        df_p[i,:,:] = df_tmp.reshape(1,6,31)

    df_p = df_p.sum(axis=0)/3
    df_p = pd.DataFrame(df_p)
    df_p_median = pd.DataFrame(df_p_median)


    df3 = pd.Series(list(testData['Sequence'][index]+'AUGC'))
    df4=(pd.get_dummies(df3)).T
    df4 = df4.iloc[:,:31]
    df2=pd.DataFrame(index=np.arange(2), columns=np.arange(31)).fillna(1)
    merge = df4.append(df2, ignore_index=True)
    df_final=df_p*merge
    df_final_median = df_p_median*merge
    
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

    if smooth_steps >= 3:
        df_final.iloc[:4,:] = df_final.iloc[:4,:].sum(axis=0)
        df_final_median.iloc[:4,:] = df_final_median.iloc[:4,:].sum(axis=0)
        
        df_final = savgol_filter(df_final, smooth_steps, 1, axis=1, mode='nearest')
        df_final_median = savgol_filter(df_final_median, smooth_steps, 1, axis=1, mode='nearest')
        
        df_final=df_final*merge
        df_final_median=df_final_median*merge
    else:
        pass
    

    df3 = pd.Series(list(testData['Structure'][index]+'.()'))
    df5=(pd.get_dummies(df3)).T
    df5 = df5.iloc[:,:31]
    merge = df4.append(df5, ignore_index=True)

    w = plot_saliency(merge.iloc[:7,:].values, df_final.iloc[:5,:].values, nt_width=100, norm_factor=3, str_null=np.zeros(31), outdir=resultsF + '/2_RNA_final_'+str(index)+'.pdf')
    w_median = plot_saliency(merge.iloc[:7,:].values, df_final_median.iloc[:5,:].values, nt_width=100, norm_factor=3, str_null=np.zeros(31), outdir=resultsF + '/3_RNA_final_'+str(index)+'.pdf')

    return w, w_median

def load(inputPath, select_cv=0):
    restsAll = pd.DataFrame({})
    for i in range(5):
        resTmp = pd.read_csv(inputPath+'/CV_'+str(i+1)+'/results_sort.txt', sep='\t', index_col=None,)
        if i==0:
            restsAll = pd.concat([restsAll,resTmp['index']],axis=1)
            restsAll = pd.concat([restsAll,resTmp['drug']],axis=1)
            restsAll = pd.concat([restsAll,resTmp['rna']],axis=1)
        restsAll = pd.concat([restsAll,resTmp['prob']],axis=1)
    restsAll.columns = ['index','drug','rna','probs1','probs2','probs3','probs4','probs5']
    restsAll['median'] = restsAll[['probs1','probs2','probs3','probs4','probs5']].median(axis=1)
    drug_visualization,rna_visualization = [],{'merge':[],'median':[]}
    for i in range(len(restsAll.index)):
        #drug_visualization.append(xsimles(inputPath+'/merge_results.txt',inputPath,i,3,select_cv))
        fig_merge, fig_median = RNA_plot(inputPath+'/merge_results.txt',inputPath,i,3)
        rna_visualization['merge'].append(fig_merge)
        rna_visualization['median'].append(fig_median)
    rna_visualization = pd.DataFrame(rna_visualization)
    print("There are "+str(len(rna_visualization))+" of interactions has been loaded...")
    print("Please use visualization.loc[index,'merge'],index=(1,2,3..."+str(len(rna_visualization))+") to visualize the merge 5-CV binding sites prediction")
    print("Please use visualization.loc[index,'median'],index=(1,2,3..."+str(len(rna_visualization))+") to visualize the final results' binding sites prediction")
    return restsAll, rna_visualization
