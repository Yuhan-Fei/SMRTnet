"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import os, pickle, torch
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from loader import data_process_loader
from torch.utils.data import SequentialSampler
from torch.utils import data
from model import SmrtNet
from utils import dgl_collate_func_ds, fix_seed
from explain import interpretability_P, merge_plot
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dict(path):
    with open(os.path.join(path), 'rb') as f:
        return pickle.load(f)

def load_model(config_dir, model_dir, args):
    config = load_dict(config_dir)
    config.cuda = args.cuda
    config.do_test = args.do_test
    config.do_infer = args.do_infer
    config.do_ensemble = args.do_ensemble
    config.do_explain = args.do_explain
    model = SmrtNet(config).to(device)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cuda:' + str(args.cuda))))
    return model


def load_rna(path, n=50):
    try:
        file = open(path, "r")
    except:
        print('RNA Path Not Found, please double check!')
    X_name = []
    X_seq = []
    X_struct = []

    for line in file.readlines()[0:n]:
        values = line.split('\t')
        X_name.append(values[0].replace('\n', ''))
        X_seq.append(values[1].replace('\n', ''))
        X_struct.append(values[2].replace('\n', ''))
    file.close()
    return np.array(X_name), np.array(X_seq), np.array(X_struct)


def load_drug(path, n=50):
    try:
        file = open(path, "r")
    except:
        print('Drug Path Not Found, please double check!')
    X_name = []
    X_smiles = []
    #breakpoint()
    for line in file.readlines()[0:n]:
        values = line.split('\t')
        X_name.append(values[0].replace('\n', ''))
        X_smiles.append(values[1].replace('\n', ''))
    file.close()
    return np.array(X_name), np.array(X_smiles)


def length_func(list_or_tensor):
    if type(list_or_tensor) == list:
        return len(list_or_tensor)
    return list_or_tensor.shape[0]



def predict(df_data,model,resultF, args):
    model.to(device)

    info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, args)

    params = {'batch_size': 1,'shuffle': False,'num_workers': 0,'drop_last': False,'sampler': SequentialSampler(info)}
    params['collate_fn'] = dgl_collate_func_ds
    infer_loader = data.DataLoader(info, **params)

    y_pred = []
    y_label = []
    model.eval()
    #infer_pbar = tqdm(enumerate(BackgroundGenerator(infer_loader)), total=len(infer_loader), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}')
    for i, (infer_ds, infer_de, infer_rs, infer_re, infer_y) in enumerate(infer_loader):
        fix_seed(args.seed)

        #infer_ds, infer_de, infer_rs, infer_re, infer_y = infer_data[1]

        infer_ds, infer_rs = infer_ds.to(device), infer_rs.float().to(device)
        de_idx = torch.tensor(infer_de['input_ids']).to(device)
        de_mask = torch.tensor(infer_de['attention_mask']).to(device)

        infer_re = {k: v.to(device) for k, v in infer_re.items() if k in ['input_ids', 'attention_mask']}
        re_atten_mask = infer_re['attention_mask']
        re_input_ids = infer_re['input_ids']

        output = model(infer_ds, de_idx, de_mask, infer_rs, re_input_ids, re_atten_mask)

        prob = torch.sigmoid(output)
        logits = torch.squeeze(prob).detach().cpu().numpy()

        label_ids = infer_y.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

        #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) > 0.5)])
        resultF_list = resultF.split('/')
        if args.do_explain:
            if prob.item() > 0.70:
                #breakpoint()
                interpretability_P(df_data,model,device,[(infer_ds, infer_de, infer_rs, infer_re, infer_y)],'/'.join(resultF_list[:-1]),i,args.smooth_steps)
        else:
            pass
    return y_pred

def data_process(X_drug, X_target, X_structure):

    y = [0] * int(len(X_drug) * len(X_target))

    if isinstance(X_target, str):
        X_target = [X_target]

    d_n = []
    t_n = []
    s_n = []

    if len(X_target) == 1 and len(X_drug) > 1:
        # one target high throughput screening setting
        t_n = np.tile(X_target, (length_func(X_drug),))
        d_n = X_drug
        s_n = np.tile(X_structure, (length_func(X_drug),))
    elif len(X_drug) == 1 and len(X_target) > 1:
        # one drug high throughput screening setting
        d_n = np.tile(X_drug, (length_func(X_target),))
        t_n = X_target
        s_n = X_structure
    elif len(X_drug) == 1 and len(X_target) == 1:
        d_n = X_drug
        t_n = X_target
        s_n = X_structure
    elif len(X_drug) > 1 and len(X_target) > 1:
        for i in range(len(X_target)):
            for j in range(len(X_drug)):
                d_n.append(X_drug[j])
                t_n.append(X_target[i])
                s_n.append(X_structure[i])

    df_data = pd.DataFrame(zip(d_n, t_n, s_n, y))
    df_data.rename(columns={0: 'SMILES',
                            1: 'Sequence',
                            2: 'Structure',
                            3: 'Label'},
                   inplace=True)

    return  df_data.reset_index(drop=True)

def infer(smiles, sequence, structure, drug_names, target_names, model, args, result_folder, output_num_max=10):

    fo = result_folder
    print_list = []

    table_header = ["Rank", "index", "Drug Name", "RNA Name", "Interaction","Probability"]
    table = PrettyTable(table_header)

    with open(fo, 'w') as fout:
        print('Drug repurpose...')
        df_data = data_process(smiles, sequence, structure)
        y_pred = predict(df_data, model, result_folder,args)

        d_n = []
        t_n = []
        if len(target_names) == 1:
            t_n = np.tile(target_names, (len(drug_names),1)).reshape(-1,).tolist()
            d_n=drug_names
        elif len(drug_names) == 1:
            d_n = np.tile(drug_names, (len(target_names),1)).reshape(-1,).tolist()
            t_n=target_names
        elif len(target_names) >1 and len(drug_names) > 1:
            for i in range(len(drug_names)):
                for j in range(len(target_names)):
                    d_n.append(drug_names[i])
                    t_n.append(target_names[j])


        for i in range(len(d_n)):
            if y_pred[i] > 0.5:
                string_lst = [i,d_n[i], t_n[i], "YES", "{0:.3f}".format(y_pred[i])]
            else:
                string_lst = [i,d_n[i], t_n[i], "NO", "{0:.3f}".format(y_pred[i])]

            print_list.append((string_lst, y_pred[i]))

        print_list.sort(key=lambda x: x[1], reverse=True)
        print_list = [i[0] for i in print_list]

        print_list_df = pd.DataFrame(print_list,columns=['index','drug','rna','inter','prob'])
        print_list_df.sort_values("index",inplace=True)
        print_list_df.to_csv(os.path.splitext(result_folder)[0]+"_sort"+".txt", sep='\t', index=False)
        for idx, lst in enumerate(print_list):
            lst = [str(idx + 1)] + lst
            table.add_row(lst)
        fout.write(table.get_string())

    with open(fo, 'r') as fin:
        #print(fin)
        lines = fin.readlines()

        for idx, line in enumerate(lines):
            if idx < output_num_max+3:
                print(line, end='')
            else:
                print('checkout ' + fo + ' for the whole list')
                break
        print("\n")




"""
def ensemble(i,smiles, sequence, structure, model, args, result_folder, output_num_max=10):

    fo = result_folder

    with open(fo, 'w') as fout:
        print('Predicting interactions based on ('+str(i)+'/5) '+ 'model...')
        #if length>31 切分完后用for循环读入df_data

        df_data = pd.DataFrame()
        sequence_slice=[]
        for i in range(len(sequence)): #是否多条RNA
            if len(sequence[i])>31: #RNA长度是否>31
                for k in range(len(sequence[i])-31):
                    sequence_slice.append(sequence[i][k:k+31].split())
                    df_data=df_data.append(data_process(smiles, sequence_slice[i], structure))
            else:
                df_data = data_process(smiles, sequence[i], structure)

        y_pred = predict(df_data, model, result_folder,args)

    return y_pred
"""

def ensemble(i,smiles, sequence, structure, model, args, result_folder, output_num_max=10):
    fo = result_folder+"/results.txt"

    with open(fo, 'w') as fout:
        print('Predicting interactions based on ('+str(i)+'/5) '+ 'model...')
        #if length>31 切分完后用for循环读入df_data
    
        df_data = data_process(smiles, sequence, structure)

        y_pred = predict(df_data, model, result_folder,args)

    return y_pred


def show_merge(results_median, smiles, sequence, structure, drug_names, target_names, \
               model, args, result_folder, output_num_max=10, minWinSize=4, minSeqLen=40):

    fo = result_folder+"/results.txt"
    print_list = []

    table_header = ["Rank", "index", "Drug Name", "RNA Name", "Interaction","Probability"]
    table = PrettyTable(table_header)

    with open(fo, 'w') as fout:
        print('The results are sorted by probability below...')
        df_data = data_process(smiles, sequence, structure)
        y_pred = results_median
        df_data['prob'] = y_pred

        d_n = []
        t_n = []
        if len(target_names) == 1:
            t_n = np.tile(target_names, (len(drug_names),1)).reshape(-1,).tolist()
            d_n=drug_names
        elif len(drug_names) == 1:
            d_n = np.tile(drug_names, (len(target_names),1)).reshape(-1,).tolist()
            t_n=target_names
        elif len(target_names) >1 and len(drug_names) > 1:
            """
            for i in range(len(drug_names)):
                for j in range(len(target_names)):
                    d_n.append(drug_names[i])
                    t_n.append(target_names[j])
            """
            for i in range(len(target_names)):
                for j in range(len(drug_names)):
                    d_n.append(drug_names[j])
                    t_n.append(target_names[i])

        
        df_data['drugName'] = d_n
        df_data['rnaName'] = t_n
        df_data['rds'] = t_n
        for i in range(df_data.shape[0]):
            NameList = df_data.loc[i,'rnaName'].split('_')
            if len(NameList)>1:
                rdsName = '_'.join(NameList[:-1])
            else:
                rdsName = NameList[0]
            df_data.loc[i,'rds'] = rdsName
        #idsSets = df_data['rnaName'].str.split('_',expand=True)
        #print(idsSets)
        #df_data['rds'] = idsSets[0].str.cat(idsSets[1:-1], sep='_')

        rdsList = df_data['rds'].unique().tolist()   
        restsMaxConMax = pd.DataFrame({})
        ddsList = df_data['drugName'].unique().tolist()  
        for dds in ddsList:
            for rds in rdsList:
                rests = df_data[(df_data['rds']==rds) & (df_data['drugName']==dds)]
                rests = rests.reset_index(level=0, drop=True)
                #if rests.shape[0]==2 and rests.loc[0,'ids']==rests.loc[1,'ids']:
                #    rests = rests.loc[0:0]

                if rests.shape[0]>=minSeqLen:
                    win = minWinSize
                else:
                    win = 1
                
                cutoff=0.5
                df_win=rests['prob'].rolling(win).agg(lambda x: np.all(x>=cutoff)).shift(-win+1).to_frame()
                df_win=df_win.rename(columns={'prob':'win'})
                df_merge=pd.concat([rests, df_win], axis=1)

                nums_sort = df_merge[df_merge['win'] == 1].index.tolist()
                if len(nums_sort)==0:
                    min_row = rests['prob'].idxmin()
                    restsMaxConMax = pd.concat([restsMaxConMax, rests.iloc[min_row:min_row+1]], axis=0)
                    continue
                lengthf = len(nums_sort)
                max_bound = rests.shape[0]
                for i in range(lengthf):
                    if nums_sort[i]+1 not in nums_sort:
                        extendMax = min(max_bound,nums_sort[i]+1+win)
                        nums_sort.extend(list(range(nums_sort[i]+1,extendMax,1)))
                nums_sort = sorted(list(set(nums_sort)))

                df_filter = rests.iloc[nums_sort]
                max_row = df_filter['prob'].idxmax()
                restsMaxConMax = pd.concat([restsMaxConMax, rests.iloc[max_row:max_row+1]], axis=0)
                #restsAllInfo = pd.concat([restsAllInfo, rests], axis=0)
            
        restsMaxConMax = restsMaxConMax.reset_index(level=0, drop=True)

        for i in range(restsMaxConMax.shape[0]):
            if restsMaxConMax.loc[i,'prob'] > 0.5:
                string_lst = [i, restsMaxConMax.loc[i,'drugName'], restsMaxConMax.loc[i,'rnaName'], "YES", "{0:.3f}".format(restsMaxConMax.loc[i,'prob'])]
            else:
                string_lst = [i, restsMaxConMax.loc[i,'drugName'], restsMaxConMax.loc[i,'rnaName'], "NO", "{0:.3f}".format(restsMaxConMax.loc[i,'prob'])]

            print_list.append((string_lst, restsMaxConMax.loc[i,'prob']))
        

        print_list.sort(key=lambda x: x[1], reverse=True)

        print_list = [i[0] for i in print_list]
        
        for idx, lst in enumerate(print_list):
            lst = [str(idx + 1)] + lst
            table.add_row(lst)
        fout.write(table.get_string())


    with open(fo, 'r') as fin:
        #print(fin)
        lines = fin.readlines()

        for idx, line in enumerate(lines):
            if idx < output_num_max+3:
                print(line, end='')
            else:
                print('checkout ' + fo + ' for the whole list')
                break
        print("\n")



def show_ensemble(results_median, smiles, sequence, structure, drug_names, target_names, model, args, result_folder, output_num_max=10):

    fo = result_folder+"/results.txt" 
    print_list = []

    table_header = ["Rank", "index", "Drug Name", "RNA Name", "Interaction","Probability"]
    table = PrettyTable(table_header)

    with open(fo, 'w') as fout:
        print('The results are sorted by probability below...')
        #df_data = data_process(smiles, sequence, structure)
        y_pred = results_median


        d_n = []
        t_n = []
        if len(target_names) == 1:
            t_n = np.tile(target_names, (len(drug_names),1)).reshape(-1,).tolist()
            d_n=drug_names
        elif len(drug_names) == 1:
            d_n = np.tile(drug_names, (len(target_names),1)).reshape(-1,).tolist()
            t_n=target_names
        
        elif len(target_names) >1 and len(drug_names) > 1:
            """
            for i in range(len(drug_names)):
                for j in range(len(target_names)):
                    d_n.append(drug_names[i])
                    t_n.append(target_names[j])
            """
            for i in range(len(target_names)):
                for j in range(len(drug_names)):
                    d_n.append(drug_names[j])
                    t_n.append(target_names[i])



        for i in range(len(d_n)):
            if y_pred[i] > 0.7:
                string_lst = [i, d_n[i], t_n[i], "YES", "{0:.3f}".format(y_pred[i])]
            else:
                string_lst = [i, d_n[i], t_n[i], "NO", "{0:.3f}".format(y_pred[i])]

            print_list.append((string_lst, y_pred[i]))
        #breakpoint()
        print_list.sort(key=lambda x: x[1], reverse=True)

        print_list = [i[0] for i in print_list]
        
        for idx, lst in enumerate(print_list):
            lst = [str(idx + 1)] + lst
            table.add_row(lst)
        fout.write(table.get_string())


    with open(fo, 'r') as fin:
        #print(fin)
        lines = fin.readlines()

        for idx, line in enumerate(lines):
            if idx < output_num_max+3:
                print(line, end='')
            else:
                print('checkout ' + fo + ' for the whole list')
                break
        print("\n")


def show_delta(results_median_list,results_median_final, smiles, sequence, structure, drug_names, target_names, model, args, result_folder, output_num_max=10):

    fo = result_folder+"/results.txt" 
    print_list = []

    table_header = ["Rank", "index", "Drug Name", target_names[0], target_names[1],"Delta"]
    table = PrettyTable(table_header)

    with open(fo, 'w') as fout:
        print('The results are sorted by delta below...')
        df_data = data_process(smiles, sequence, structure)
        y_pred = results_median_final

        d_n = []
        t_n = []
        cas =[]
        if len(target_names) == 1:
            pass
        elif len(drug_names) == 1:
            split=int(len(results_median_list)/2)
            d_n=results_median_list[:split]
            t_n=results_median_list[split:]
        elif len(target_names) >1 and len(drug_names) > 1:
            split=int(len(results_median_list)/2)
            d_n=results_median_list[:split]
            t_n=results_median_list[split:]
            for i in range(len(drug_names)):
                cas.append(drug_names[i])


        for i in range(len(d_n)):
            string_lst = [i, cas[i],  "{0:.3f}".format(d_n[i]), "{0:.3f}".format(t_n[i]), "{0:.3f}".format(y_pred[i])]


            print_list.append((string_lst, y_pred[i]))
        print_list.sort(key=lambda x: x[1], reverse=True)

        print_list = [i[0] for i in print_list]
        
        for idx, lst in enumerate(print_list):
            lst = [str(idx + 1)] + lst
            table.add_row(lst)
        fout.write(table.get_string())


    with open(fo, 'r') as fin:
        #print(fin)
        lines = fin.readlines()

        for idx, line in enumerate(lines):
            if idx < output_num_max+3:
                print(line, end='')
            else:
                print('checkout ' + fo + ' for the whole list')
                break
        print("\n")


def rna_slice(inputPath, step):
    inputPath_fa = os.path.splitext(inputPath)[0]+'.fa'
    with open(inputPath) as f:
        line = f.readline()
    column=len(line.split())

    if(column==3):
        ids, seqs, strs = [], [], []
        for line in open(inputPath,'r'):
            id, seq, struct = line.split('\t')
            ids.append(id)
            seqs.append(seq)
            strs.append(struct.replace('\n',''))
        #print("SmrtNet get "+len(ids)+" RNA sequences...")
        with open(inputPath_fa,'w') as f:
            for i in range(len(ids)):
                #print(">"+ids[i]+'\n'+seqs[i]+'\n'+strs[i])
                f.write(">"+ids[i]+'\n'+seqs[i]+'\n'+strs[i]+'\n')
        f.close()
    elif(column==1):
        pass
    else:
        print("input format error")

    outputPath=os.path.splitext(inputPath)[0]+"_slice"+".txt"
    write_object = open(outputPath,'w')
    with open(inputPath_fa) as read_object:
        for i,line in enumerate(read_object):
            if i%3==0:
                name = line.strip().replace('>','')
            elif i%3==1:
                sequences = line.strip().upper().replace('T','U')
            else:
                structures = line.strip()

                if(len(sequences)>31):
                    print("Length of "+name+" > 31nt, slicing to "+str(len(sequences)-30)+" 31nt fragments...")
                    for j in range(0,len(sequences)-30,step):
                        write_object.write('{}\t{}\t{}\n'.format(name+'_'+str(j),sequences[j:j+31],structures[j:j+31]))
                elif(len(sequences)==31):
                    print("Length of "+name+" = 31nt...")
                    write_object.write('{}\t{}\t{}\n'.format(name+'_0',sequences,structures))
                else:
                    print("Length of "+name+" is too short")

    write_object.close()


def explain_merge(inputPath, drugPath, rnaPath,smooth_steps):
    "Generating merged results...."
    drugs = pd.read_csv(drugPath,sep='\t',header=None,index_col=None,names=['name','smiles'])
    rnas = pd.read_csv(rnaPath,sep='\t',header=None,index_col=None,names=['name','seq','struct'])
    restsAll = pd.DataFrame({})
    for i in range(5):

        resTmp = pd.read_csv(inputPath+'/CV_'+str(i+1)+'/results_sort.txt', sep='\t', index_col=None)
        if i==0:
            restsAll = pd.concat([restsAll,resTmp['index']],axis=1)
            restsAll = pd.concat([restsAll,resTmp['drug']],axis=1)
            restsAll = pd.concat([restsAll,resTmp['rna']],axis=1)
        restsAll = pd.concat([restsAll,resTmp['prob']],axis=1)

    restsAll.columns = ['index','drug','rna','probs1','probs2','probs3','probs4','probs5']
    restsAll['median'] = restsAll[['probs1','probs2','probs3','probs4','probs5']].median(axis=1)
    with open(inputPath+'/merge_results.txt','w') as write_object:
        for i in range(len(restsAll.index)):
            write_object.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(restsAll.loc[i,'index'], restsAll.loc[i,'drug'], restsAll.loc[i,'rna'],\
                drugs[drugs['name']==restsAll.loc[i,'drug']]['smiles'].values[0], rnas[rnas['name']==restsAll.loc[i,'rna']]['seq'].values[0],\
                rnas[rnas['name']==restsAll.loc[i,'rna']]['struct'].values[0], restsAll.loc[i,'median']))

    for i in range(len(restsAll.index)):
        _ = merge_plot(inputPath+'/merge_results.txt',inputPath,i,smooth_steps)





def slidingwindow(inputPath,tmpPath):
    print("Slicing all RNA to 31nt fragments...")
    PAIRS = {'G':'C',
            'C':'G',
            'A':'U',
            'U':'A'}
    
    write_object = open( tmpPath + '/testInput.txt','w')
    datasets = pd.read_csv(inputPath, sep='\t', header=None, index_col=None,\
                            names=['Index', 'SMILES', 'Target_RNA_sequence', 'Structure', 'Label'])

    for i in range(datasets.shape[0]):
        smile = datasets.loc[i,'SMILES']
        seqs = datasets.loc[i,'Target_RNA_sequence']
        structs = datasets.loc[i,'Structure']

        if len(structs) != len(seqs):
            print('Warning: sequence not equal to structure length\n')
            continue

        if len(seqs) > 31:
            for j in range(0,len(seqs)-30,1):
                write_object.write('{}\t{}\t{}\t{}\t{}\n'.format(str(datasets.loc[i,'Index'])+'_'+str(j+1),smile,seqs[j:j+31],structs[j:j+31],datasets.loc[i,'Label']))
        elif len(seqs) >= 25:
            extend_l = (31 - len(seqs))//2
            extend_r = 31 - len(seqs) - extend_l
            seqs = 'G' * extend_l + seqs + 'C'*extend_r
            structs = '('*extend_l + structs + ')'*extend_r
            write_object.write('{}\t{}\t{}\t{}\t{}\n'.format(str(datasets.loc[i,'Index'])+'_1',smile,seqs,structs,datasets.loc[i,'Label']))
            write_object.write('{}\t{}\t{}\t{}\t{}\n'.format(str(datasets.loc[i,'Index'])+'_1',smile,seqs,structs,datasets.loc[i,'Label']))
        elif len(seqs) < 25:
            print('Warning: RNA length is shorter than 25 nt\n')
            continue
        else:
            write_object.write('{}\t{}\t{}\t{}\t{}\n'.format(str(datasets.loc[i,'Index'])+'_1',smile,seqs,structs,datasets.loc[i,'Label']))
            write_object.write('{}\t{}\t{}\t{}\t{}\n'.format(str(datasets.loc[i,'Index'])+'_1',smile,seqs,structs,datasets.loc[i,'Label']))


    write_object.close()




def matrix(infer_out_dir, minWinSize, minSeqLen):
    tmpPath = infer_out_dir
    outPath = infer_out_dir+"/results_tmp.txt"
    finalPath = infer_out_dir+"/results.txt"
    winMax = minWinSize
    stageL = minSeqLen
    datasets = pd.read_csv(tmpPath + '/testInput.txt',sep='\t', header=None, index_col=None,\
                        names=['ids','smiles','seqs','structs','labels'])

    restsAll = pd.DataFrame({})
    for j in range(5):
        rstsets = pd.read_csv(tmpPath +'/CV_'+str(j+1)+'_prediction.txt',sep='\t', header=None, index_col=None,\
                        names=['preds','probs'])
        restsAll = pd.concat([restsAll,rstsets['probs']],axis=1)
    restsAll['ids'] = datasets['ids']
    idsSets = restsAll['ids'].str.split('_',expand=True)
    restsAll['rds'] = idsSets[0]
    rdsList = restsAll['rds'].unique().tolist()

    restsMaxConMax = pd.DataFrame({})
    restsAllInfo = pd.DataFrame({})
    for rds in rdsList:
        rests = restsAll[restsAll['rds']==rds]

        rests.columns = ['probs1','probs2','probs3','probs4','probs5','ids','rds']
        rests = rests[['ids','probs1','probs2','probs3','probs4','probs5']]
        rests['avg'] = rests[['probs1','probs2','probs3','probs4','probs5']].mean(axis=1)
        rests['median'] = rests[['probs1','probs2','probs3','probs4','probs5']].median(axis=1)
        rests['max'] = rests[['probs1','probs2','probs3','probs4','probs5']].max(axis=1)
        rests['min'] = rests[['probs1','probs2','probs3','probs4','probs5']].min(axis=1)
        rests['labels'] = datasets['labels']
        rests['seqs'] = datasets['seqs']
        rests['structs'] = datasets['structs']
        rests['smiles'] = datasets['smiles']
        rests = rests.reset_index(level=0, drop=True)
        if rests.shape[0]==2 and rests.loc[0,'ids']==rests.loc[1,'ids']:
            rests = rests.loc[0:0]

        #win = (rests.shape[0]//20) + 1
        if rests.shape[0]>=int(stageL):
            win = int(winMax)
        else:
            win = int(1)
        
        cutoff=0.5
        df_win=rests['median'].rolling(win).agg(lambda x: np.all(x>=cutoff)).shift(-win+1).to_frame()
        df_win=df_win.rename(columns={'median':'win'})
        df_merge=pd.concat([rests, df_win], axis=1)

        nums_sort = df_merge[df_merge['win'] == 1].index.tolist()
        if len(nums_sort)==0:
            min_row = rests['median'].idxmin()
            restsMaxConMax = pd.concat([restsMaxConMax, rests.iloc[min_row:min_row+1]], axis=0)
            restsAllInfo = pd.concat([restsAllInfo, rests], axis=0)
            restsAllInfo = restsAllInfo.append(pd.Series(),ignore_index=True)
            continue
        lengthf = len(nums_sort)
        max_bound = rests.shape[0]
        for i in range(lengthf):
            if nums_sort[i]+1 not in nums_sort:
                extendMax = min(max_bound,nums_sort[i]+1+win)
                nums_sort.extend(list(range(nums_sort[i]+1,extendMax,1)))
        nums_sort = sorted(list(set(nums_sort)))

        df_filter = rests.iloc[nums_sort]
        max_row = df_filter['median'].idxmax()
        restsMaxConMax = pd.concat([restsMaxConMax, rests.iloc[max_row:max_row+1]], axis=0)
        restsAllInfo = pd.concat([restsAllInfo, rests], axis=0)
        restsAllInfo = restsAllInfo.append(pd.Series(),ignore_index=True)

    restsAllInfo.to_csv(outPath, sep='\t', index=None)
    restsMaxConMax.to_csv(finalPath, sep='\t', index=None)

    prediction = restsMaxConMax['median'].tolist()
    label  = restsMaxConMax['labels'].tolist()

    K = 0.5
    repl_true = "1"
    repl_false = "0"
    res = []
    for ele in prediction:
        if ele >= K :
            res.append(repl_true)
        else :
            res.append(repl_false)

    res = [eval(i) for i in res]

    accuracy=accuracy_score(label, res, normalize=True, sample_weight=None)
    print('accuracy : %.3f' % accuracy)

    precision = precision_score(label, res, average='binary')
    print('precision: %.3f' % precision)

    recall = recall_score(label, res, average='binary')
    print('recall   : %.3f' % recall)

    f1 = f1_score(label, res, average='binary')
    print('f1_score : %.3f' % f1)

    label = np.array(label)
    y_scores = np.array(prediction)
    auroc=roc_auc_score(label, y_scores)
    print('AUROC    : %.3f' % auroc)

    precision, recall, _ = precision_recall_curve(label, y_scores)
    auc_precision_recall = auc(recall, precision)
    print('AUPRC    : %.3f' % auc_precision_recall)


"""
def show_matrix(input_path):
    results_df=pd.read_table(input_path,sep='\t')

    prediction = results_df['median'].tolist()
    label  = results_df['labels'].tolist()

    K = 0.5
    repl_true = "1"
    repl_false = "0"
    res = []
    for ele in prediction:
        if ele >= K :
            res.append(repl_true)
        else :
            res.append(repl_false)

    res = [eval(i) for i in res]

    accuracy=accuracy_score(label, res, normalize=True, sample_weight=None)
    print('accuracy : %.3f' % accuracy)

    precision = precision_score(label, res, average='binary')
    print('precision: %.3f' % precision)

    recall = recall_score(label, res, average='binary')
    print('recall   : %.3f' % recall)

    f1 = f1_score(label, res, average='binary')
    print('f1_score : %.3f' % f1)

    label = np.array(label)
    y_scores = np.array(prediction)
    auroc=roc_auc_score(label, y_scores)
    print('AUROC    : %.3f' % auroc)

    precision, recall, _ = precision_recall_curve(label, y_scores)
    auc_precision_recall = auc(recall, precision)
    print('AUPRC    : %.3f' % auc_precision_recall)

   #print('AUPRC : %.3f, AUROC: %.3f' % (auroc, auc_precision_recall))
"""
