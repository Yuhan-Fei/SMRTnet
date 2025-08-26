import pandas as pd
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='Merge_CV')
parser.add_argument('--data_dir', type=str, default="./results/", help='data path')
parser.add_argument('--results_name', type=str, default="results", help='output directory')
args = parser.parse_args()

restsAll = pd.DataFrame({})
for j in range(5):
    rstsets = pd.read_csv(args.data_dir + '/CV_'+str(j+1)+'/'+args.results_name +'_sort.txt',sep='\t', header=0, index_col=0)
    restsAll = pd.concat([restsAll,rstsets['prob']],axis=1)

restsAll['drug'] = rstsets['drug']
restsAll['rna'] = rstsets['rna']
restsAll.columns = ['probs1','probs2','probs3','probs4','probs5','drug','rna']
restsAll = restsAll[['drug','rna','probs1','probs2','probs3','probs4','probs5']]
restsAll['median'] = restsAll[['probs1','probs2','probs3','probs4','probs5']].median(axis=1)
idsSets = restsAll['drug'].str.split('_',expand=True)
restsAll['rds'] = idsSets[0]
rdsList = restsAll['rds'].unique().tolist()

restsMaxConMax = pd.DataFrame({})
restsAllInfo = pd.DataFrame({})
for rds in rdsList:
    rests = restsAll[restsAll['rds']==rds].reset_index(level=0, drop=True)
    if rests.shape[0]==2 and rests.loc[0,'drug']==rests.loc[1,'drug']:
        rests = rests.loc[0:0]
    
    if int(rests.shape[0])>=int(40):
        win = int(8)
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
        restsAllInfo = pd.concat([restsAllInfo, pd.Series()], axis=0)
        continue
    maxStart,maxEnd = 0,0
    start,end = 0,0
    for i in range(len(nums_sort)-1):
        if nums_sort[i+1]==nums_sort[i]+1:
            end = i+1
        elif nums_sort[i+1]==nums_sort[i]:
            end = i+1
        else:
            if end-start >= maxEnd-maxStart:
                maxStart,maxEnd = start,end
            start, end = i+1, i+1    
    if end-start >= maxEnd-maxStart:
        maxStart,maxEnd = start,end
    df_filter = rests.iloc[nums_sort[maxStart] : nums_sort[maxEnd]+ win]
    max_row = df_filter['median'].idxmax()
    restsMaxConMax = pd.concat([restsMaxConMax, rests.iloc[max_row:max_row+1]], axis=0)
    restsAllInfo = pd.concat([restsAllInfo, rests], axis=0)
    restsAllInfo = pd.concat([restsAllInfo, pd.Series()], axis=0)

restsAllInfo.to_csv(args.data_dir + '/'+args.results_name+'.txt', sep='\t', index=None)
restsMaxConMax.to_csv(args.data_dir + '/'+args.results_name+'_final.txt', sep='\t', index=None)



