

"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import torch, os, argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from smrtnet.utils import fix_seed, make_directory , dgl_collate_func_ds
from smrtnet.loader import data_process_loader
from smrtnet.loop import test
from smrtnet.infer import *

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='SmrtNet')

    # Data options
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument('--data_dir', type=str, default="./data/SMRTnet-data-demo.txt", help='data path')
    parser.add_argument('--out_dir', type=str, default="./results/model_output", help='output directory')
    parser.add_argument("--do_infer", action='store_true', help="Whether to run infer on the dev set")
    parser.add_argument('--infer_rna_dir', type=str, default="./data/MYC_IRES.txt", help='infer rna directory')
    parser.add_argument('--infer_drug_dir', type=str, default="./data/IHT.txt", help='infer drug directory')
    parser.add_argument('--infer_config_dir', type=str, default="./results/SMRTnet_model/config.pkl", help='infer config directory')
    parser.add_argument('--infer_model_dir', type=str, default="./results/SMRTnet_model/SMRTnet_cv1.pth", help='infer model directory')
    parser.add_argument('--infer_out_dir', type=str, default="./results/results.txt", help='infer output directory')
    parser.add_argument("--do_explain", action='store_true',  help="Whether to run infer on the dev set.")
    parser.add_argument("--do_ensemble", action='store_true',  help="Whether to run infer based on 5 models")

    parser.add_argument('--mode', type=str, default="SPU", help='data mode')
    parser.add_argument('--cuda', type=int, default=0, help='number of GPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Training Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function')

    # Language model parameters
    parser.add_argument('--lm_rna_config', type=str, default="./LM_RNA/parameters.json", help='pretrained hyperparameters .json file path')
    parser.add_argument('--lm_rna_model', type=str, default="./LM_RNA/model_state_dict/rnaall_img0_min30_lr5e5_bs30_2w_7136294_norm1_05_1025_150M_16_rope_fa2_noropeflash_eps1e6_aucgave_1213/epoch_0/LMmodel.pt", help='pretrained model .pt file path')
    parser.add_argument('--lm_mol_config', type=str, default="./LM_Mol/bert_vocab.txt", help='Smiles vocal')
    parser.add_argument('--lm_mol_model', type=str, default="./LM_Mol/pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt", help='pretrained model .ckpt file path')

    #inter
    parser.add_argument('--smooth_steps', type=int, default=3, help='interpreter smooth steps, must be odd')
    parser.add_argument('--maxRNA', type=int, default=100000, help='infer RNA max numbers')
    parser.add_argument('--maxDrug', type=int, default=1000000, help='infer Drug max numbers')
    parser.add_argument('--minWinSize', type=int, default=4, help='continous')
    parser.add_argument('--minSeqLen', type=int, default=40, help='length of ensemble')

    # log
    parser.add_argument('--tfboard', action='store_true', help='tf board')

    args = parser.parse_args()
    #######################################################################################################################


    if  not args.do_test and not args.do_infer and not args.do_explain and not args.do_ensemble:
        raise ValueError("At least one of `do_test`, `do_infer`, `do_explain`, or `do_ensemble` must be True.")
    
    filePathList = {'RNA Language model config':args.lm_rna_config+'/config.json', \
        'RNA Language model parameters':args.lm_rna_model, \
        'small molecule Language model config':args.lm_mol_config, \
        'small molecule Language model parameters':args.lm_mol_model}
    for key,filePath in filePathList.items():
        if not os.path.exists(filePath):
            raise ValueError(key + " file is not in the designated location. " + \
                "Please refer to the contents of the README to download the specified model file and "+\
                "extract it to the correct location or to the designated path to the corresponding location.")
    
    dir = os.path.join(args.out_dir, '')
    if os.path.isdir(dir):
        print("Log file already existed! Please change the name of log file.")

    fix_seed(args.seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    torch.cuda.set_device(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('No. GPU: '+str(args.cuda))

    """Print Mode & Load data"""
    if (args.mode == "SEQ"):
        dataset = pd.read_csv(args.data_dir, sep="\t", header=None, dtype=str, names=["SMILES", "Sequence", "Label"])
    elif (args.mode == "SPU"):
        dataset = pd.read_csv(args.data_dir, sep="\t", header=None, dtype=str, names=["SMILES", "Sequence", "Structure", "Label"])
    elif (args.mode == "PU"):
        dataset = pd.read_csv(args.data_dir, sep="\t", header=None, dtype=str, names=["SMILES", "Structure", "Label"])

    
    pos_num = len(dataset[dataset["Label"] == '1'])
    neg_num = len(dataset[dataset["Label"] == '0'])
    total_num=int(pos_num+neg_num)
    BCE_weight = int(neg_num / pos_num)

    if args.do_test:
        best_model = load_model(args.infer_config_dir, args.infer_model_dir, args)
        total_params = sum(torch.numel(p) for p in best_model.parameters())
        print(f"Model total parameters: {total_params:,d}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_weight))
        test_df = dataset

        test_df.reset_index(drop=True, inplace=True)

        params_test = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False,
                       'sampler': SequentialSampler(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args))}


        params_test['collate_fn'] = dgl_collate_func_ds


        test_loader = DataLoader(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args), **params_test)

        best_model.eval()
        met, loss, Y, P = test(args, best_model, device, test_loader, criterion)
        #print(Y,P)
        best_test_msg = (f'test__loss: {met.other[0]:.3f} ' +
                         f'test__accuracy: {met.acc:.3f} ' +
                         f'test__precision: {met.pre:.3f} ' +
                         f'test__recall: {met.rec:.3f} ' +
                         f'valid_f1: {met.f1:.3f} ' +
                         f'test__AUC: {met.auc:.3f} ' +
                         f'test__PRC: {met.prc:.3f} ')

        #print("\n")
        print(best_test_msg)
        #print("\n")

        save_path = make_directory(args.out_dir, '')

        P = np.squeeze(P)

        with open(save_path + "/{}_prediction.txt".format("Test"), 'w') as f:
            for i in range(len(Y)):
                f.write(str(Y[i]) + "\t" + str(P[i]) + '\n')


    elif args.do_infer:
        model = load_model(args.infer_config_dir, args.infer_model_dir, args)
        #rna, seq, struct = load_rna(args.infer_rna_dir, args.maxRNA)
        rna_slice(inputPath=args.infer_rna_dir,step=1)
        rna, seq, struct = load_rna(os.path.splitext(args.infer_rna_dir)[0]+"_slice"+".txt", args.maxRNA)
        drug, smiles = load_drug(args.infer_drug_dir, args.maxDrug)
        infer(smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)


    elif args.do_explain:
        make_directory(args.infer_out_dir,'/')
        for i in range(1,6):
            make_directory(args.infer_out_dir,"CV_"+str(i))
            model_path=str(args.infer_model_dir)+"/SMRTnet_cv"+str(i)+".pth"
            model = load_model(args.infer_config_dir, model_path, args)
            rna, seq, struct = load_rna(args.infer_rna_dir, args.maxRNA)
            drug, smiles = load_drug(args.infer_drug_dir, args.maxDrug)
            infer(smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir+"/CV_"+str(i)+"/results.txt", output_num_max=99999)
        explain_merge(args.infer_out_dir, args.infer_drug_dir, args.infer_rna_dir, args.smooth_steps)


    elif args.do_ensemble:
        make_directory(args.infer_out_dir, '/')
        df_predict = pd.DataFrame()
        rna_slice(inputPath=args.infer_rna_dir,step=1)
        for i in range(1,6):
            model_path=str(args.infer_model_dir)+"/SMRTnet_cv"+str(i)+".pth"
            model = load_model(args.infer_config_dir, model_path, args)
            rna, seq, struct = load_rna(os.path.splitext(args.infer_rna_dir)[0]+"_slice"+".txt", args.maxRNA)
            drug, smiles = load_drug(args.infer_drug_dir, args.maxDrug)
            predict = ensemble(i, smiles=smiles, sequence=seq, structure=struct, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)
            df_predict = df_predict.append(pd.DataFrame([predict])) #5-fold tmp
        df_predict.T.to_csv(os.path.splitext(args.infer_out_dir)[0]+"_tmp"+".txt", sep='\t', index=True,header=['CV1','CV2','CV3','CV4','CV5'] )


        df_predict_median = df_predict.median().tolist()
        show_merge(df_predict_median, smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir.replace('.txt','_final.txt'), output_num_max=99999,minWinSize=4, minSeqLen=40)


if __name__ == '__main__':
    main()







