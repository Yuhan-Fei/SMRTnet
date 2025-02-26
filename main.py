"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import torch, os, argparse, copy
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from model import SmrtNet
from utils import fix_seed, make_directory ,save_dict, dgl_collate_func_ds, shuffle_dataset, get_kfold_data, get_kfold_data_target, get_kfold_data_drug_fix, get_kfold_data_drug_fix_best_final, GradualWarmupScheduler
from loader import data_process_loader
from loop import train, valid, test, bench
from infer import *
from tensorboardX import SummaryWriter

from dgllife.utils import smiles_to_bigraph, smiles_to_complete_graph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='SmrtNet')

    # Data options
    parser.add_argument("--do_train", action='store_true', help="Whether to run training with cross validation.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument('--data_dir', type=str, default="./data/SMRTnet-data-demo.txt", help='data path')
    parser.add_argument('--out_dir', type=str, default="./results/model_output", help='output directory')
    parser.add_argument("--do_infer", action='store_true', help="Whether to run infer on the dev set")
    parser.add_argument('--infer_rna_dir', type=str, default="./data/MYC_IRES.txt", help='infer rna directory')
    parser.add_argument('--infer_drug_dir', type=str, default="./data/IHT.txt", help='infer drug directory')
    parser.add_argument('--infer_config_dir', type=str, default="./results/SMRTnet_model/config.pkl", help='infer config directory')
    parser.add_argument('--infer_model_dir', type=str, default="./results/SMRTnet_model/SMRTnet_cvx.pth", help='infer model directory')
    parser.add_argument('--infer_out_dir', type=str, default="./results/results.txt", help='infer output directory')
    parser.add_argument("--do_explain", action='store_true',  help="Whether to run infer on the dev set.")
    parser.add_argument("--do_ensemble", action='store_true',  help="Whether to run infer based on 5 models")
    parser.add_argument("--do_delta", action='store_true',  help="Wild type - mutant")
    parser.add_argument("--do_check", action='store_true',  help="Whether to check input data")
    parser.add_argument("--do_benchmark", action='store_true',  help="Whether to check input data")


    parser.add_argument('--mode', type=str, default="SPU", help='data mode')
    parser.add_argument('--split', type=str, default="drug_fix_best", help='data split method')
    parser.add_argument('--cuda', type=int, default=0, help='number of GPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--kfold', type=int, default=5, help='K-fold cross validation')


    # Training Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--patiences', type=int, default=20, help='early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--lr_bert', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function')


    # Language model parameters
    parser.add_argument('--lm_rna_config', type=str, default="./LM_RNA/parameters.json", help='pretrained hyperparameters .json file path')
    parser.add_argument('--lm_rna_model', type=str, default="./LM_RNA/model_state_dict/rnaall_img0_min30_lr5e5_bs30_2w_7136294_norm1_05_1025_150M_16_rope_fa2_noropeflash_eps1e6_aucgave_1213/epoch_0/LMmodel.pt", help='pretrained model .pt file path')
    parser.add_argument('--lm_mol_config', type=str, default="./LM_Mol/bert_vocab.txt", help='Smiles vocal')
    parser.add_argument('--lm_mol_model', type=str, default="./LM_Mol/pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt", help='pretrained model .ckpt file path')
    parser.add_argument('--lm_ft', type=str, default=['molformer','rnalm'], help='rnalm | molformer') #, ,'molformer'


    # output_layer
    parser.add_argument('--hidden_dim_graph', type=int, default=128, help=' ')
    parser.add_argument('--hidden_dim_rna', type=int, default=128, help=' ')
    parser.add_argument('--hidden_dim_rna_lm', type=int, default=128, help=' ')
    parser.add_argument('--hidden_dim_mol_lm', type=int, default=128, help=' ')


    # Predictor
    parser.add_argument('--cls', nargs='+', type=int, default=[1024, 1024, 1024, 512], help='input requires: 1024 1024 1024 512')

    # Res_CNN
    parser.add_argument('--kernal', type=int, default=7, help=' ')
    parser.add_argument('--pad', type=int, default=3, help=' ')
    parser.add_argument('--channel', type=int, default=16, help=' ')


    # GNN parameters
    parser.add_argument('--gnn_in_feats', type=int, default=74, help=' ')
    parser.add_argument('--gnn_num_layers', type=int, default=3, help=' ')
    parser.add_argument('--gnn_hid_dim_drug', type=int, default=256, help=' ')
    # parser.add_argument('--gnn_activations', type=str, default=F.relu, help=' ')

    # GAT parameters
    parser.add_argument('--gat_num_heads', type=int, default=3, help=' ')
    parser.add_argument('--gat_feat_drops', type=float, default=0.2, help=' ')
    parser.add_argument('--gat_attn_drops', type=float, default=0.2, help=' ')


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


    if  not args.do_test and not args.do_train and not args.do_infer and not args.do_explain and not args.do_ensemble and not args.do_check and not args.do_delta and not args.do_benchmark:
        raise ValueError("At least one of `do_train` or `do_infer` must be True.")

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


    if args.do_train:

        accuracy_kfold=[]
        precision_kfold=[]
        recall_kfold=[]
        f1_kfold=[]
        AUC_kfold=[]
        PRC_kfold=[]
        for i_fold in range(args.kfold):

            print('*' * 52, 'No.', i_fold + 1, '-fold', '*' * 52)

            model = SmrtNet(args).to(device)
            total_params = sum(torch.numel(p) for p in model.parameters())
            print(f"Model total parameters: {total_params:,d}")

            base_params = list(map(id, model.pretrain_bert.parameters()))

            base_params.extend(list(map(id, model.tok_emb.parameters())))
            base_params.extend(list(map(id, model.blocks.parameters())))
            logits_params = filter(lambda p: id(p) not in base_params, model.parameters())

            params_lr = [{"params": logits_params, "lr": args.lr},
                            {"params": model.pretrain_bert.parameters(), "lr": args.lr_bert},
                            {"params": model.tok_emb.parameters(), "lr": args.lr_bert},
                            {"params": model.blocks.parameters(), "lr": args.lr_bert}, ]

            optimizer = optim.Adam(params_lr, betas=(0.9, 0.999), weight_decay=args.wd)

            scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=float(args.epoch), after_scheduler=None)

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_weight))

            if args.tfboard:
                i_fold_name = "CV_" + str(i_fold + 1)
                tfb_dir = make_directory(args.out_dir, i_fold_name)
                writer = SummaryWriter(log_dir=tfb_dir)
            else:
                i_fold_name = "CV_" + str(i_fold + 1)
                tfb_dir = make_directory(args.out_dir, i_fold_name)
                writer = None

            file_results = args.out_dir + '/result_' + str(i_fold_name) + '.txt'
            # print(file_results)
            with open(file_results, 'w') as f:
                hp_attr = '\n'.join(['%s:%s' % item for item in args.__dict__.items()])
                f.write(hp_attr + '\n')

            if (args.split == "random"):
                dataset = shuffle_dataset(dataset, args.seed)
                train_df, valid_df, test_df = get_kfold_data(i_fold, dataset, args.kfold, v=1, random_state=args.seed)
            elif (args.split == "drug"):
                train_df, valid_df, test_df = get_kfold_data_drug_fix(i_fold, dataset, args.kfold, v=1, random_state=args.seed)          
            elif (args.split == "target"):
                train_df, valid_df, test_df = get_kfold_data_target(i_fold, dataset, args.kfold, v=1, random_state=args.seed)
            elif (args.split == "drug_fix_best"):

                train_df, valid_df, test_df = get_kfold_data_drug_fix_best_final(i_fold, dataset, args.kfold, v=1, random_state=args.seed)
            print("Train:" + str(len(train_df)), "Validation:" + str(len(valid_df)), "Test:" + str(len(test_df)))
            data_num = "Train:" + str(len(train_df)), "Validation:" + str(len(valid_df)), "Test:" + str(len(test_df))

            with open(file_results, 'a') as f:
                f.write(str(data_num) + '\n')

            # 11.创建Dataloader
            params_train = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'drop_last': False}
            params_valid = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0, 'drop_last': False}
            params_test = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False,
                        'sampler': SequentialSampler(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args))}

            params_train['collate_fn'] = dgl_collate_func_ds
            params_valid['collate_fn'] = dgl_collate_func_ds
            params_test['collate_fn'] = dgl_collate_func_ds

            train_loader = DataLoader(data_process_loader(train_df.index.values, train_df.Label.values, train_df, args), **params_train)
            valid_loader = DataLoader(data_process_loader(valid_df.index.values, valid_df.Label.values, valid_df, args), **params_valid)
            test_loader = DataLoader(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args), **params_test)

            best_accuracy = 0
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            best_auc = 0
            best_epoch = 0
            bet_loss = 999

            for epoch in range(1, args.epoch + 1):
                # print(device)
                t_met, t_loss = train(args, model, device, train_loader, optimizer, criterion, epoch)

                v_met, v_loss, _, _ = valid(args, model, device, valid_loader, criterion, epoch)

                scheduler.step(epoch)
                lr = scheduler.get_lr()[0]

                model_dir = make_directory(args.out_dir, '')
                model_path = os.path.join(model_dir, "model_" + i_fold_name + "_{}.pth")

                if best_auc < v_met.auc:
                    best_model = copy.deepcopy(model)
                    best_accuracy = v_met.acc
                    best_precision = v_met.pre
                    best_recall = v_met.rec
                    best_f1 = v_met.f1
                    best_auc = v_met.auc
                    best_epoch = epoch
                    # best_loss = val_loss
                    filename = model_path.format("best")
                    torch.save(best_model.state_dict(), filename)
                    save_dict(model_dir, args)
                    # torch.save(best_model.state_dict(), output_model_file)

                if epoch - best_epoch >= args.patiences:
                    print("Early stop at %d, %s " % (best_epoch, "and generate best model!"))
                    break

                ## 14. 显示terminal日志文件
                epoch_len = len(str(args.epoch))
                if (epoch == best_epoch):
                    print_valid_msg = (f'Valid at Epoch:{epoch:>{epoch_len}}, ' +
                                    f'train_loss: {t_met.other[0]:.3f} ' +
                                    f'valid_loss: {v_met.other[0]:.3f} ' +
                                    f'valid_acc: {v_met.acc:.3f} ' +
                                    f'valid_pre: {v_met.pre:.3f} ' +
                                    f'valid_rec: {v_met.rec:.3f} ' +
                                    f'valid_auc: {v_met.auc:.3f} ' +
                                    f'valid_prc: {v_met.prc:.3f} ' +
                                    f'***')
                else:
                    print_valid_msg = (f'Valid at Epoch:{epoch:>{epoch_len}}, ' +
                                    f'train_loss: {t_met.other[0]:.3f} ' +
                                    f'valid_loss: {v_met.other[0]:.3f} ' +
                                    f'valid_acc: {v_met.acc:.3f} ' +
                                    f'valid_pre: {v_met.pre:.3f} ' +
                                    f'valid_rec: {v_met.rec:.3f} ' +
                                    f'valid_auc: {v_met.auc:.3f} ' +
                                    f'valid_prc: {v_met.prc:.3f} '
                                    )

                with open(file_results, 'a') as f:
                    f.write(print_valid_msg + '\n')
                print(print_valid_msg)

                if args.tfboard and writer is not None:
                    writer.add_scalar('loss/train',      t_met.other[0], epoch)
                    writer.add_scalar('accuracy/train',  t_met.acc, epoch)
                    writer.add_scalar('precision/train', t_met.pre, epoch)
                    writer.add_scalar('recall/train',    t_met.rec, epoch)
                    writer.add_scalar('f1/train',        t_met.f1, epoch)
                    writer.add_scalar('AUC/train',       t_met.auc, epoch)
                    writer.add_scalar('PRC/train',       t_met.prc, epoch)
                    writer.add_scalar('loss/valid',      v_met.other[0], epoch)
                    writer.add_scalar('accuracy/valid',  v_met.acc, epoch)
                    writer.add_scalar('precision/valid', v_met.pre, epoch)
                    writer.add_scalar('recall/valid',    v_met.rec, epoch)
                    writer.add_scalar('f1/valid',        v_met.f1, epoch)
                    writer.add_scalar('AUC/valid',       v_met.auc, epoch)
                    writer.add_scalar('PRC/valid',       v_met.prc, epoch)
                    writer.add_scalar('lr', lr, epoch)
            best_model.eval()
            t_met, t_loss, t_Y, t_P = valid(args, best_model, device, train_loader, criterion, epoch)
            v_met, v_loss, v_Y, v_P = valid(args, best_model, device, valid_loader, criterion, epoch)
            met, loss, Y, P = valid(args, best_model, device, test_loader, criterion, epoch)


            best_train_msg = (f'Best_train at Epoch: {best_epoch:d} ' +
                            f'train_loss: {t_met.other[0]:.3f} ' +
                            f'train_accuracy: {t_met.acc:.3f} ' +
                            f'train_precision: {t_met.pre:.3f} ' +
                            f'train_recall: {t_met.rec:.3f} ' +
                            f'train_f1: {t_met.f1:.3f} ' +
                            f'train_AUC: {t_met.auc:.3f} ' +
                            f'train_PRC: {t_met.prc:.3f} ')

            best_valid_msg = (f'Best_valid at Epoch: {best_epoch:d} ' +
                            f'valid_loss: {v_met.other[0]:.3f} ' +
                            f'valid_accuracy: {v_met.acc:.3f} ' +
                            f'valid_precision: {v_met.pre:.3f} ' +
                            f'valid_recall: {v_met.rec:.3f} ' +
                            f'valid_f1: {v_met.f1:.3f} ' +
                            f'valid_AUC: {v_met.auc:.3f} ' +
                            f'valid_PRC: {v_met.prc:.3f} ')

            best_test_msg = (f'Best__test at Epoch: {best_epoch:d} ' +
                            f'test__loss: {met.other[0]:.3f} ' +
                            f'test__accuracy: {met.acc:.3f} ' +
                            f'test__precision: {met.pre:.3f} ' +
                            f'test__recall: {met.rec:.3f} ' +
                            f'valid_f1: {met.f1:.3f} ' +
                            f'test__AUC: {met.auc:.3f} ' +
                            f'test__PRC: {met.prc:.3f} ')


            print("\n")
            print(best_train_msg)
            print(best_valid_msg)
            print(best_test_msg)
            print("\n")

            with open(file_results, 'a') as f:
                f.write('\n' + best_train_msg + '\n')
                f.write(best_valid_msg + '\n')
                f.write(best_test_msg + '\n')

            save_path = make_directory(args.out_dir, i_fold_name)
            #17.记录label和probabilit用于AUC绘制
            t_P = np.squeeze(t_P)
            v_P = np.squeeze(v_P)
            P = np.squeeze(P)
            with open(save_path + "/{}_prediction.txt".format("Train"), 'a') as f:
                for i in range(len(t_Y)):
                    f.write(str(t_Y[i]) + " " + str(t_P[i]) + '\n')
            with open(save_path + "/{}_prediction.txt".format("Valid"), 'a') as f:
                for i in range(len(v_Y)):
                    f.write(str(v_Y[i]) + " " + str(v_P[i]) + '\n')
            with open(save_path + "/{}_prediction.txt".format("Test"), 'a') as f:
                for i in range(len(Y)):
                    f.write(str(Y[i]) + " " + str(P[i]) + '\n')

            accuracy_kfold.append(met.acc)
            precision_kfold.append(met.pre)
            recall_kfold.append(met.rec)
            f1_kfold.append(met.f1)
            AUC_kfold.append(met.auc)
            PRC_kfold.append(met.prc)


        Accuracy_mean, Accuracy_var = np.mean(accuracy_kfold), np.std(accuracy_kfold, ddof=1)
        Precision_mean, Precision_var = np.mean(precision_kfold), np.std(precision_kfold, ddof=1)
        Recall_mean, Recall_var = np.mean(recall_kfold), np.std(recall_kfold, ddof=1)
        F1_mean, F1_var = np.mean(f1_kfold), np.std(f1_kfold, ddof=1)
        AUC_mean, AUC_var = np.mean(AUC_kfold), np.std(AUC_kfold, ddof=1)
        PRC_mean, PRC_var = np.mean(PRC_kfold), np.std(PRC_kfold, ddof=1)

        print("The average k-fold performance of the model:")
        print('Accuracy (std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
        print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
        print('Recall   (std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
        print('F1   (std):{:.4f}({:.4f})'.format(F1_mean, F1_var))
        print('AUC      (std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
        print('PRC      (std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

        with open(args.out_dir + "/results_final.txt", 'w') as f:
            f.write('Accuracy (std): {:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var, ) + " | " + str(accuracy_kfold) + '\n')
            f.write('Precision(std): {:.4f}({:.4f})'.format(Precision_mean, Precision_var) + " | " + str(precision_kfold) + '\n')
            f.write('Recall   (std): {:.4f}({:.4f})'.format(Recall_mean, Recall_var) + " | " + str(recall_kfold) + '\n')
            f.write('F1       (std): {:.4f}({:.4f})'.format(F1_mean, F1_var) + " | " + str(f1_kfold) + '\n')
            f.write('AUC      (std): {:.4f}({:.4f})'.format(AUC_mean, AUC_var) + " | " + str(AUC_kfold) + '\n')
            f.write('PRC      (std): {:.4f}({:.4f})'.format(PRC_mean, PRC_var) + " | " + str(PRC_kfold) + '\n')



    elif args.do_test:

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
        best_test_msg = (#f'Best__test at Epoch: {best_epoch:d} ' +
                         f'test__loss: {met.other[0]:.3f} ' +
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
            model_path=str(args.infer_model_dir)+"/model_CV_"+str(i)+"_best.pth"
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
            model_path=str(args.infer_model_dir)+"/model_CV_"+str(i)+"_best.pth"
            model = load_model(args.infer_config_dir, model_path, args)
            rna, seq, struct = load_rna(os.path.splitext(args.infer_rna_dir)[0]+"_slice"+".txt", args.maxRNA)
            drug, smiles = load_drug(args.infer_drug_dir, args.maxDrug)
            predict = ensemble(i, smiles=smiles, sequence=seq, structure=struct, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)
            df_predict = df_predict.append(pd.DataFrame([predict])) #5-fold tmp
        df_predict.T.to_csv(os.path.splitext(args.infer_out_dir)[0]+"_tmp"+".txt", sep='\t', index=True,header=['CV1','CV2','CV3','CV4','CV5'] )


        df_predict_median = df_predict.median().tolist()
        #breakpoint()
        #show_ensemble(df_predict_median, smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)
        show_merge(df_predict_median, smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir.replace('.txt','_final.txt'), output_num_max=99999,minWinSize=4, minSeqLen=40)
        


    elif args.do_delta:
        make_directory(args.infer_out_dir, '/')
        df_predict = pd.DataFrame()
        rna_slice(inputPath=args.infer_rna_dir,step=1)
        #cheeck length of RNA equal to 31nt, number =2
        for i in range(1,6):
            model_path=str(args.infer_model_dir)+"/model_CV_"+str(i)+"_best.pth"
            model = load_model(args.infer_config_dir, model_path, args)
            rna, seq, struct = load_rna(args.infer_rna_dir, args.maxRNA)
            drug, smiles = load_drug(args.infer_drug_dir, args.maxDrug)
            predict = ensemble(i, smiles=smiles, sequence=seq, structure=struct, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)

            df_predict = df_predict.append(pd.DataFrame([predict])) #5-fold tmp
        """
        split=int(df_predict.shape[1]/2)
        df_predict_wt=df_predict.iloc[:,:split]
        df_predict_mut=df_predict.iloc[:,split:]
        df_predict_mut=df_predict_mut.T.reset_index(drop=True).T
        df_predict_delta = df_predict_wt - df_predict_mut
        df_predict_delta.T.to_csv(os.path.splitext(args.infer_out_dir)[0]+"_tmp"+".txt", sep='\t', index=True,header=['CV1','CV2','CV3','CV4','CV5'] )
        df_predict_median_final = df_predict_delta.median().tolist()
        breakpoint()
        """

        df_predict.T.to_csv(args.infer_out_dir+"/results_tmp.txt", sep='\t', index=True,header=['CV1','CV2','CV3','CV4','CV5'] )
        df_predict_median = df_predict.median().tolist()
        split=int(len(df_predict_median)/2)
        df_predict_median_wt=df_predict_median[:split]
        df_predict_median_mut=df_predict_median[split:]
        df_predict_median_final = [a - b for a, b in zip(df_predict_median_wt, df_predict_median_mut)]
 
        show_delta(df_predict_median, df_predict_median_final, smiles=smiles, sequence=seq, structure=struct, drug_names=drug, target_names=rna, model=model, args=args, result_folder=args.infer_out_dir, output_num_max=99999)
       


    elif args.do_check:
        print("")
        print("Checking the validity of small molecules...")
        smi_df = pd.read_table(args.infer_drug_dir, sep='\t', header=None)
        smi_df.columns = ['CAS','SMILES']
        node_featurizer = CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        transform = partial(smiles_to_bigraph, add_self_loop=True)

        for i in range(smi_df.shape[0]):
            smiles = smi_df.loc[i,'SMILES']
            cas = smi_df.loc[i,'CAS']
            drug_encoding = transform(smiles=smiles, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
            if drug_encoding:
                print(cas+' is valid')
            else:
                print(cas+'\t'+smiles+' is invalid')

        print("")
        print("Checking the validity of RNAs...")
        rna_df = pd.read_table(args.infer_rna_dir, sep='\t', header=None)
        rna_df.columns = ['name','sequence','structure']
        for i in range(rna_df.shape[0]):
            name = rna_df.loc[i,'name']
            sequence = rna_df.loc[i,'sequence']
            structure = rna_df.loc[i,'structure']
            
            if len(sequence)==len(structure):
                if len(sequence)==31:
                    print(name+' RNA is 31nt')
                elif len(sequence)>31:
                    print(name+' RNA is '+str(len(sequence))+'nt and need to slice to 31nt fragments')
                else:
                    print(name+' is too short')
                    
            else:
                print(name+' sequence length is not equal to structure length')


    elif args.do_benchmark:
        make_directory(args.infer_out_dir, "")
        slidingwindow(args.data_dir, args.infer_out_dir)

        test_df = pd.read_table(args.infer_out_dir+"/testInput.txt",sep='\t',header=None)
        test_df = test_df.iloc[:,1:]
        test_df.reset_index(drop=True, inplace=True)
        test_df.columns=["SMILES", "Sequence", "Structure", "Label"]

        for i in range(1,6):
            print('Predicting interactions based on ('+str(i)+'/5) '+ 'model...')
            model_path=str(args.infer_model_dir)+"/model_CV_"+str(i)+"_best.pth"
            model = load_model(args.infer_config_dir, model_path, args)
            params_test = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'drop_last': False,
                        'sampler': SequentialSampler(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args))}
            params_test['collate_fn'] = dgl_collate_func_ds
            test_loader = DataLoader(data_process_loader(test_df.index.values, test_df.Label.values, test_df, args), **params_test)
            model.eval()
            Y, P = bench(args, model, device, test_loader)
            save_path = make_directory(args.infer_out_dir, '')
            P = np.squeeze(P)
            with open(save_path + "/{}_prediction.txt".format("CV_"+str(i)), 'w') as f:
                for i in range(len(Y)):
                    f.write(str(Y[i]) + "\t" + str(P[i]) + '\n')
        print("Merge results based on 5 models...")
        matrix(args.infer_out_dir,args.minWinSize,args.minSeqLen)




if __name__ == '__main__':
    main()
