"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time v0:  12 May 2023
# Revised Time v1:  22 Feb 2024
# Revised Time v2:  29 May 2024
"""

import pandas as pd
import numpy as np
import random
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from dgllife.utils import smiles_to_bigraph, smiles_to_complete_graph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

smiles_char = ['@', '/', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2',
               '5', '4', '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G',
               'F', 'I', 'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T',
               'W', 'V', 'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g',
               'f', 'i', 'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']  #'\\'

nucleic_char = ['A', 'U', 'C', 'G'] #4

structure_char = ['.', '(',')']  # 3

enc_protein = OneHotEncoder().fit(np.array(nucleic_char).reshape(-1, 1))
enc_structure = OneHotEncoder().fit(np.array(structure_char).reshape(-1, 1))
enc_structure2 =  LabelEncoder().fit(np.array(structure_char))

class data_process_loader(data.Dataset):
    def __init__(self, list_IDs, labels, df, args):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.args = args
        self.df = df
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.fc2 = partial(smiles_to_complete_graph, add_self_loop=True)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        drug_encoding = self.df.iloc[index]['SMILES']
        label_encoding = np.int64(self.labels[index])
        drug_encoding_smi = np.zeros((1,))
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        drug_encoding = self.fc(smiles=drug_encoding, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer)

        if self.args.mode == "SEQ":
            sequence_endocing = self.df.iloc[index]['Sequence']
            sequence_endocing = [*sequence_endocing]
            sequence_endocing = sequence_OneHot(sequence_endocing)
            target_encoding = np.expand_dims(sequence_endocing, axis=2).transpose([2, 1, 0])

        elif self.args.mode=="SPU":
            sequence_endocing = self.df.iloc[index]['Sequence']
            sequence_endocing = [*sequence_endocing]
            structure_encoding = self.df.iloc[index]['Structure']
            structure_encoding = [*structure_encoding]
            sequence_endocing = sequence_OneHot(sequence_endocing)
            structure_encoding = structure_Label(structure_encoding)
            target_encoding = np.vstack((sequence_endocing,structure_encoding))
            target_encoding = np.expand_dims(target_encoding, axis=2).transpose([2, 1, 0])

        elif self.args.mode=="PU":
            structure_encoding = self.df.iloc[index]['Structure']
            structure_encoding = [*structure_encoding]
            structure_encoding = structure_OneHot(structure_encoding)
            target_encoding = np.expand_dims(structure_encoding, axis=2).transpose([2, 1, 0])


        if self.args.mode=="PU":
                sequence_endocing_trans = self.df.iloc[index]['Structure'].replace("(","A").replace(")","A").replace(".","A")
        else:
            sequence_endocing_trans = self.df.iloc[index]['Sequence']

        embeddings = sequence_endocing_trans

        drug_encoding_smi = self.df.iloc[index]['SMILES']

        return drug_encoding, drug_encoding_smi, target_encoding, embeddings, label_encoding

def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray().T

def sequence_OneHot(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T

def structure_OneHot(x):
    return enc_structure.transform(np.array(x).reshape(-1, 1)).toarray().T

def structure_Label(x):
    return enc_structure2.transform(np.array(x))