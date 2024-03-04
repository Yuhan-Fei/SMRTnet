"""
# Author: Yuhan Fei & Jiasheng Zhang
# Created Time :  15 May 2021
# Revised Time :  12 May 2023
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import nn

from transformers import EsmModel as pretrain_bert
from transformers import EsmConfig
from LM_Mol.tokenizer import MolTranBertTokenizer
from LM_Mol.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask as LM
from functools import partial

from utils import normalize, create_var, index_select_ND

device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')



class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        #Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Cross-Att: Key and Value should always come from the same source (Aiming to forcus on), Query comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]
        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        #energy = torch.mul(Q, K)
        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix





class lm_layer(nn.Module):
    def __init__(self, n_embd, n_vocab):
        super().__init__()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class ResidualBlock1D(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1, padding=5, bias=False)
        # self.c2 = nn.Conv1d(planes,   planes*2, kernel_size=7, stride=1,padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock2D(nn.Module):

    def __init__(self, planes, kernel_size=(7, 4), padding=(3, 1), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        # print(out.shape,identity.shape)
        out += identity
        out = self.relu(out)

        return out


class SmrtNet(nn.Module):

    def __init__(self, args):
        super(SmrtNet, self).__init__()
        ## parameters
        self.args = args
        self.input_dim_drug = args.hidden_dim_graph
        self.input_dim_target = args.hidden_dim_rna
        #self.hidden_dims = args.hidden_dims_cls
        self.hidden_dims = args.cls



        self.input_dim_rna_lm = args.hidden_dim_rna_lm
        self.input_dim_mol_lm = args.hidden_dim_mol_lm
        layer_size = len(self.hidden_dims) + 1
        self.fusionMethod = args.fusion

        self.output_dim = self.input_dim_drug + self.input_dim_target + self.input_dim_rna_lm + self.input_dim_mol_lm
        self.output_dim_lm = self.input_dim_rna_lm

        dims = [self.output_dim] + self.hidden_dims + [1]

        if args.mode == "SEQ":
            h_p, h_k = 1, 3
            self.n_features = 4
        elif args.mode == "SPU" or args.mode == "SPUK" or args.mode == "SPUC":
            h_p, h_k = 2, 5
            self.n_features = 5
        elif args.mode == "SPUK2":
            h_p, h_k = 2, 5
            self.n_features = 6
        elif args.mode == "PU":
            h_p, h_k = 1, 3
            self.n_features = 3
        elif args.mode == "PUK":
            h_p, h_k = 1, 3
            self.n_features = 4

        ## drug graph network


        if args.drug_embed == "GCN":
            from dgllife.model.gnn.gcn import GCN
            from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
            self.gnn = GCN(in_feats=args.gnn_in_feats,
                           hidden_feats=[args.gnn_hid_dim_drug] * args.gnn_num_layers,
                           activation=[F.relu] * args.gnn_num_layers,
                           residual= [True] * args.gnn_num_layers,
                           batchnorm=[True] * args.gnn_num_layers,
                           dropout=[0.0] * args.gnn_num_layers
                           )
            gnn_out_feats = self.gnn.hidden_feats[-1]
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, args.hidden_dim_graph)


        elif args.drug_embed == "GAT":
            from dgllife.model.gnn.gat import GAT
            from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
            self.gnn = GAT(in_feats=args.gnn_in_feats,
                           hidden_feats=[args.gnn_hid_dim_drug] * args.gnn_num_layers,
                           num_heads=[args.gat_num_heads] * args.gnn_num_layers,
                           feat_drops=[args.gat_feat_drops] * args.gnn_num_layers,
                           attn_drops=[args.gat_attn_drops] * args.gnn_num_layers,
                           alphas=[0.2] * args.gnn_num_layers,
                           residuals=[True] * args.gnn_num_layers,
                           agg_modes=['flatten'] * args.gnn_num_layers, #mean
                           activations=[F.relu] * args.gnn_num_layers,
                           biases=[None] * args.gnn_num_layers  # True
                           )

            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2 * self.gnn.num_heads[-1], args.hidden_dim_graph)

        elif args.drug_embed == "GATv2":
            from dgllife.model.gnn.gatv2 import GATv2
            from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
            self.gnn = GATv2(in_feats=args.gnn_in_feats,
                             hidden_feats=[args.gnn_hid_dim_drug] * args.gnn_num_layers,
                             activations=[F.relu] * args.gnn_num_layers,
                             num_heads=[args.gat_num_heads] * args.gnn_num_layers,
                             feat_drops=[args.gat_feat_drops] * args.gnn_num_layers,
                             attn_drops=[args.gat_attn_drops] * args.gnn_num_layers,
                             residuals= [True] * args.gnn_num_layers,
                             agg_modes=['flatten'] * args.gnn_num_layers,
                             #alphas=0.2,
                             allow_zero_in_degree=False
                             #biases=biases,
                             #share_weights=share_weights
                           )
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
            self.readout = WeightedSumAndMax(gnn_out_feats)
            self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2 * self.gnn.num_heads[-1], args.hidden_dim_graph)


        elif args.drug_embed == "GIN_M":
            from dgllife.model import load_pretrained
            from dgl.nn.pytorch.glob import AvgPooling
            self.gnn = load_pretrained('gin_supervised_masking')
            self.readout = AvgPooling()
            self.transform = nn.Linear(300, args.hidden_dim_graph)

        elif args.drug_embed == "GIN_C":
            from dgllife.model import load_pretrained
            from dgl.nn.pytorch.glob import AvgPooling

            self.gnn = load_pretrained('gin_supervised_contextpred')
            self.readout = AvgPooling()
            self.transform = nn.Linear(300, args.hidden_dim_graph)

        elif args.drug_embed == "GIN_I":
            from dgllife.model import load_pretrained
            from dgl.nn.pytorch.glob import AvgPooling

            self.gnn = load_pretrained('gin_supervised_infomax')
            self.readout = AvgPooling()
            self.transform = nn.Linear(300, args.hidden_dim_graph)




        elif args.drug_embed == "GSA":
            from dgllife.model.gnn.graphsage import GraphSAGE
            from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
            self.gnn = GraphSAGE(in_feats=args.gnn_in_feats,
                                 hidden_feats=[args.gnn_hid_dim_drug] * args.gnn_num_layers,
                                 activation=[F.relu] * args.gnn_num_layers,
                                 aggregator_type=['mean']  * args.gnn_num_layers #pool, mean, gcn, lstm,
                                 #dropout=[0.2] * args.gnn_num_layers
                           )
            gnn_out_feats = self.gnn.hidden_feats[-1]
            self.readout = WeightedSumAndMax(gnn_out_feats)
            #breakpoint()
            self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, args.hidden_dim_graph)



        elif args.drug_embed == "PAGTN":
            from dgllife.model.gnn import PAGTNGNN
            from dgllife.model.readout import MLPNodeReadout
            self.gnn = PAGTNGNN(node_in_feats=args.pagtn_in_feats,
                                node_out_feats=args.pagtn_out_feats,
                                node_hid_feats=args.pagtn_hid_feats,
                                edge_feats=14,
                                depth=5,
                                nheads=1,
                                dropout=0.1,
                                activation=nn.LeakyReLU(0.2),
                           )

            self.readout = MLPNodeReadout(args.pagtn_out_feats + args.pagtn_in_feats, args.pagtn_out_feats, args.hidden_dim_graph,mode='sum') #max, mean
            #self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, args.hidden_dim_graph)


        elif args.drug_embed == "NeuralFP":
            from dgllife.model.gnn.nf import NFGNN
            from dgllife.model.readout.sum_and_max import SumAndMax

            self.gnn = NFGNN(in_feats=args.gnn_in_feats,
                             hidden_feats=[args.gnn_hid_dim_drug] * args.gnn_num_layers,
                             max_degree=10,
                             activation=[F.relu] * args.gnn_num_layers
                             )
            gnn_out_feats = self.gnn.gnn_layers[-1].out_feats
            self.node_to_graph = nn.Linear(gnn_out_feats, 128)
            self.predictor_activation = torch.tanh

            self.readout = SumAndMax()
            self.transform = nn.Linear(128 * 2, args.hidden_dim_graph)


        elif args.drug_embed == "AttentiveFP":
            from dgllife.model.gnn import AttentiveFPGNN
            from dgllife.model.readout import AttentiveFPReadout

            self.gnn = AttentiveFPGNN(  # node_feat_size=node_feat_size,
                # edge_feat_size=edge_feat_size,
                # num_layers=num_layers,
                # graph_feat_size=graph_feat_size
                node_feat_size=39,
                edge_feat_size=11,
                num_layers=args.gnn_num_layers,
                graph_feat_size=256)

            self.readout = AttentiveFPReadout(feat_size=256, num_timesteps=2)  # num_timesteps=config['attentivefp_num_timesteps']

            self.transform = nn.Linear(256, args.hidden_dim_graph)

        elif args.drug_embed == "MPNN_c":
            from dgl.nn.pytorch import Set2Set
            from dgllife.model.gnn import MPNNGNN
            self.gnn = MPNNGNN(node_in_feats=74, #fix
                               node_out_feats=64,
                               edge_in_feats=13, #fix
                               edge_hidden_feats=128,
                               num_step_message_passing=6)

            self.readout = Set2Set(input_dim=64,
                                   n_iters=6,
                                   n_layers=3)
            self.transform = nn.Sequential(
                nn.Linear(2 * 64, 64),
                nn.ReLU(),
                nn.Linear(64, args.hidden_dim_graph)
            )
        elif args.drug_embed == "MPNN_a":
            from dgl.nn.pytorch import Set2Set
            from dgllife.model.gnn import MPNNGNN
            self.gnn = MPNNGNN(node_in_feats=39, #fix
                               node_out_feats=64,
                               edge_in_feats=11, #fix
                               edge_hidden_feats=128,
                               num_step_message_passing=6)

            self.readout = Set2Set(input_dim=64,
                                   n_iters=6,
                                   n_layers=3)
            self.transform = nn.Sequential(
                nn.Linear(2 * 64, 64),
                nn.ReLU(),
                nn.Linear(64, args.hidden_dim_graph)
            )

        elif args.drug_embed == "Weave":
            from dgllife.model.gnn import WeaveGNN
            from dgllife.model.readout import WeaveGather

            self.gnn = WeaveGNN(node_in_feats=27,
                                edge_in_feats=7,
                                num_layers=2,
                                hidden_feats=50,
                                activation=F.relu)
            self.node_to_graph = nn.Sequential(nn.Linear(50, 128),nn.Tanh(),nn.BatchNorm1d(128))
            self.readout = WeaveGather(node_in_feats=128,
                                       gaussian_expand=True,
                                       gaussian_memberships=None,
                                       activation=nn.Tanh())
            self.transform = nn.Linear(128, args.hidden_dim_graph)




        ## pretrained RNA model
        if args.lm_readout != 'None':
            self.mid_dim_bert = 768
            configuration_pretrain = EsmConfig.from_pretrained(args.lm_rna_config)
            #breakpoint()
            self.pretrain_bert = pretrain_bert(configuration_pretrain)

            #self.Dense = nn.Linear(configuration_pretrain.hidden_size, configuration_pretrain.hidden_size)
            #self.ActFn = nn.GELU()
            #self.LayerNorm = nn.LayerNorm(configuration_pretrain.hidden_size, eps=configuration_pretrain.layer_norm_eps)
            #self.Transfer = nn.Linear(configuration_pretrain.hidden_size, self.mid_dim_bert)
            #breakpoint()
            dict_para_pretrain = torch.load(args.lm_rna_model, map_location=torch.device('cuda:'+str(args.cuda)))

            for name_, para_ in self.pretrain_bert.state_dict().items():
                if 'esm.' + name_ in dict_para_pretrain.keys():
                    self.pretrain_bert.state_dict()[name_].copy_(dict_para_pretrain['esm.' + name_])
            for para in self.pretrain_bert.parameters():
                if 'rnalm' in args.lm_ft:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            """
            self.pretrain_bert, alphabet = fm.pretrained.rna_fm_t12()
            self.batch_converter = alphabet.get_batch_converter()
            self.Dense = nn.Linear(640,640)
            self.ActFn = nn.GELU()
            self.LayerNorm = nn.LayerNorm(640, eps=1e-12)
            self.Transfer = nn.Linear(640,self.mid_dim_bert)
            #self.Transfer = nn.Linear(640,640)
            """
        else:
            pass
        #breakpoint()
        ## how to add pretrained embeddings
        if args.lm_readout == 'JustAvgPooling':
            self.avgpool_e = nn.AvgPool2d((31, self.n_features))
        elif args.lm_readout == 'GlobalPooling_mlp':
            self.mlp_e = nn.Linear(31, self.input_dim_rna_lm)  # 31
            #############################################
            #self.mlp_e1 = nn.Linear(configuration_pretrain.hidden_size, 256)
            #self.mlp_e2 = nn.Linear(256, self.input_dim_rna_lm)
            #self.mlp_d1 = nn.Linear(768, 256)
            #self.mlp_d2 = nn.Linear(256, self.input_dim_mol_lm)
            #self.mlp_vPs = nn.Linear(configuration_pretrain.hidden_size, self.input_dim_mol_lm)  # 31

            ###################l##########################
            """
            self.mlp_l1 = nn.Linear(640, 256)
            self.mlp_l2 = nn.Linear(256, 128)
            self.mlp_l3 = nn.Linear(768, 256)
            self.mlp_l4 = nn.Linear(3968, 1024)
            self.mlp_l5 = nn.Linear(1024, 512)
            self.mlp_l6 = nn.Linear(512, 128)
            self.mlp_d1 = nn.Linear(768, 256)
            self.mlp_d2 = nn.Linear(256, self.input_dim_mol_lm)
            """
            #################ll########################
            self.mlp_l1 = nn.Linear(640, 256)
            self.mlp_l2 = nn.Linear(256, 128)
            self.mlp_l3 = nn.Linear(768, 256)
            self.mlp_l4 = nn.Linear(3968, 1024)
            self.mlp_l5 = nn.Linear(1024, 512)
            self.mlp_l6 = nn.Linear(512, 128)
            self.mlp_d1 = nn.Linear(768, 256)
            self.mlp_d2 = nn.Linear(256, 128)
            self.mlp_d3 = nn.Linear(128, 64)
            self.mlp_d4 = nn.Linear(8192, 2048)
            self.mlp_d5 = nn.Linear(2048, 1024)
            self.mlp_d6 = nn.Linear(1024, 512)
            self.mlp_d7 = nn.Linear(512, self.input_dim_mol_lm)

            self.mlp_l7 = nn.Linear(640, 128)
            self.mlp_l8 = nn.Linear(3968, 128)

        elif args.lm_readout == 'CNN_mlp':
            self.CNN = nn.Conv1d(31, 16, kernel_size=7)
            self.CNN2 = nn.Conv1d(16, 1, kernel_size=5)
            self.mlp_c = nn.Linear(630, self.input_dim_rna_lm)
        elif args.lm_readout == 'earlyCat':
            self.output_dim -= self.input_dim_rna_lm
            dims = [self.output_dim] + self.hidden_dims + [1]
            self.n_features = self.n_features + self.mid_dim_bert
            # self.P_Encoding = nn.Embedding(12,768)
    
        ###################################20231115_yh_linear############################################
        elif self.args.lm_readout == 'rna_linear':
            self.mlp_l1 = nn.Linear(640, 512)
            self.mlp_l2 = nn.Linear(512, 128)
            self.mlp_l3 = nn.Linear(3968, 1024)
            self.mlp_l4 = nn.Linear(1024, self.input_dim_rna_lm) 
            
            #rna_linear_simple
            #self.mlp_l1 = nn.Linear(640, 128)
            #self.mlp_l2 = nn.Linear(3968, self.input_dim_rna_lm)
    
        else:
            self.output_dim -= self.input_dim_rna_lm
            dims = [self.output_dim] + self.hidden_dims + [1]


        ## RNA ResNet_CNN module
        # h_p, h_k = 0, 1
        # self.n_features = 1

        # h_p, h_k = 1,2
        # self.n_features = 2

        # 9-4,7-3,5-2,3-1 for 31bp
        kernal = args.kernal
        pad = args.pad
        base_channel = args.channel

        self.conv = Conv2d(1, base_channel, kernel_size=(kernal, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(kernal, h_k), padding=(pad, h_p))  #
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fct = nn.Linear(base_channel * 4 * 8, self.input_dim_target)

        ###dropout
        self.dropout = nn.Dropout(0.1)


        ## fingerprints
        if args.drug_embed2 == "erg":
            input_dim = 315
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "maccs":
            input_dim = 167
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "pubchem":
            input_dim = 881
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "morgan":
            input_dim = 2586
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "daylight":
            input_dim = 2048
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "mordred":
            input_dim = 1613
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "descriptors":
            input_dim = 208
            hidden_dims_lst = [1024, 256, 128]  # output_dim=64
        elif args.drug_embed2 == "molformer":
            tokenizer = MolTranBertTokenizer('./LM_Mol/bert_vocab.txt')
            n_vocab = len(tokenizer.vocab)
            self.tok_emb = nn.Embedding(n_vocab, 768)
            self.drop_molformer = nn.Dropout(0.1)
            builder = rotate_builder.from_kwargs(
                n_layers=12,
                n_heads=12,
                query_dimensions=768 // 12,
                value_dimensions=768 // 12,
                feed_forward_dimensions=768,
                attention_type='linear',
                #attention_type='full',
                feature_map=partial(GeneralizedRandomFeatures, n_dims=32),
                activation='gelu',
            )
            self.blocks = builder.get()
            self.lang_model = lm_layer(768, n_vocab)

            mol_para_pretrain = torch.load(args.lm_mol_model, map_location=torch.device('cuda:'+str(args.cuda)))
            mol_para_pretrain = mol_para_pretrain['state_dict']

            for name_, para_ in self.blocks.state_dict().items():
                if 'blocks.' + name_ in mol_para_pretrain.keys():
                    self.blocks.state_dict()[name_].copy_(mol_para_pretrain['blocks.' + name_])
            for name_, para_ in self.tok_emb.state_dict().items():
                if 'tok_emb.' + name_ in mol_para_pretrain.keys():
                    self.tok_emb.state_dict()[name_].copy_(mol_para_pretrain['tok_emb.' + name_])
            for name_, para_ in self.lang_model.state_dict().items():
                if 'lang_model.' + name_ in mol_para_pretrain.keys():
                    self.lang_model.state_dict()[name_].copy_(mol_para_pretrain['lang_model.' + name_])



            self.mlp_vDs = nn.Linear(768, self.input_dim_mol_lm)
            #self.mlp_vDs = nn.Linear(256, self.input_dim_mol_lm)
            """
            mollm_hidden_dims_lst=[1024,256,128]
            mollm_layer_size = len(mollm_hidden_dims_lst) + 1
            mollm_dims = [768] + mollm_hidden_dims_lst + [self.input_dim_mol_lm]
            self.mlp_vDs = nn.ModuleList(
                [nn.Linear(mollm_dims[i], mollm_dims[i + 1]) for i in range(mollm_layer_size)])
                
            """

            """   
            #drug_linear

            self.mlp_d1 = nn.Linear(768, 512)  # 31
            self.mlp_d2 = nn.Linear(512, self.input_dim_mol_lm)  # 31
            self.mlp_d3 = nn.Linear(128*int(self.input_dim_mol_lm), 2048)  # 31 #1024
            self.mlp_d4 = nn.Linear(2048, self.input_dim_mol_lm)  # 31
            """

            #drug_linear_simple
            self.mlp_d1 = nn.Linear(768, self.input_dim_mol_lm)  # 31
            self.mlp_d2 = nn.Linear(16384, self.input_dim_mol_lm)  # 31
  


            for para in self.blocks.parameters():
                if 'molformer' in args.lm_ft:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        else:
            self.output_dim -= self.input_dim_fingerprint
            dims = [self.output_dim] + self.hidden_dims + [1]

        if args.drug_embed2 == 'None' or args.drug_embed2 == 'none' or args.drug_embed2 == '':
            pass
        elif args.drug_embed2 == 'molformer':
            pass
        else:
            finger_layer_size = len(hidden_dims_lst) + 1
            finger_dims = [input_dim] + hidden_dims_lst + [self.input_dim_fingerprint]
            self.predictor = nn.ModuleList(
                [nn.Linear(finger_dims[i], finger_dims[i + 1]) for i in range(finger_layer_size)])

        ## fusion method
            
        if self.fusionMethod == 'concat':
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1
        
        elif self.fusionMethod == 'concat_ln':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1
        
        elif self.fusionMethod == 'concat_ln_lm_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

         
        elif self.fusionMethod == 'concat_ln_lm_randn':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'concat_ln_str_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

         
        elif self.fusionMethod == 'concat_ln_str_randn':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'concat_ln_drug_str_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'concat_ln_drug_seq_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'concat_ln_rna_str_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'concat_ln_rna_seq_zeros':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1





        elif self.fusionMethod == 'BNCAB2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB2_1':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB2_2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB2_3':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB3_1':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB3_2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB4_0':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB4_1':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB4_2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB4_3':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB5_1':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB5_2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB5_3':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm+self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB6_1':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB6_2':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB6_3':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'BNCAB6_4':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm+self.output_dim_lm, 2, 0.1)
            self.LN = nn.LayerNorm(self.output_dim_lm)
            self.BN = nn.BatchNorm1d(self.output_dim_lm+self.output_dim_lm)
            dims = [self.output_dim_lm+self.output_dim_lm] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_PU':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmresnet':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmln':
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_c1':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*2] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_c2':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*2] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer2':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            #self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            #self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c1':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            #self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            #self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*2] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1
        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c1_rmresnet':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            #self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            #self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*2] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c2':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            #self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            #self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            #self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*2] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'LBNCAB4_v3_sequence':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_structure':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_g1l2':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_g1l1':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 2, 0.1)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_chresnet':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1

        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_rmresnet':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153':
            self.LN1 = nn.LayerNorm(self.output_dim_lm)
            self.LN2 = nn.LayerNorm(self.output_dim_lm)
            self.LN3 = nn.LayerNorm(self.output_dim_lm)
            self.LN4 = nn.LayerNorm(self.output_dim_lm)
            self.LN5 = nn.LayerNorm(self.output_dim_lm)
            self.LN6 = nn.LayerNorm(self.output_dim_lm)
            self.LN7 = nn.LayerNorm(self.output_dim_lm)
            self.LN8 = nn.LayerNorm(self.output_dim_lm)
            self.LN9 = nn.LayerNorm(self.output_dim_lm*2)
            self.LN10 = nn.LayerNorm(self.output_dim_lm*2)
            self.BN = nn.BatchNorm1d(self.output_dim_lm*4)
            self.attentionBlock1 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock2 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock3 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock4 = AttentionBlock(self.output_dim_lm, 2, 0.1)
            self.attentionBlock5 = AttentionBlock(self.output_dim_lm*4, 8, 0.3)
            self.attentionBlock6 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            self.attentionBlock7 = AttentionBlock(self.output_dim_lm*2, 4, 0.5)
            dims = [self.output_dim_lm*4] + self.hidden_dims[1:] + [1]
            layer_size = layer_size - 1


        else:
            pass

        ###Fully-connected
        self.fc = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])


    def forward(self, v_D, de_idx, de_mask, v_P, re_input_ids, re_atten_mask, get_attention=False):

        if self.args.drug_embed == "GIN_M":
            node_feats = [v_D.ndata['atomic_number'], v_D.ndata['chirality_type']]
            edge_feats = [v_D.edata['bond_type'], v_D.edata['bond_direction_type']]
            node_feats = self.gnn(v_D, node_feats, edge_feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)


        elif self.args.drug_embed == "GIN_C":
            node_feats = [v_D.ndata['atomic_number'], v_D.ndata['chirality_type']]
            edge_feats = [v_D.edata['bond_type'], v_D.edata['bond_direction_type']]
            node_feats = self.gnn(v_D, node_feats, edge_feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)


        elif self.args.drug_embed == "GIN_I":
            node_feats = [v_D.ndata['atomic_number'], v_D.ndata['chirality_type']]
            edge_feats = [v_D.edata['bond_type'], v_D.edata['bond_direction_type']]
            node_feats = self.gnn(v_D, node_feats, edge_feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)

        elif self.args.drug_embed == "GAT":
            feats = v_D.ndata['h']
            node_feats = self.gnn(v_D, feats)
            # node_feats = self.gnn(v_D, feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)

        elif self.args.drug_embed == "GATv2":
            feats = v_D.ndata['h']
            node_feats = self.gnn(v_D, feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)
            # node_feats,attn = self.gnn(v_D, feats, get_attention=True)
            # graph_feats = self.readout(v_D, node_feats)
            # v_D, _ = self.transform(graph_feats)


        elif self.args.drug_embed == "GCN":
            feats = v_D.ndata['h']
            node_feats = self.gnn(v_D, feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)



        elif self.args.drug_embed == "GSA":
            feats = v_D.ndata['h']
            node_feats = self.gnn(v_D, feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)


        elif self.args.drug_embed == "PAGTN":

            node_feats = v_D.ndata['h']
            edge_feats = v_D.edata['e']
            #breakpoint()
            atom_h = self.gnn(v_D, node_feats, edge_feats)
            atom_h = torch.cat([atom_h, node_feats], dim=1)
            v_D = self.readout(v_D, atom_h)
            #v_D = self.transform(graph_feats)
            #breakpoint()

        elif self.args.drug_embed == "NeuralFP":
            feats = v_D.ndata['h']
            node_feats = self.gnn(v_D, feats)
            node_feats = self.node_to_graph(node_feats)
            graph_feats = self.readout(v_D, node_feats)
            graph_feats = self.predictor_activation(graph_feats)
            v_D = self.transform(graph_feats)

        elif self.args.drug_embed == "AttentiveFP":
            node_feats = v_D.ndata['h']
            edge_feats = v_D.edata['e']
            node_feats = self.gnn(v_D, node_feats, edge_feats)
            graph_feats = self.readout(v_D, node_feats, False)
            v_D = self.transform(graph_feats)


        elif self.args.drug_embed == "MPNN_a" or self.args.drug_embed == "MPNN_c":
            node_feats = v_D.ndata['h']
            edge_feats = v_D.edata['e']
            node_feats = self.gnn(v_D, node_feats, edge_feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)


        elif self.args.drug_embed == "Weave":
            node_feats = v_D.ndata['h']
            edge_feats = v_D.edata['e']
            node_feats = self.gnn(v_D, node_feats, edge_feats, node_only=True)
            node_feats = self.node_to_graph(node_feats)
            graph_feats = self.readout(v_D, node_feats)
            v_D = self.transform(graph_feats)

        if get_attention:
            node_feats.retain_grad()


        if self.args.drug_embed2 == "maccs":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
            v_De = v_De.view(v_De.shape[0], v_De.shape[2])
        elif self.args.drug_embed2 == "erg":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "daylight":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "morgan":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "pubchem":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "mordred":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "descriptors":
            for i, l in enumerate(self.predictor):
                v_De = F.relu(l(v_De))
        elif self.args.drug_embed2 == "molformer":
            token_embeddings = self.tok_emb(de_idx.long())  # each index maps to a (learnable) vector
            if get_attention:
                token_embeddings.retain_grad()
            #x = self.drop_molformer(token_embeddings)
            #x = self.blocks(x, length_mask=LM(de_mask.sum(-1)))
            #x = x.view(x.shape[0], x.shape[2], x.shape[1])


            v_De = self.drop_molformer(token_embeddings)
            v_De = self.blocks(v_De, length_mask=LM(de_mask.sum(-1)))
            
            #drug_linear
            """
            # B*L * 768
            v_De = F.dropout(self.mlp_d1(v_De), 0.3, training=self.training) # B * L * 512
            v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training) # B * L * 128
            #breakpoint()
            v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1]) # B * 128 * L
            m = torch.nn.AdaptiveMaxPool1d(128)
            v_De = m(v_De) #B * 128 * 128
            v_De = v_De.view(v_De.shape[0], v_De.shape[2] * v_De.shape[1]) #B * 16384
            v_De = F.dropout(self.mlp_d3(v_De), 0.3, training=self.training) #B * 2048
            v_De = F.dropout(self.mlp_d4(v_De), 0.1, training=self.training)  # B * 128
            """

            #drug_linear_simple
            # B*L * 768
            v_De = F.dropout(self.mlp_d1(v_De), 0.1, training=self.training) # B * L * 128
            #breakpoint()
            v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1]) # B * 128 * L
            m = torch.nn.AdaptiveMaxPool1d(128)
            v_De = m(v_De) #B * 128 * 128
            v_De = v_De.view(v_De.shape[0], v_De.shape[2] * v_De.shape[1]) #B * 16384
            v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training)  # B * 128



            #v_De = self.gpool(x)
            #v_De = v_De.view(v_De.shape[0], v_De.shape[1])
            #v_De = self.mlp_vDs(v_De)
            #v_De = F.dropout(v_De, 0.1, training=self.training)


            #breakpoint()
            ###############################################################
            #v_De = self.blocks(x, length_mask=LM(de_mask.sum(-1)))
            #x = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1])  #B*768*99
            #v_De = self.gpool(x)
            #v_De = v_De.view(v_De.shape[0], v_De.shape[1])
            #v_De = self.mlp_vDs(v_De)
            #v_De = F.dropout(v_De, 0.1, training=self.training)

            ###################################20231111_yh############################################
            #breakpoint()
            #v_De = self.gpool(x)  # B * L * 1
            #v_De = v_De.view(v_De.shape[0], v_De.shape[1])  # B * L
            #m = torch.nn.AdaptiveMaxPool1d(768)
            #v_De = m(v_De)
            #v_De = F.dropout(self.mlp_vDs(v_De), 0.1, training=self.training)  # B * 128




        else:
            v_De = torch.FloatTensor([]).to(device)

        if self.args.lm_readout != 'None':
            v_Pe, v_Pe_embeddings = self.pretrain_bert(**{'input_ids': re_input_ids.long(), 'attention_mask':re_atten_mask})
            #breakpoint()
            if get_attention:
                v_Pe_embeddings.retain_grad()

            v_Pe = v_Pe.last_hidden_state
            v_Pe = v_Pe[:, 1:, :]
            """
            seq_str_list = [('RNA_%s'%(i), ei) for i,ei in enumerate(v_Pe)]
            _, _, input_ids = self.batch_converter(seq_str_list)
            v_Pe = self.pretrain_bert(input_ids.to(device), repr_layers=[12])["representations"][12]
            v_Pe = v_Pe[:,1:-1,:]
            """
        else:
            pass

        if self.args.lm_readout == 'JustAvgPooling':
            v_Pe = self.avgpool_e(v_Pe)
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[2])
        elif self.args.lm_readout == 'GlobalPooling_mlp':

            #v_Pe = self.gpool(v_Pe)
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1])
            #v_Pe = F.dropout(self.mlp_e(v_Pe), 0.1, training=self.training)


            ###################################20231111_yh############################################
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[2], v_Pe.shape[1])  # B * 640 * 31
            #v_Pe = self.gpool(v_Pe) # B * 640 * 1
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1])  # B * 640
            #v_Pe = self.mlp_vPs(v_Pe) # B * 128


            ###################################20231112_js############################################

            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[2], v_Pe.shape[1])  # B * 640 * 31
            #v_Pe = self.gpool(v_Pe) # B * 640 * 1
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1])  # B * 640
            #v_Pe = F.dropout(self.mlp_e1(v_Pe), 0.3, training=self.training)
            #v_Pe = F.dropout(self.mlp_e2(v_Pe), 0.1, training=self.training)

            #v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1])  # B * 768 * L
            #v_De = self.gpool(v_De) # B * 768 * 1
            #v_De = v_De.view(v_De.shape[0], v_De.shape[1])  # B * 768
            #v_De = F.dropout(self.mlp_d1(v_De), 0.3, training=self.training)
            #v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training)


            ###################################20231111_yh############################################

            #v_Pe = self.gpool(v_Pe) # B * 31 * 1
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1])  # B * 31
            #v_Pe = F.dropout(self.mlp_e(v_Pe), 0.1, training=self.training) # B * 128

            #v_De = self.gpool(x)  # B * L * 1
            #v_De = v_De.view(v_De.shape[0], v_De.shape[1])  # B * L
            #m = torch.nn.AdaptiveMaxPool1d(256)
            #v_De = m(v_De)
            #v_De = F.dropout(self.mlp_vDs(v_De), 0.1, training=self.training)  # B * 128


            ###################################20231114_yh_ll############################################
            """
            #v_Pe  # B * 31 * 640
            v_Pe = F.dropout(self.mlp_l1(v_Pe), 0.3, training=self.training) # B * 31 * 256
            v_Pe = F.dropout(self.mlp_l2(v_Pe), 0.1, training=self.training) # B * 31 * 128
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1]* v_Pe.shape[2]) #B * 3968
            v_Pe = F.dropout(self.mlp_l4(v_Pe), 0.3, training=self.training)  # B * 1024
            v_Pe = F.dropout(self.mlp_l5(v_Pe), 0.1, training=self.training)  # B * 512
            v_Pe = F.dropout(self.mlp_l6(v_Pe), 0.1, training=self.training)  # B * 128


            v_De = self.blocks(x, length_mask=LM(de_mask.sum(-1)))
            v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1])  # B * 768 * L
            v_De = self.gpool(v_De) # B * 768 * 1
            v_De = v_De.view(v_De.shape[0], v_De.shape[1])  # B * 768
            v_De = F.dropout(self.mlp_d1(v_De), 0.3, training=self.training)
            v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training)
            """

            ###################################20231114_yh_ll2############################################
            """
            #v_Pe  # B * 31 * 640
            v_Pe = F.dropout(self.mlp_l1(v_Pe), 0.3, training=self.training) # B * 31 * 256
            v_Pe = F.dropout(self.mlp_l2(v_Pe), 0.1, training=self.training) # B * 31 * 128
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1]* v_Pe.shape[2]) #B * 3968
            v_Pe = F.dropout(self.mlp_l4(v_Pe), 0.3, training=self.training)  # B * 1024
            v_Pe = F.dropout(self.mlp_l5(v_Pe), 0.1, training=self.training)  # B * 512
            v_Pe = F.dropout(self.mlp_l6(v_Pe), 0.1, training=self.training)  # B * 128


            v_De = self.blocks(x, length_mask=LM(de_mask.sum(-1)))
            #v_De  # B * L * 768
            v_De = F.dropout(self.mlp_d1(v_De), 0.3, training=self.training) # B * L * 256
            v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training) # B * L * 128
            v_De = F.dropout(self.mlp_d3(v_De), 0.1, training=self.training)  # B * L * 64
            #breakpoint()
            v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1]) # B * 64 * L

            m = torch.nn.AdaptiveMaxPool1d(128)
            v_De = m(v_De) #B * 64 * 128
            v_De = v_De.view(v_De.shape[0], v_De.shape[2] * v_De.shape[1]) #B * 8192
            v_De = F.dropout(self.mlp_d4(v_De), 0.3, training=self.training)  # B * 2048
            v_De = F.dropout(self.mlp_d5(v_De), 0.1, training=self.training)  # B * 1024
            v_De = F.dropout(self.mlp_d6(v_De), 0.1, training=self.training)  # B * 512
            v_De = F.dropout(self.mlp_d7(v_De), 0.1, training=self.training)  # B * 128
            """

            ###################################20231114_yh_ll3############################################

            #v_Pe  # B * 31 * 640
            v_Pe = F.dropout(self.mlp_l7(v_Pe), 0.1, training=self.training) # B * 31 * 128
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1]* v_Pe.shape[2]) #B * 3968
            v_Pe = F.dropout(self.mlp_l8(v_Pe), 0.1, training=self.training)  # B * 128


            v_De = self.blocks(x, length_mask=LM(de_mask.sum(-1)))
            v_De = v_De.view(v_De.shape[0], v_De.shape[2], v_De.shape[1])  # B * 768 * L
            v_De = self.gpool(v_De) # B * 768 * 1
            v_De = v_De.view(v_De.shape[0], v_De.shape[1])  # B * 768
            v_De = F.dropout(self.mlp_d1(v_De), 0.3, training=self.training)
            v_De = F.dropout(self.mlp_d2(v_De), 0.1, training=self.training)

        elif self.args.lm_readout == 'rna_linear':
            ###################################20231115_yh_linear############################################
            v_Pe  # B * 31 * 640
            v_Pe = F.dropout(self.mlp_l1(v_Pe), 0.3, training=self.training) # B * 31 * 512
            v_Pe = F.dropout(self.mlp_l2(v_Pe), 0.1, training=self.training)  # B * 31 * 128
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1]* v_Pe.shape[2]) #B * 3968
            v_Pe = F.dropout(self.mlp_l3(v_Pe), 0.3, training=self.training)  # B * 1024
            v_Pe = F.dropout(self.mlp_l4(v_Pe), 0.1, training=self.training)  # B * 128

            #rna_linear_simple
            #v_Pe = F.dropout(self.mlp_l1(v_Pe), 0.1, training=self.training) # B * 31 * 512
            #v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[1]* v_Pe.shape[2]) #B * 3968
            #v_Pe = F.dropout(self.mlp_l2(v_Pe), 0.1, training=self.training)  # B * 1024



        elif self.args.lm_readout == 'CNN_mlp':
            v_Pe = self.CNN(v_Pe)
            v_Pe = self.CNN2(v_Pe)
            v_Pe = v_Pe.view(v_Pe.shape[0], v_Pe.shape[2])
            v_Pe = F.dropout(self.mlp_c(v_Pe), 0.3, training=self.training)
        else:
            v_Pe = torch.FloatTensor([]).to(device)

        x = self.conv(v_P)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(z * x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        v_P = self.fct(x)

        # concatenate and classify
        if self.fusionMethod == 'concat':
            #v_D, v_De, v_P, v_Pe = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_ln':
            v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)
        
        elif self.fusionMethod == 'concat_sequence':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_D=torch.zeros(v_D.shape[0],v_D.shape[1]).to(device) #rand, zeros
            v_P=torch.zeros(v_P.shape[0],v_P.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_structure':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)


        elif self.fusionMethod == 'concat_lm_zeros':
            v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)


        elif self.fusionMethod == 'concat_ln_lm_zeros':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            #v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            #v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_De=torch.randn(v_De.shape[0],v_De.shape[1]).to(device)
            v_Pe=torch.randn(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_ln_lm_randn':
            v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_De=torch.randn(v_De.shape[0],v_De.shape[1]).to(device)
            v_Pe=torch.randn(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)



        elif self.fusionMethod == 'concat_ln_str_zeros':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            #v_D=torch.zeros(v_D.shape[0],v_D.shape[1]).to(device)
            #v_P=torch.zeros(v_P.shape[0],v_P.shape[1]).to(device)
            v_D=torch.randn(v_D.shape[0],v_D.shape[1]).to(device)
            v_P=torch.randn(v_P.shape[0],v_P.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_ln_str_randn':
            v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_D=torch.randn(v_D.shape[0],v_D.shape[1]).to(device)
            v_P=torch.randn(v_P.shape[0],v_P.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)


        elif self.fusionMethod == 'concat_ln_drug_str_zeros':
            v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_D=torch.zeros(v_D.shape[0],v_D.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_ln_drug_seq_zeros':
            v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)


        elif self.fusionMethod == 'concat_drug_str_randn':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            #v_D=torch.zeros(v_D.shape[0],v_D.shape[1]).to(device)
            v_D=torch.randn(v_D.shape[0],v_D.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)

        elif self.fusionMethod == 'concat_drug_seq_randn':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            #v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            v_De=torch.randn(v_De.shape[0],v_De.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)



        elif self.fusionMethod == 'concat_ln_rna_str_zeros':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_P=torch.zeros(v_P.shape[0],v_P.shape[1]).to(device)
            #v_P=torch.randn(v_P.shape[0],v_P.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)




        elif self.fusionMethod == 'concat_ln_rna_seq_zeros':
            #v_D, v_De, v_P, v_Pe = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)


        elif self.fusionMethod == 'BNCAB2':
            #breakpoint()
            CAB_g = self.attentionBlock1(v_Pe, v_Pe, v_De)
            CAB_l = self.attentionBlock2(v_P, v_P, v_D)
            CAB = torch.cat((CAB_g,CAB_l), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB2_1':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_seq = self.attentionBlock1(v_Pe, v_Pe, v_De)
            CAB_str = self.attentionBlock2(v_P, v_P, v_D)
            CAB_seq = self.LN5(CAB_seq)
            CAB_str = self.LN6(CAB_str)
            CAB = torch.cat((CAB_seq,CAB_str), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB2_2':
            #v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq = self.attentionBlock1(v_Pe, v_Pe, v_De)
            CAB_str = self.attentionBlock2(v_P, v_P, v_D)
            CAB_seq = self.LN1(CAB_seq)
            CAB_str = self.LN2(CAB_str)
            CAB = torch.cat((CAB_seq,CAB_str), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB2_3':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_seq = self.attentionBlock1(v_Pe, v_Pe, v_De)
            CAB_str = self.attentionBlock2(v_P, v_P, v_D)
            #CAB_seq = self.LN(CAB_seq)
            #CAB_str = self.LN(CAB_str)
            CAB = torch.cat((CAB_seq,CAB_str), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB3_1':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_seq = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq,CAB_str = self.LN5(CAB_seq), self.LN6(CAB_str)
            CAB = torch.cat((CAB_seq,CAB_str), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB3_2':

            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_seq = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq,CAB_str = self.LN5(CAB_seq), self.LN6(CAB_str)

            CAB = torch.cat((CAB_seq,CAB_str), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB4_0':

            #v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            #CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1,CAB_str1,CAB_seq2,CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB4_1':

            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN5(CAB_seq1), self.LN6(CAB_str1), self.LN7(CAB_seq2), self.LN8(CAB_str2)

            CAB = torch.cat((CAB_seq1,CAB_str1,CAB_seq2,CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB4_2':

            #v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1,CAB_str1,CAB_seq2,CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB4_3':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            #CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1,CAB_str1,CAB_seq2,CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB5_1':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_D, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_De, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_P, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_Pe, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1, CAB_str1, CAB_seq2, CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB5_2':

            #v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_D, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_De, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_P, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_Pe, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1, CAB_str1, CAB_seq2, CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB5_3':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_D, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_De, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_P, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_Pe, v_D, v_D)

            #CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB = torch.cat((CAB_seq1, CAB_str1, CAB_seq2, CAB_str2), 1)
            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock3(v_f, v_f, v_f)



        elif self.fusionMethod == 'BNCAB6_1':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB_seq3 = self.attentionBlock3(CAB_seq2, CAB_seq1, CAB_seq1)

            CAB_str3 = self.attentionBlock3(CAB_str2, CAB_str1, CAB_str1)

            CAB_seq3, CAB_str3= self.LN(CAB_seq3), self.LN(CAB_str3)

            CAB = torch.cat((CAB_seq3,CAB_str3), 1)

            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock4(v_f, v_f, v_f)


        elif self.fusionMethod == 'BNCAB6_2':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB_seq3 = self.attentionBlock3(CAB_seq1, CAB_seq2, CAB_seq2)

            CAB_str3 = self.attentionBlock3(CAB_str1, CAB_str2, CAB_str2)

            CAB_seq3, CAB_str3= self.LN(CAB_seq3), self.LN(CAB_str3)

            CAB = torch.cat((CAB_seq3,CAB_str3), 1)

            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock4(v_f, v_f, v_f)


        elif self.fusionMethod == 'BNCAB6_3':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_De, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_D, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_P, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB_seq3 = self.attentionBlock3(CAB_str1, CAB_seq2, CAB_seq2)

            CAB_str3 = self.attentionBlock3(CAB_str2, CAB_seq1, CAB_seq1)

            CAB_seq3, CAB_str3= self.LN(CAB_seq3), self.LN(CAB_str3)

            CAB = torch.cat((CAB_seq3,CAB_str3), 1)

            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock4(v_f, v_f, v_f)

        elif self.fusionMethod == 'BNCAB6_4':

            v_Pe, v_De, v_P, v_D = self.LN(v_Pe), self.LN(v_De), self.LN(v_P), self.LN(v_D)
            CAB_seq1 = self.attentionBlock1(v_P, v_Pe, v_Pe)
            CAB_str1 = self.attentionBlock2(v_Pe, v_P, v_P)

            CAB_seq2 = self.attentionBlock1(v_D, v_De, v_De)
            CAB_str2 = self.attentionBlock2(v_De, v_D, v_D)

            CAB_seq1,CAB_str1,CAB_seq2,CAB_str2 = self.LN(CAB_seq1), self.LN(CAB_str1), self.LN(CAB_seq2), self.LN(CAB_str2)

            CAB_seq3 = self.attentionBlock3(CAB_seq1, CAB_seq2, CAB_seq2)

            CAB_str3 = self.attentionBlock3(CAB_str2, CAB_str2, CAB_seq2)

            CAB_seq3, CAB_str3= self.LN(CAB_seq3), self.LN(CAB_str3)

            CAB = torch.cat((CAB_seq3,CAB_str3), 1)

            v_f = self.BN(CAB)
            v_f = v_f + self.attentionBlock4(v_f, v_f, v_f)

        elif self.fusionMethod == 'LBNCAB4_v3':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)

        elif self.fusionMethod == 'LBNCAB4_v3_PU':
            v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)
############################################################################################################
        elif self.fusionMethod == 'LBNCAB4_v3_rmresnet':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1) 

            v_f1 = self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = self.attentionBlock5(v_f, v_f, v_f)

        elif self.fusionMethod == 'LBNCAB4_v3_rmln':
            #v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = self.attentionBlock4(v_D, v_P, v_P)
            #CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = self.attentionBlock7(v_f2, v_f2, v_f2)

            #v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1, v_f2), 1)
            v_f = self.BN(v_f)
            v_f = self.attentionBlock5(v_f, v_f, v_f)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f = torch.cat((CAB_g1, CAB_g2, CAB_l1, CAB_l2), 1)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_c1':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f2, v_f2)
            #v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f1, v_f1)

            v_f = self.LN9(v_f1)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_c2':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f2, v_f2)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f1, v_f1)

            v_f = self.LN10(v_f2)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer2':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f = torch.cat((CAB_g1, CAB_g2, CAB_l1, CAB_l2), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)



        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c1':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g1, CAB_l1 = self.LN5(CAB_g1),self.LN7(CAB_l1)

            v_f = torch.cat((CAB_g1, CAB_l1), 1)


        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c1_rmresnet':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = self.attentionBlock2(v_P, v_D, v_D)
            CAB_g1, CAB_l1 = self.LN5(CAB_g1),self.LN7(CAB_l1)

            v_f = torch.cat((CAB_g1, CAB_l1), 1)


        elif self.fusionMethod == 'LBNCAB4_v3_rmlayer3_2_c2':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g2,  CAB_l2=  self.LN6(CAB_g2),  self.LN8(CAB_l2)

            v_f = torch.cat((CAB_g2, CAB_l2), 1)


        elif self.fusionMethod == 'LBNCAB4_v3_sequence':
            #breakpoint()
            v_D=torch.randn(v_D.shape[0],v_D.shape[1]).to(device) #rand, zeros
            v_P=torch.randn(v_P.shape[0],v_P.shape[1]).to(device)

            #v_D=torch.zeros(v_D.shape[0],v_D.shape[1]).to(device)
            #v_P=torch.zeros(v_P.shape[0],v_P.shape[1]).to(device)
            #breakpoint()
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)


        elif self.fusionMethod == 'LBNCAB4_v3_structure':
            #v_De=torch.randn(v_De.shape[0],v_De.shape[1]).to(device)
            #v_Pe=torch.randn(v_Pe.shape[0],v_Pe.shape[1]).to(device)


            v_De=torch.zeros(v_De.shape[0],v_De.shape[1]).to(device)
            v_Pe=torch.zeros(v_Pe.shape[0],v_Pe.shape[1]).to(device)

            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)


        elif self.fusionMethod == 'LBNCAB4_v3_RNA_all_smi':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)


        elif self.fusionMethod == 'LBNCAB4_v3_RNA_all_str':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)




        elif self.fusionMethod == 'LBNCAB4_v3_drug_all_seq':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)



        elif self.fusionMethod == 'LBNCAB4_v3_drug_all_structure':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)



############################################################################################################


        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_g1l2':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)


            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            #v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            #v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)

        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_g1l1':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            #v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            #v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)







        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_chresnet':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_De + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_D + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_Pe + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_P + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)

        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153_rmresnet':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 =  self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)


        elif self.fusionMethod == 'LBNCAB4_v3_head248_dp153':
            v_Pe, v_De, v_P, v_D = self.LN1(v_Pe), self.LN2(v_De), self.LN3(v_P), self.LN4(v_D)
            CAB_g1 = v_Pe + self.attentionBlock1(v_Pe, v_De, v_De)
            CAB_l1 = v_P + self.attentionBlock2(v_P, v_D, v_D)
            CAB_g2 = v_De + self.attentionBlock3(v_De, v_Pe, v_Pe)
            CAB_l2 = v_D + self.attentionBlock4(v_D, v_P, v_P)
            CAB_g1, CAB_g2, CAB_l1, CAB_l2= self.LN5(CAB_g1), self.LN6(CAB_g2), self.LN7(CAB_l1), self.LN8(CAB_l2)

            v_f1 = torch.cat((CAB_g1, CAB_g2), 1)
            v_f2 = torch.cat((CAB_l1, CAB_l2), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l2), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l1), 1)

            #v_f1 = torch.cat((CAB_g1, CAB_l1), 1)
            #v_f2 = torch.cat((CAB_g2, CAB_l2), 1)           

            v_f1 = v_f1 + self.attentionBlock6(v_f1, v_f1, v_f1)
            v_f2 = v_f2 + self.attentionBlock7(v_f2, v_f2, v_f2)

            v_f1n, v_f2n = self.LN9(v_f1), self.LN10(v_f2)

            v_f = torch.cat((v_f1n, v_f2n), 1)
            v_f = self.BN(v_f)
            v_f = v_f + self.attentionBlock5(v_f, v_f, v_f)







        else:
            v_f = torch.cat((v_D, v_De, v_P, v_Pe), 1)




        for i, l in enumerate(self.fc):
            if i == (len(self.fc) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        if get_attention:
            return v_f, token_embeddings, v_Pe_embeddings, node_feats
        else:
            return v_f


