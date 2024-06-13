# SmrtNet: Predicting small molecule and RNA target interactions using deep neural network

RNA-targeting small molecules can bind RNA to regulate its function, providing a promising alternative approach for the treatment of human disease. However, the development of this field has been hindered by the lack of effective experimental probing and computational prediction approaches. Here, we present SmrtNet (Predicting small molecule and RNA target interactions using deep neural network), a deep learning-based framework designed to predict interactions between small molecules and RNA targets. SmrtNet extracts sequence and structural information from both small molecule and RNA using **large-scale language models** and **attention-based neural networks**, and integrates mutual information among different encoders using **a attention-based feature fusion strategy**.

<p align="center"><img src="figs/workflow.png" width=100% /></p>

## Cite us
If you found this package useful, please cite [our paper](xxx):
```
Yuhan Fei, SmrtNet: Predicting small molecule and RNA target interactions using deep neural network, xxx, 2024
```
## Contact us
Please contact me if you are interested in my work and look for academic collaboration. (yuhan_fei@outlook.com)

## Table of contents
- [Getting Started](#getting-started)
- [Repo Structure](#repo-structure)
- [Datasets](#datasets)
  - Dataset construction
  - Traing, validation, and test data of SmrtNet
  - RNA sequence datasets for RNA language model (RNA-LM)
  - SMILES datasets for chemical language model (MoLFormer)
  - Small molecule library for inference
  - RNA target format for inference
  - Small molecule format for inference
- [Usage](#usage)
  - How to train your own model 
  - How to test the performance of model
  - How to inference based on our pre-trained model
- [Example](#example)
  - High throughput drug screening
  - Binding site prediction
  - Key functional group prediction
  - Fragment-based drug design
  - Transcriptome-wide target discovery
- [Web Server](#web-server)
- [Referenced Repos](#referenced-repos)
- [Copyright and License](#copyright-and-license)
- [Disclaimer](#disclaimer)

## Getting started

### Requirements
```bash
 - Python 3.8.10
 - PyTorch 1.10.1+cu111 (Visit https://pytorch.org/get-started/previous-versions/ to install the correct version)
 - torchvision 0.11.2+cu111
 - torchaudio 0.10.1
 - pytorch-fast-transformers 0.3.0
 - pytorch-lightning 1.1.5
 - transformers 4.40.2
 - dgllife 0.3.2
 - dgl-cuda10.2 0.9.1post1
 - rdkit 2022.3.5
 - scipy 1.10.1
 - pandas 1.2.4
 - scikit-learn 0.24.2
 - numpy 1.20.3
 - prettytable 3.10.0
 - notebook 7.1.3
 - tebsnrboardX 2.6.2.2
 - prefetch-generator 1.0.3
 - matplotlib 3.7.5
 - seaborn 0.13.2
```

### Install via Conda
```bash
git clone https://github.com/Yuhan-Fei/SmrtNet.git
cd SmrtNet
conda env create -f environment.yml
conda activate SmrtNet
```

### Install via Pip
```bash
conda create -n smrtnet python=3.8.10
conda activate smrtnet
pip install smrtnet
```


### Install via Conda and Pip
```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install prettytable notebook tensorboardX prefetch_generator numpy==1.20.3 transformers pytorch-fast-transformers==0.3.0 pytorch-lightning==1.1.5 rdkit==2022.3.5 scipy==1.10.1 pandas==1.2.4 scikit-learn==0.24.2 
pip install matplotlib seaborn xsmiles
conda install dgllife -c conda-forge
conda install dglteam::dgl-cuda10.2
```

### Disable CPU in fast-transformer
```bash
DIR=/home/yuhan/anaconda3/envs/smrtnet2/lib/python3.8/site-packages/fast_transformers
sed -i '9,10 s/^/#/' ${DIR}/causal_product/__init__.py
sed -i '24 s/^/#/' ${DIR}/causal_product/__init__.py
sed -i '28 s/^/#/' ${DIR}/causal_product/__init__.py
sed -i '10,11 s/^/#/' ${DIR}/aggregate/__init__.py
sed -i '10 s/^/#/' ${DIR}/clustering/hamming/__init__.py
sed -i '10 s/^/#/' ${DIR}/hashing/__init__.py
sed -i '10,14 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '28,34 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '54 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '58 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '100 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '104 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '153 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '157 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '227 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '231 s/^/#/' ${DIR}/sparse_product/__init__.py
sed -i '8,11 s/^/#/' ${DIR}/local_product/__init__.py
sed -i '30 s/^/#/' ${DIR}/local_product/__init__.py
sed -i '34 s/^/#/' ${DIR}/local_product/__init__.py
sed -i '72 s/^/#/' ${DIR}/local_product/__init__.py
sed -i '76 s/^/#/' ${DIR}/local_product/__init__.py

```

## Repo Structure:
After adding all our examples, the repo has the following structure:
```
├── LM_Mol
|  └── Pretrained
|      └── hparams.yaml
|      └── checkpoints
|          └── N-Step-Checkpoint_3_30000.ckpt
|  └── attention_layer.py
|  └── bert_vocab.txt
|  └── rotary.py
|  └── rotate_builder.py
|  └── tokenizer.py
|
├── LM_RNA
|   └── model_state_dict
|      └── rnaall_img0_min30_lr5e5_bs30_2w_7136294_norm1_05_1025_150M_16_rope_fa2_noropeflash_eps1e6_aucgave_1213
|          └── epoch_0
|              └── LMmodel.pt
|   └── parameters.json
|   └── pretrained
|   └── activations.py
|   └── bert.py
|   └── doc.py
|   └── modeling_utils.py
|   └── transformers_output.py
|
├── data
|
├── datasets
|   └── 5.5.1.3_10A_norm_simple_unk_single_O4_ion_ext_new_III_2_42.txt
|
├── dataset_cv_best
|
├── img_log
|   └── 5.5.1.3_10A_norm_simple_unk_single_O4_ion_ext_new_III_2_42.txt
|
├── results
|   └── 20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix
|
├── explain.py
├── infer.py
├── loader.py
├── loop.py
├── main.py
├── model.py
└── utils.sh
```

## Datasets

### Prepare the datasets

Scripts and pipeline are in preparing, currently, we provide xxx samples data in *.txt format for training and testing SmrtNet.


```
# Training data
./SmrtNet/data
```


### Format of input RNA target

The length of RNA should >31nt, and the sequence length should equal to the structure length. Data are split by tab and ignore the first header row.

| RNA  | Sequence | Structure |
|-----------------|-------------|-------------|
| MYC | GGGGGGGCUUCGCCUCUGGCCCAGCCCUCCC | (((((((((..(((...)))..))))))))) |
| Pre-miR21 | GAUGUUGACUGUUGAAUCUCAUGGCAACACC | (.(((((.((((.(.....)))))))))).) |


### Format of input small molecule:
The SMILES of small molecule should meet the requirement of RDkit. Data are split by tab and ignore the first header row.

| CAS | SMILES |
|-----------------|-------------|
| 20013-75-6 | CC1=CC2=C(CC1)C(=CC3=C2C(=CO3)C)C|
| 90-33-5| CC1=CC(=O)OC2=C1C=CC(=C2)O|
| 55084-08-7 | COC1=CC=CC(=C1C2=CC(=O)C3=C(C(=C(C(=C3O2)OC)OC)OC)O)O | 





## Usage

### Network Architecture

<p align="center"><img src="figs/architecture.png" width=100% /></p>


### Check your input data format

Check input format
```
python main.py --do_check
```

### Training 

To train the model from scratch, run
```python
python main.py --do_train \
               --in_dir ./dataset/5.5.1.3_10A_norm_simple_unk_single_O4_ion_ext_new_III_2_42.txt \
               --out_dir=./results/20240521_seqstr_benchmark \
               --cuda 0 \
               --batch_size 32 \
               --epoch 100\
               --patiences 20 \
 ```
where you replace `in_dir` with the directory of the data file you want to use, you will load your own data for the training. Hyper-parameters could be tuned in xxx. For available training options, please take a look at `main.py --help`.

To monitor the training process, add option `--tfboard` in `main.py`, and view page at http://localhost:6006 using tensorboard:
```
python main.py --do_train --tfboard
```

### Evaluation
For evaluation of the models, we provide the script `eval.sh`. You can run it using
```
python main.py --do_test
```


### Inference
For inference data (the same format as the *.tsv file used in [Datasets](#datasets)) using the trained models, we provide the script `infer.sh`. You can run it using
```
python main.py --do_ensemble
```

For evaluation of the ensemble models, 
```
python main.py --do_infer
```

### Interpretability
For computing high attention regions using the trained models, we provide the script `har.sh`. You can run it using
```
python main.py --do_explain
```

## Example

### Case Study 1: N small molecules vs N RNA target:

<details>
   <summary>Click here for the code!</summary>
 
```python

DIR=/data2/feiyuhan/SmrtNet_v3.2_yh/results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix
INPUTPATH=/data2/feiyuhan/SmrtNet_v3.2_yh

cd /data2/feiyuhan/SmrtNet_v3.2_yh

python main.py --do_ensemble --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir ${INPUTPATH}/data/ensemble --infer_rna_dir ${INPUTPATH}/data/rna.txt --infer_drug_dir ${INPUTPATH}/data/drug.txt

 ```
</details>

### Case Study 2: benchmark evalutation:

<details>
   <summary>Click here for the code!</summary>
 
```python

INPUTPATH=/data2/feiyuhan/SmrtNet_v3.2_yh
DIR=/data2/feiyuhan/SmrtNet_v3.2_yh/results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix

python main.py --do_benchmark --cuda 0 --data_dir ${INPUTPATH}/demo/ours_v3.txt --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir /data2/feiyuhan/SmrtNet_v3.2_yh/results/benchmark

```
</details>

### Case Study 3: Use multiple GPU

<details>
   <summary>Click here for the code!</summary>
  
```python


CV=2
nohup python main.py --do_infer --cuda 6 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results_case/screenDrug/screen_all_20240517/results_all_screen_${CV}_DL.txt --infer_rna_dir ${INPUTPATH}/drug_like/experiment_6_target/all.txt \
    --infer_drug_dir ${INPUTPATH}/drug_like/drug_like_molecules/all_databaseI_drug_iso.txt &

CV=3
nohup python main.py --do_infer --cuda 7 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results_case/screenDrug/screen_all_20240517/results_all_screen_${CV}_DL.txt --infer_rna_dir ${INPUTPATH}/drug_like/experiment_6_target/all.txt \
    --infer_drug_dir ${INPUTPATH}/drug_like/drug_like_molecules/all_databaseI_drug_iso.txt &

CV=4
nohup python main.py --do_infer --cuda 8 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results_case/screenDrug/screen_all_20240517/results_all_screen_${CV}_DL.txt --infer_rna_dir ${INPUTPATH}/drug_like/experiment_6_target/all.txt \
    --infer_drug_dir ${INPUTPATH}/drug_like/drug_like_molecules/all_databaseI_drug_iso.txt &

CV=5
nohup python main.py --do_infer --cuda 9 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results_case/screenDrug/screen_all_20240517/results_all_screen_${CV}_DL.txt --infer_rna_dir ${INPUTPATH}/drug_like/experiment_6_target/all.txt \
    --infer_drug_dir ${INPUTPATH}/drug_like/drug_like_molecules/all_databaseI_drug_iso.txt &

```
</details>

### Case Study 4: Binding site prediction:

<details>
   <summary>Click here for the code!</summary>
  
```python

DIR=/data2/feiyuhan/SmrtNet_v3.2_yh/results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix
INPUTPATH=/data2/feiyuhan/SmrtNet_v3.2_yh

cd /data2/feiyuhan/SmrtNet_v3.2_yh

python main.py --do_explain --cuda 2 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} \
    --infer_out_dir /data2/feiyuhan/SmrtNet_v3.2_yh/results/MYC --infer_rna_dir ${INPUTPATH}/data/rna.txt \
    --infer_drug_dir ${INPUTPATH}/data/drug.txt --smooth_steps 3

jupyter-lab --no-browser --port 1111
/data2/feiyuhan/SmrtNet_v3.2_yh/xsmiles.ipynb

```
</details>

### Case Study 4: Key functional group prediction

<details>
   <summary>Click here for the code!</summary>
  
```python

DIR=/data2/feiyuhan/SmrtNet_v3.2_yh/results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix
INPUTPATH=/data2/feiyuhan/SmrtNet_v3.2_yh

cd /data2/feiyuhan/SmrtNet_v3.2_yh

python main.py --do_explain --cuda 2 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} \
    --infer_out_dir /data2/feiyuhan/SmrtNet_v3.2_yh/results/MYC --infer_rna_dir ${INPUTPATH}/data/rna.txt \
    --infer_drug_dir ${INPUTPATH}/data/drug.txt --smooth_steps 3

jupyter-lab --no-browser --port 1111
/data2/feiyuhan/SmrtNet_v3.2_yh/xsmiles.ipynb
```
</details>


### Case Study 5: Fragment-based design

<details>
   <summary>Click here for the code!</summary>
  
```python

cd /data2/feiyuhan/SmrtNet_v3.2_yh
python main.py --do_delta --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir ${INPUTPATH}/data/delta --infer_rna_dir ${INPUTPATH}/data/rna2.txt --infer_drug_dir ${INPUTPATH}/data/drug.txt


Draw linkers for small molecule using [OPENBABEL](https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html)
```
</details>

### Case Study 6: transcript-wide analysis (RNA targets more than 31nt)

<details>
   <summary>Click here for the code!</summary>
  
```python

  python main.py --do_ensemble

```
</details>


## Web Server
We also provide a website [http://smrtnet.zhanglab.net/](http://101.6.120.41:9990/drug/) to predict and visualize the interactions between small molecule and RNA.
<p align="center"><img src="figs/webserver.png" width=100% /></p>

## Referenced Repos
1. MoLFormer
2. CNN
3. ResNet
4. GAT
5. Transformer
6. OPENBABEL
7. DSSR

## Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.



## Disclaimer
The prediction of SmrtNet should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.
