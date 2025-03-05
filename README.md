## SMRTnet: Predicting small molecule and RNA target interactions using deep neural network

Small molecules can bind RNAs to regulate their fate and functions, providing promising opportunities for treating human diseases. However, current tools for predicting small molecule-RNA interactions (SRIs) require prior knowledge of RNA tertiary structures, limiting their utility in drug discovery. Here, we present SMRTnet, a deep learning method to predict SRIs based on RNA secondary structure. By integrating **large language models**, **convolutional neural networks**, **graph attention networks**, and **multimodal data fusion**, SMRTnet achieves high performance across multiple experimental benchmarks, substantially outperforming existing state-of-the-art tools.

<p align="center"><img src="figs/workflow.png" width=100% /></p>

## Cite us
If you found this package useful, please cite [our paper](xxx):
```
Yuhan Fei, Pengfei Wang, Jiasheng Zhang, Xinyue Shan, Zilin Cai, Jianbo Ma, Yangming Wang, Qiangfeng Cliff Zhang,

Predicting small molecule and RNA target interactions using deep neural network, under review, 2025.
```
## Contact us
Please contact us if you are interested in our work and look for academic collaboration:  
- Dr. Yuhan Fei, School of Life Sciences, Tsinghua University, Posdoc, yuhan_fei@outlook.com  
- Jiasheng Zhang, School of Life Sciences, Tsinghua University, PhD student, zjs21@mails.tsinghua.edu.cn

## Table of contents
- [Getting Started](#getting-started)
- [Repo Structure](#repo-structure)
- [SMRTnet Architecture](#smrtnet-architecture)
  - Download our pre-trained models from zenodo
- [Datasets](#datasets)
  - Datasets for training
  - RNA target format for inference
  - Small molecule format for inference
  - RNA sequence datasets for RNA language model (RNASwan-seq)
  - SMILES datasets for chemical language model (MoLFormer)
- [Usage](#usage)
  - How to check your input format
  - How to train your own model (Coming soon...)
  - How to test the performance of model
  - How to inference based on the SMRTnet model
  - How to perform model interpretibility
<!-- - [Example](#example)-->
<!-- - [Web Server (Coming soon...)](#web-server)-->
- [Referenced Repos](#referenced-repos)
- [Copyright and License](#copyright-and-license)
- [Disclaimer](#disclaimer)

## Getting started
<!--
### Requirements
```bash
 - Python 3.8.10
 - PyTorch 1.10.1+cu111 
 - torchvision 0.11.2+cu111
 - torchaudio 0.10.1
 - pytorch-fast-transformers 0.3.0
 - pytorch-lightning 1.1.5
 - transformers 4.28.1
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
-->


### Install via Conda and Pip manually (recommend!!!)
```bash
## To set up the SMRTnet environment with CUDA version 12.1,
## please check your CUDA version and install corresponding torch and dgl.
## nvcc --version
conda create -n smrtnet python=3.8.10
conda activate smrtnet
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install prettytable notebook tensorboardX prefetch_generator numpy==1.20.3 transformers==4.28.1 pytorch-lightning==1.1.5 rdkit==2022.3.5 scipy==1.10.1 pandas==1.2.4 scikit-learn==0.24.2 pytorch-fast-transformers==0.3.0
pip install matplotlib seaborn xsmiles
conda install dgllife -c conda-forge
conda install dglteam/label/th24_cu121::dgl
cp ./env/modeling_esm.py ~/anaconda3/envs/smrtnet/lib/python3.8/site-packages/transformers/models/esm/modeling_esm.py ## because we modified this file.


## To set up the SMRTnet environment with CUDA version 11.1
conda create -n smrtnet python=3.8.10
conda activate smrtnet
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install prettytable notebook tensorboardX prefetch_generator numpy==1.20.3 transformers==4.28.1 pytorch-lightning==1.1.5 rdkit==2022.3.5 scipy==1.10.1 pandas==1.2.4 scikit-learn==0.24.2 
pip install matplotlib seaborn xsmiles
conda install dgllife -c conda-forge
conda install dglteam::dgl-cuda10.2
pip install pytorch-fast-transformers==0.3.0 	## If this installation step fails, you can directly copy `./fast_transformers` to your environment directory.
cp ./env/modeling_esm.py ~/anaconda3/envs/smrtnet/lib/python3.8/site-packages/transformers/models/esm/modeling_esm.py ## because we modified this file.
```
Please visit https://pytorch.org/get-started/previous-versions/ to install the correct torch and the correponding [dgl-cuda](https://anaconda.org/dglteam/repo) according to your CUDA version

### Disable CPU in fast-transformer

<details>
   <summary>Click here for the code!</summary>

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
</details>
<!--
### Install via Conda (Coming soon...)
```bash
git clone https://github.com/Yuhan-Fei/SMRTnet.git
cd SMRTnet
conda env create -f environment.yml
conda activate SMRTnet
```
-->

### Install via offline package 
```bash
## download the offline package smrtnet_env.tar.gz and install it, then you can directly source it.
pip install conda-pack
mkdir -p smrtnet_env
tar -xzf ${PATH}/smrtnet_env.tar.gz -C smrtnet_env
source ./smrtnet_env/bin/activate

```

### Install via Pip (Comming soon...)
```bash
conda create -n smrtnet python=3.8.10
conda activate smrtnet
pip install smrtnet
```

### Install via Docker (Comming soon...)
```bash
xxx
```

## Repo Structure:
After adding all our data, the repo has the following structure:

<details>
   <summary>Click here for the code!</summary>

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
|   └── SMRTnet-data-demo.txt
|   └── MYC_RIBOTAC.txt
|   └── MYC_IRES.txt
|
├── dataset_cv_best
|
├── img_log
|   └── 1.png
|   └── 2.png
|   └── 3.png
|   └── acgu.npz
|   └── dot_bracket.npz
|
├── results
|   └── SMRTNet_model
|
├── explain.py
├── infer.py
├── loader.py
├── loop.py
├── main.py
├── model.py
└── utils.py

```
</details>

## SMRTnet Architecture

<p align="center"><img src="figs/architecture.png" width=100% /></p>

### Download our pre-trained models from zenodo （Required）
Please download models, including RNA language model (LM_RNA), chemical language model (LM_Mol), and SMRTnet(SMRTnet_model), from link below, and place these models into the SMRTnet folder:  
https://zenodo.org/records/14715564

<p align="center"><img src="figs/zenodo.png" width=100% /></p>
  
## Datasets

### Datasets for training

Download and extract the dataset in datasets folder: SMRTnet-data.txt

The original SMRTnet dataset can be found at https://www.rcsb.org/ and process by custom scripts.

The format of data for training is show as follow:

| SMILES | Sequence | Structure | label |
|-----------------|-------------|-------------|-------------|
| CC1=CC2=C(CC1)C(=CC3=C2C(=CO3)C)C | GGGGGGGCUUCGCCUCUGGCCCAGCCCUCCC | (((((((((..(((...)))..))))))))) | 1 |
| CC1=CC(=O)OC2=C1C=CC(=C2)O | GAUGUUGACUGUUGAAUCUCAUGGCAACACC | (.(((((.((((.(.....)))))))))).) | 0 | 


### Format of input RNA target for inference:

The length of RNA should >=31nt, and the sequence length should equal to the structure length. Data are split by tab and ignore the first header row.  


| RNA  | Sequence | Structure |
|-----------------|-------------|-------------|
| MYC_IRES | GAGGGGGCUUCGCCUCUGGCCCAGCCCUCUC | (((((((((..(((...)))..))))))))) |


### Format of input small molecule  for inference:
The SMILES of small molecule should meet the requirement of RDkit. Data are split by tab and ignore the first header row.

| CAS | SMILES |
|-----------------|-------------|
| 3902-71-4 | CC1=CC(=O)OC2=C1C=C3C=C(OC3=C2C)C |
| 149-91-7 | C1=C(C=C(C(=C1O)O)O)C(=O)O |
| 132201-33-3 | C1=CC=C(C=C1)C(C(C(=O)O)O)NC(=O)C2=CC=CC=C2 | 

### RNA sequence datasets for RNA language model (RNASwan-seq)

The dataset used for RNA language model was compiled from 7 sources: the European Nucleotide Archive, NCBI's nucleotide database, GenBank, Ensembl, RNAcentral, CSCD2, and GreeNC 2.0, encompassing a total of 470 million RNA sequences. We de-duplicated with 100% sequence similarity using MMSeqs2, resulting in about 214 million unique RNA sequences


### SMILES datasets for chemical language model (MoLFormer)

Datasets are available at https://ibm.box.com/v/MoLFormer-data

More details can be found in https://github.com/IBM/molformer


## Usage

The training of SMRTnet requires **~10G** of GPU memory (with batch_size = 32),  
while the testing process needs approximately **3G** of GPU memory.

### Check your input data format

Check input format
```
python main.py --do_check
```

### Training 

To train the model from scratch, run (Coming soon...)

```python
python main.py --do_train
```
where you replace `in_dir` with the directory of the data file you want to use, you will load your own data for the training. Hyper-parameters could be tuned in xxx. For available training options, please take a look at `main.py --help`. To monitor the training process, add option `--tfboard` in `main.py`, and view page at http://localhost:6006 using tensorboard
<!--
We provide the example scripts to train the model from scratch:

```python
python main.py --do_train \
               --data_dir=./data/SMRTnet-data-demo.txt \
               --cuda 0 \
               --batch_size 16 \
               --out_dir=./results/benchmark
```
-->
<p align="center"><img src="figs/demo1.png" width=100% /></p>  

### Evaluation
For evaluation of the models, we provide the script `eval.sh`. You can run it using
```
python main.py --do_test
```
We provide the example scripts to test the model:

```python
python main.py --do_test \
               --data_dir=./data/SMRTnet-data-demo.txt \
               --infer_config_dir ${DIR}/config.pkl \
               --infer_model_dir ${DIR}/SMRTnet_cv1.pth \
               --cuda 0 \
               --batch_size 16 \
               --out_dir=./results/benchmark
```
This case represents the results of the model from the 1-fold CV (SMRTnet_cv1.pth).  
To obtain the results for other folds, the infer_model_dir parameter needs to be modified to SMRTnet_cv2.pth, SMRTnet_cv3.pth, SMRTnet_cv4.pth, and SMRTnet_cv5.pth, respectively.

<p align="center"><img src="figs/demo2.png" width=100% /></p>  

### Inference
For inference data (the same format as the *.tsv file used in [Datasets](#datasets)) using the 5 models from 5-fold cross-validation (CV) based on ensemble scoring strategy  
<p align="center"><img src="figs/scoring.png" width=100% /></p>  

You can run the inference using:  
```
python main.py --do_ensemble
```
<!--
we provide the script `infer.sh`. 
or 

```
python main.py --do_infer
```
The difference between do_ensemble and do_infer is whether multiple GPUs are used at the same time.
-->
We provide the example scripts to perform inference of model:
```python
DIR=./results/SMRTnet_model

cd ${WorkDir}

python main.py --do_ensemble --cuda 0 \
               --infer_config_dir ${DIR}/config.pkl \
               --infer_model_dir ${DIR} \
               --infer_out_dir ./data/ensemble \
               --infer_rna_dir ${INPUTPATH}/data/MYC_IRES.txt \
               --infer_drug_dir ${INPUTPATH}/data/MYC_RIBOTAC.txt

```
<p align="center"><img src="figs/demo3.png" width=60% /></p>  

<!--
or 

```python
CV=1
nohup python main.py --do_infer --cuda 0 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results/screenDrug/results_all_screen_${CV}_DL.txt \
    --infer_rna_dir ${INPUTPATH}/data/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/data/all_databaseI_drug_iso.txt &

CV=2
nohup python main.py --do_infer --cuda 1 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results/screenDrug/results_all_screen_${CV}_DL.txt \
    --infer_rna_dir ${INPUTPATH}/data/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/data/all_databaseI_drug_iso.txt &

CV=3
nohup python main.py --do_infer --cuda 2 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/data/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/data/all_databaseI_drug_iso.txt &

CV=4
nohup python main.py --do_infer --cuda 3 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/data/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/data/all_databaseI_drug_iso.txt &

CV=5
nohup python main.py --do_infer --cuda 4 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ${INPUTPATH}/results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/data/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/data/all_databaseI_drug_iso.txt &

```
-->

### Interpretability
For computing high attention regions using the trained models, You can run it using the following scripts and visualize the results in jupyter-notebook
```
python main.py --do_explain
```
We provide the example scripts to perform interpretability of model:

```python
DIR=./results/SMRTnet_model

cd ${WorkDir}

python main.py --do_explain --cuda 0
    --infer_config_dir ${DIR}/config.pkl \
    --infer_model_dir ${DIR} \
    --infer_out_dir ./results/MYC --infer_rna_dir ${INPUTPATH}/data/MYC_IRES.txt \
    --infer_drug_dir ${INPUTPATH}/data/MYC_RIBOTAC.txt --smooth_steps 3

```

<p align="center"><img src="figs/demo4.png" width=60% /></p>  


<!--
## Example

### Case Study 1: Inference: N small molecules vs N RNA target:

<details>
   <summary>Click here for the code!</summary>

```python
DIR=./results/SMRTnet_model

cd ${WorkDir}

python main.py --do_ensemble --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir ./results/ensemble --infer_rna_dir ${INPUTPATH}/data/rna.txt --infer_drug_dir ${INPUTPATH}/data/drug.txt

```

or 

```python
CV=1
nohup python main.py --do_infer --cuda 0 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ./results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/dataset/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/dataset/all_databaseI_drug_iso.txt &

CV=2
nohup python main.py --do_infer --cuda 1 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ./results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/dataset/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/dataset/all_databaseI_drug_iso.txt &

CV=3
nohup python main.py --do_infer --cuda 2 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ./results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/dataset/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/dataset/all_databaseI_drug_iso.txt &

CV=4
nohup python main.py --do_infer --cuda 3 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ./results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/dataset/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/dataset/all_databaseI_drug_iso.txt &

CV=5
nohup python main.py --do_infer --cuda 4 \
    --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR}/model_CV_${CV}_best.pth \
    --infer_out_dir ./results/screenDrug/results_all_screen_${CV}_DL.txt \
	--infer_rna_dir ${INPUTPATH}/dataset/experiment_6_target.txt \
    --infer_drug_dir ${INPUTPATH}/dataset/all_databaseI_drug_iso.txt &

```

</details>


### Case Study 2: Benchmarking: benchmark evalutation:

<details>
   <summary>Click here for the code!</summary>

```python
DIR=./results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix

cd ${WorkDir}

python main.py --do_benchmark --cuda 0 --data_dir ${INPUTPATH}/demo/ours_v3.txt --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir ./results/benchmark

```
</details>

### Case Study 3: transcript-wide analysis (RNA targets more than 31nt)

<details>
   <summary>Click here for the code!</summary>

```python

  python main.py --do_ensemble

```
</details>


### Case Study 4: Binding site prediction:

<details>
   <summary>Click here for the code!</summary>

```python
DIR=./results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix

cd ${WorkDir}

python main.py --do_explain --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} \
    --infer_out_dir ./results/MYC --infer_rna_dir ${INPUTPATH}/data/rna.txt \
    --infer_drug_dir ${INPUTPATH}/data/drug.txt --smooth_steps 3

```
</details>

### Case Study 5: Key functional group prediction

<details>
   <summary>Click here for the code!</summary>

```python
DIR=./results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix

cd ${WorkDir}
python main.py --do_explain --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} \
    --infer_out_dir ./results/MYC --infer_rna_dir ${INPUTPATH}/data/rna.txt \
    --infer_drug_dir ${INPUTPATH}/data/drug.txt --smooth_steps 3

```
</details>


### Case Study 6: Fragment-based design

<details>
   <summary>Click here for the code!</summary>

```python
DIR=./results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix

cd ${WorkDir}
python main.py --do_delta --cuda 0 --infer_config_dir ${DIR}/config.pkl --infer_model_dir ${DIR} --infer_out_dir ./results/delta --infer_rna_dir ${INPUTPATH}/data/rna2.txt --infer_drug_dir ${INPUTPATH}/data/drug.txt

Draw linkers for small molecule using [OPENBABEL](https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html)
```
</details>
-->

<!--
## Web Server
We also provide a website [http://smrtnet.zhanglab.net/](http://101.6.120.41:9990/drug/) to predict and visualize the interactions between small molecule and RNA.
<p align="center"><img src="figs/webserver.png" width=100% /></p>
-->
<!--
## Referenced Repos
1. [MoLFormer](https://github.com/IBM/molformer)
2. CNN: [LeNet](https://doi.org/10.1109/5.726791) and [AlexNet](https://doi.org/10.1145/3065386)
3. [ResNet](https://doi.org/10.48550/arXiv.1512.03385)
4. [GAT](https://doi.org/10.48550/arXiv.1710.10903)
5. [Transformer](https://doi.org/10.48550/arXiv.1706.03762)
6. [OPENBABEL](https://github.com/openbabel/openbabel) and [its web](https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html)
7. [DSSR](http://home.x3dna.org/)
-->
## Referenced Repos
1. MoLFormer: [https://github.com/IBM/molformer](https://github.com/IBM/molformer)
2. Convolutional neural networks: [LeNet](https://doi.org/10.1109/5.726791) and [AlexNet](https://doi.org/10.1145/3065386)
3. Residual neutral networks: [https://doi.org/10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
4. Graph Attention networks: [https://github.com/awslabs/dgl-lifesci](https://github.com/awslabs/dgl-lifesci)
5. Transformer: [https://doi.org/10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
6. OPENBABEL: [https://github.com/openbabel/openbabel](https://github.com/openbabel/openbabel) and [web server](https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html)
7. atomium: [https://github.com/samirelanduk/atomium](https://github.com/samirelanduk/atomium)
8. DSSR: [http://home.x3dna.org/](http://home.x3dna.org/)

## Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.



## Disclaimer
The prediction of SMRTnet should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.
