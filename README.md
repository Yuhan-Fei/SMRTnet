# SmrtNet: Predicting small molecule and RNA target interactions using deep neural network

RNA-targeting small molecules can bind RNA to regulate its function, providing a promising alternative approach for the treatment of human disease. However, the development of this field has been hindered by the lack of effective experimental probing and computational prediction approaches. Here, we present SmrtNet (Predicting small molecule and RNA target interactions using deep neural network), a deep learning-based framework designed to predict interactions between small molecules and RNA targets. SmrtNet extracts sequence and structural information from both small molecule and RNA target using large-scale language models and attention-based neural networks, and integrates mutual information among different encoders using a novel feature fusion strategy.

<p align="center"><img src="figs/workflow.png" width=100% /></p>

## Cite us
If you found this package useful, please cite [our paper](xxx):
```
Yuhan Fei and Jiasheng Zhang, xxx
```
## Table of contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Example](#example)
- [Copyright and License](#copyright-and-license)
- [Disclaimer](#disclaimer)


## Installation

### `Requirements`
 - Python 3.8
 - PyTorch 1.1.0, with NVIDIA CUDA Support
 - pip
   


### Build from Source
Clone repository: 
```bash
git clone https://github.com/Yuhan-Fei/SmrtNet.git
cd SmrtNet
pip install -r requirements.txt
pip install -e .

or

conda env create -f environment.yml

conda activate SmrtNet
```

### `conda`
```bash
conda create -n smrtnet python=3.8.10
conda activate smrtnet
pip install smrtnet
```


## Datasets

### Prepare the datasets

Scripts and pipeline are in preparing, currently, we provide xxx samples data in *.txt format for training and testing SmrtNet.


```
# Download data
cd SmrtNet/data

```


## Usage

### Network Architecture

<p align="center"><img src="figs/architecture.png" width=100% /></p>


### Check

Check input format
```
python main.py --do_check
```

### Training 

To train one single protein model from scratch, run
```
python main.py --do_train
```
where you replace `TIA1_Hela` with the name of the data file you want to use, you replace EXP_NAME with a specific name of this experiment. Hyper-parameters could be tuned in `exp/prismnet/train.sh`. For available training options, please take a look at `tools/train.py`.

To monitor the training process, add option `-tfboard` in `exp/prismnet/train.sh`, and view page at http://localhost:6006 using tensorboard:
```
tensorboard --logdir exp/EXP_NAME/out/tfb
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

### Case Study 1: Train you own model:

<details>
   <summary>Click here for the code!</summary>
 
```python
cd ~/SmrtNet
python main.py --do_train \
               --in_dir ./dataset/5.5.1.3_10A_norm_simple_unk_single_O4_ion_ext_new_III_2_42.txt \
               --out_dir=./results/20240521_seqstr_benchmark \
               --cuda 0 \
               --batch_size 32 \
               --epoch 100\
               --patiences 20 \
               --tfboard
 ```
</details>

### Case Study 2: Test you data:

<details>
```python
cd ~/SmrtNet
python main.py --do_test \
               --in_dir xxx \
               --out_dir=./results/20240521_seqstr_benchmark \
               --cuda 0 \
               --batch_size 1
 ```
</details>


### Case Study 3: N small molecules for 1 RNA target
In addition to the DTI prediction, we also provide repurpose and virtual screening functions to rapidly generation predictions.

<details>
  <summary>Click here for the code!</summary>

```python


```

</details>
### Case Study 2: 1 small molecule for N RNA targets

### Case Study 3: N small moleucles for N RNA targets

### Case Study 4: transcript-wide analysis (RNA targets more than 31nt)

### Case Study 5: Use multiple GPU

### Case Study 6: Binding site prediction

### Case Study 7: Key functional group prediction





## Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.



## Disclaimer
The prediction of SmrtNet should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.
