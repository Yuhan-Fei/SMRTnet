# SmrtNet: Predicting small molecule and RNA target interactions using deep neural network

RNA-targeting small molecules can bind RNA to regulate its function, providing a promising alternative approach for the treatment of human disease. However, the development of this field has been hindered by the lack of effective experimental probing and computational prediction approaches. Here, we present SmrtNet (Predicting small molecule and RNA target interactions using deep neural network), a deep learning-based framework designed to predict interactions between small molecules and RNA targets. SmrtNet extracts sequence and structural information from both small molecule and RNA target using large-scale language models and attention-based neural networks, and integrates mutual information among different encoders using a novel feature fusion strategy.


## Table of contents
- [Getting started](#Getting-started)
- [Datasets](#datasets)
- [Usage](#usage)
- [Copyright and License](#copyright-and-license)
- [Reference](#Reference)

## Getting started

### Requirements
 
 - Python 3.6
 - PyTorch 1.1.0, with NVIDIA CUDA Support
 - pip

### Installation
Clone repository: 

```bash
git clone https://github.com/Yuhan-Fei/SmrtNet.git
```
Install packages:
```bash
cd SmrtNet
pip install -r requirements.txt
pip install -e .
```

## Datasets

### Prepare the datasets

Scripts and pipeline are in preparing, currently, we provide xxx samples data in *.txt format for training and testing SmrtNet.

```
# Download data
cd SmrtNet/data
wget https://zhanglabnet.oss-cn-beijing.aliyuncs.com/prismnet/data/clip_data.tgz
tar zxvf clip_data.tgz

# Generate training and validation set for binary classification
cd PrismNet
tools/gdata_bin.sh
```

## Usage

### Network Architecture

![prismnet](https://github.com/kuixu/PrismNet/wiki/imgs/prismnet-arch.png)


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


### Compute High Attention Regions
For computing high attention regions using the trained models, we provide the script `har.sh`. You can run it using
```
python main.py --do_explain


### Example

### Case Study 1: N small molecules for 1 RNA target

### Case Study 2: 1 small molecule for N RNA targets

### Case Study 3: N small moleucles for N RNA targets

### Case Study 4: transcript-wide analysis (RNA targets more than 31nt)

### Case Study 5: Use multiple GPU

### Case Study 6: Binding site prediction

### Case Study 7: Key functional group prediction





## Copyright and License
This project is free to use for non-commercial purposes - see the [LICENSE](LICENSE) file for details.

## Reference

```
@article {}

```


