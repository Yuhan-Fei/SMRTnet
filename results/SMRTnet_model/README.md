### prepare the model of SMRTnet
```bash
cd ${Path_SMRTnet}
wget -O ./SMRTnet_model.zip https://zenodo.org/records/14715564/files/SMRTnet_model.zip?download=1
unzip ./SMRTnet_model.zip
mv ./SMRTnet_model/* ./results/20231229_lbncab4_v3_allrna_ep100_bs32_lr00001_linear_simple_drug_cls_1024_1024_1024_512_CV5_4_fix/
```

