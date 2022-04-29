# Clinically significant Prostate Cancer Classification using Fusion
Reference Implementation of paper "Paper Title" of Heejong Kim, Mert Sabuncu, Himanshu Nagar*, Daniel Margolis* (* Corresponding author) 

[comment]: <> (, to appear in "Journal".)
[comment]: <> ([Project Page]&#40;https://heejongkim.com/dwi-synthesis&#41; | [Paper]&#40;https://arxiv.org/abs/2106.13188&#41; )

[comment]: <> (![Add figure here]&#40;demo.gif&#41;)

## Dependencies 
```shell
conda env create -f environment.yml
conda activate cspca-classification-fusion
```

## PROSTATEx challenge Dataset Preparation
Please download the following files and change the file location variables to run the preprocessing script. 
- Download the image SimpleITK transform files ([Link](https://github.com/OscarPellicer/prostate_lesion_detection/blob/main/ProstateX_transforms.zip))
- Download the PROSTATEx challenge dataset ([Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656))
```python
python preprocessing.py
```

## Training
We included all configuration files for 
- One channel input (T2, ADC, DWIb800, Ktrans) 
- Three channel input (T2-ADC-DWIb800, T2-ADC-Ktrans) 
```shell script
# For example, 
python train.py --config ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml
```

## Testing 
The testing script outputs accuracy metrics, such as AUC with confidence interval and saliency maps. Other than the analysis result of testset with labels, you can also get the PROSTATEx challenge testset submission results. 
Please note that challenge submission is not available as of Apr 30th. ([Challenge board] (https://prostatex.grand-challenge.org/))


## Citation
If you use this code, please consider citing our work:
```
Paper info will be added
```
