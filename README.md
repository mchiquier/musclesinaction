# Muscles in Action Source Code

Code and pre-trained models for the Muscles in Action ICCV 2023 paper. 

## Setup
Environment: 

1. Install a new conda environment:
```commandline
$ conda create --name musclesinaction --file requirements.txt
```
2. Activate environment:
```commandline
$ conda activate musclesinaction
```

Dataset: 

The dataset can be found at this link: https://musclesinaction.cs.columbia.edu/MIADataset.tar. Download it, and rename it to MIADatasetFinal. Place this folder in the same directory as the top-level musclesinaction folder.



## Training 

```commandline
python musclesinaction/train.py
```

## Inference

See the musclesinaction/inference_scripts folder. 
