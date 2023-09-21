# Muscles in Action (ICCV 2023)

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

To train your own model, run the following command below. By default, it pulls from the musclesinaction/configs/train.yaml file. 

```commandline
python musclesinaction/train.py
```

The default is to train a pose-to-emg model, defined with 'predemg=True'. To train an emg-to-pose model, simply set it to False. 

The config file also specifies the information for what data the model is being trained on, as well as where checkpoints are saved, etc. Update it for your goals. 


## Inference

The 'musclesinaction/inference_commands' folder has many different scripts to evaluate our model and baselines, per exercise and per person, for both in-distribution and out-of-distribution experiments. For instance, to evaluate the emg-to-pose model per exercise, in-distribution, with our model, you would run the following command: 

```commandline
python musclesinaction/inference_commands/emgtopose/command_id_cond_exercises_transf_emgtopose.py
```

This will open a tmux session per exercise, and prints the error on the test set for that exercise. 
