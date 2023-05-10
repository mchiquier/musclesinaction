import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time




command = 'tmux new-session -d -s my_session_clean_Ishaan \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Ishaan_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Ishaan_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Ishaan.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Ishaan.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Sruthi \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Sruthi_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Sruthi_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Sruthi.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Sruthi.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Sonia \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Sonia_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Sonia_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Sonia.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Sonia.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Samir \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Samir_clean_baseline \
--std False \
--threed True \
--predemg False \
--cond False \
--resume checkpoints/generalization_new_nocond_Samir_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Samir.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Samir.txt" Enter'
os.system(command)

command = 'tmux new-session -d -s my_session_clean_David \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_David_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_David_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_David.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_David.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jo \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Jo_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Jo_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Jo.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Jo.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jonny \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Jonny_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Jonny_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Jonny.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Jonny.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Lionel \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Lionel_clean_baseline \
--std False \
--threed True \
--predemg False \
--cond False \
--resume checkpoints/generalization_new_nocond_Lionel_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Lionel.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Lionel.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Me \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Me_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_Me_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Me.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Me.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Serena \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/knn_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_nocond_Serena_clean_baseline \
--std False \
--threed True \
--predemg False \
--cond False \
--resume checkpoints/generalization_new_nocond_Serena_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Serena.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Serena.txt" Enter'
os.system(command)
time.sleep(20)


