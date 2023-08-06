import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_general_clean_posetoemg \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/train.py \
--name generalization_new_cond_clean_smpl_two \
--std False \
--predemg True \
--threed True \
--cond True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_general_clean_emgtopose \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name generalization_new_cond_clean_emgtopose_smpl_two \
--predemg False \
--threed True \
--learn_rate 0.0005 \
--std False \
--cond True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)
"""
command = 'tmux new-session -d -s my_session_general_clean_emgtopose_twod \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name generalization_new_cond_clean_emgtopose_twod \
--predemg False \
--threed False \
--std False \
--cond True \
--learn_rate 0.0005 \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_general_clean_posetoemg_nocond \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/train.py \
--name generalization_new_nocond_clean_posetoemg \
--std False \
--predemg True \
--cond False \
--threed True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_general_clean_emgtopose_nocond \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/train.py \
--name generalization_new_nocond_clean_emgtopose_threed \
--predemg False \
--threed True \
--learn_rate 0.0005 \
--cond False \
--std False \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_general_clean_emgtopose_twod_nocond \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/train.py \
--name generalization_new_nocond_clean_emgtopose_twod \
--predemg False \
--threed False \
--std False \
--cond False \
--learn_rate 0.0005 \
--data_path_train ../../../vondrick/mia/VIBE/generalizationFinal/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationFinal/val.txt" Enter'
os.system(command)
time.sleep(20)"""


