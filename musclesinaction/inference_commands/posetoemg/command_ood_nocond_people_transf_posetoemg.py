import os
import pdb
subject = "Subject10"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time




command = 'tmux new-session -d -s my_session_clean_Subject2 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject2_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject2_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject2.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject2.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject10 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject10_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject10_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject10.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject10.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject9 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject9_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject9_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject9.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject9.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject7 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject7_clean_baseline \
--std False \
--threed True \
--predemg True \
--cond False \
--resume checkpoints/generalization_new_nocond_Subject7_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject7.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject7.txt" Enter'
os.system(command)

command = 'tmux new-session -d -s my_session_clean_Subject1 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject1_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject1_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject1.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject1.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject3 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject3_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject3_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject3.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject3.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject4 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject4_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject4_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject4.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject4.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject5 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject5_clean_baseline \
--std False \
--threed True \
--predemg True \
--cond False \
--resume checkpoints/generalization_new_nocond_Subject5_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject5.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject5.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject6 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject6_clean_baseline \
--std False \
--threed True \
--cond False \
--predemg True \
--resume checkpoints/generalization_new_nocond_Subject6_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject6.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject6.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject8 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises.py \
--name generalization_new_nocond_Subject8_clean_baseline \
--std False \
--threed True \
--predemg True \
--cond False \
--resume checkpoints/generalization_new_nocond_Subject8_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalizationpeople/train_Subject8.txt \
--data_path_val musclesinaction/ablation/generalizationpeople/val_Subject8.txt" Enter'
os.system(command)
time.sleep(20)


