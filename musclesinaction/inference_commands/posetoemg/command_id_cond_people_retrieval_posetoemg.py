import os
import pdb
subject = "Subject10"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_clean_Subject1 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject1_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject1_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject1.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject1.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject2 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject2_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject2_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject2.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject2.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject3 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject3_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject3_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject3.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject3.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject4 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject4_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject4_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject4.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject4.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject5 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject5_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject5_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainLionel.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valLionel.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject6 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject6_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject6_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject6.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valMe.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject7 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject7_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject7_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject7.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject7.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject8 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject8_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject8_clean/model_100.pth\
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject8.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject8.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject9 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject9_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject9_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject9.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject9.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_Subject10 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_cnn_people.py \
--name generalization_test_cond_Subject10_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_test_cond_Subject10_clean/model_100.pth \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/trainSubject10.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSubject10.txt" Enter'
os.system(command)
time.sleep(20)
