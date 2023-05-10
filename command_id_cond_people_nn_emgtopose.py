import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_clean_David \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_David_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_David_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainDavid.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valDavid.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Ishaan \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Ishaan_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Ishaan_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainIshaan.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valIshaan.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jo \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Jo_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Jo_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainJo.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valJo.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jonny \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Jonny_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Jonny_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainJonny.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valJonny.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Lionel \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Lionel_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Lionel_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainLionel.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valLionel.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Me \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Me_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Me_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainMe.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valMe.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Samir \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Samir_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Samir_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainSamir.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valSamir.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Serena \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Serena_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Serena_clean_emgtopose/model_100.pth\
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainSerena.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valSerena.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Sonia \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Sonia_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Sonia_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainSonia.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valSonia.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_Sruthi \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/knn_id_cnn_people_emgtopose.py \
--name generalization_new_cond_Sruthi_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--resume checkpoints/generalization_new_cond_Sruthi_clean_emgtopose/model_100.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/trainSruthi.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_cond_exercises_nn/valSruthi.txt" Enter'
os.system(command)
time.sleep(20)
