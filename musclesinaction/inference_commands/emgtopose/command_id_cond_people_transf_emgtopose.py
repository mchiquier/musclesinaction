import os
import pdb
import time

command = 'tmux new-session -d -s my_session_clean_David \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_David_clean_baseline_perex \
--std False \
--threed True \
--cond True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valDavid.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Ishaan \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Ishaan_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valIshaan.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jo \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Jo_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valJo.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Jonny \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Jonny_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valJonny.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Lionel \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Lionel_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valLionel.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Me \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Me_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valMe.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Samir \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Samir_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSamir.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Serena \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Serena_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSerena.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Sonia \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Sonia_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSonia.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_Sruthi \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Sruthi_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_cond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSruthi.txt" Enter'
os.system(command)
time.sleep(20)
