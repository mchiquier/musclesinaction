import os
import pdb
subject = "Subject10"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_clean_Subject1 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject1_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject1.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject2 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject2_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject2.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject3 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject3_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject3.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject4 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject4_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject4.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject5 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject5_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--cond False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject5.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Subject6 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject6_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--cond False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject6.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject7 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject7_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject7.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject8 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject8_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject8.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_Subject9 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject9_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject9.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_Subject10 \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py \
--name generalization_new_cond_Subject10_clean_baseline_perex \
--std False \
--threed True \
--cond False \
--predemg False \
--resume checkpoints/generalization_new_nocond_clean_emgtopose_threed/model_200.pth \
--data_path_train ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/train.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalization_ID_nocond_people/valSubject10.txt" Enter'
os.system(command)
time.sleep(20)
