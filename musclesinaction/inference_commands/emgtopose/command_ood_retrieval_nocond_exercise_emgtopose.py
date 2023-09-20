import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_clean_Running \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_Running_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_Running.txt \
--data_path_val generalizationexercises/val_Running.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_RonddeJambe \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_RonddeJambe_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_RonddeJambe.txt \
--data_path_val generalizationexercises/val_RonddeJambe.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_LegCross \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_LegCross_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_LegCross.txt \
--data_path_val generalizationexercises/val_LegCross.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_LegBack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_LegBack_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_LegBack.txt \
--data_path_val generalizationexercises/val_LegBack.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_KneeKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_KneeKick_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_KneeKick.txt \
--data_path_val generalizationexercises/val_KneeKick.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_JumpingJack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_JumpingJack_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_JumpingJack.txt \
--data_path_val generalizationexercises/val_JumpingJack.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_HookPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_HookPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_HookPunch.txt \
--data_path_val generalizationexercises/val_HookPunch.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_HighKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_HighKick_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_HighKick.txt \
--data_path_val generalizationexercises/val_HighKick.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_FrontPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_FrontPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_FrontPunch.txt \
--data_path_val generalizationexercises/val_FrontPunch.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_FrontKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_FrontKick_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_FrontKick.txt \
--data_path_val generalizationexercises/val_FrontKick.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_ElbowPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_ElbowPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_ElbowPunch.txt \
--data_path_val generalizationexercises/val_ElbowPunch.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Shuffle \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_Shuffle_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_Shuffle.txt \
--data_path_val generalizationexercises/val_Shuffle.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSideLunges \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_SideLunges_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_SideLunges.txt \
--data_path_val generalizationexercises/val_SideLunges.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSlowSkater \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_SlowSkater_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_SlowSkater.txt \
--data_path_val generalizationexercises/val_SlowSkater.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSquat \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py \
--name generalization_test_cond_Squat_clean_baseline_perex \
--std False \
--threed True \
--predemg False \
--data_path_train generalizationexercises/train_Squat.txt \
--data_path_val generalizationexercises/val_Squat.txt" Enter'
os.system(command)
time.sleep(20)
