import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_std_ElbowPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/train.py \
--name generalization_test_cond_ElbowPunch_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_ElbowPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_ElbowPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_ElbowPunch.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_std_FrontKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name generalization_test_cond_FrontKick_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_FrontKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_FrontKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_FrontKick.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_FrontPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/train.py \
--name generalization_test_cond_FrontPunch_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_FrontPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_FrontPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_FrontPunch.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_HighKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/train.py \
--name generalization_test_cond_HighKick_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_HighKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_HighKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_HighKick.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_HookPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/train.py \
--name generalization_test_cond_HookPunch_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_HookPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_HookPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_HookPunch.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_JumpingJack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/train.py \
--name generalization_test_cond_JumpingJack_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_JumpingJack.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_JumpingJack.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_JumpingJack.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_KneeKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/train.py \
--name generalization_test_cond_KneeKick_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_KneeKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_KneeKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_KneeKick.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_LegBack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/train.py \
--name generalization_test_cond_LegBack_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_LegBack.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_LegBack.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_LegBack.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_std_LegCross \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name generalization_test_cond_LegCross_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_LegCross.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_LegCross.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_LegCross.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_RonddeJambe \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/train.py \
--name generalization_test_cond_RonddeJambe_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_RonddeJambe.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_RonddeJambe.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_RonddeJambe.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Running \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/train.py \
--name generalization_test_cond_Running_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Running.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_Running.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Running.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Shuffle \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/train.py \
--name generalization_test_cond_Shuffle_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Shuffle.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_Shuffle.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Shuffle.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_SideLunges \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/train.py \
--name generalization_test_cond_SideLunges_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_SideLunges.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_SideLunges.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_SideLunges.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_SlowSkater \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/train.py \
--name generalization_test_cond_SlowSkater_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_SlowSkater.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_SlowSkater.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_SlowSkater.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Squat \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/train.py \
--name generalization_test_cond_Squat_std \
--std True \
--data_path_train ../../../vondrick/mia/VIBE/generalizationpeople/train_Squat.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/generalizationpeople/val_Squat.txt \
--data_path_val ../../../vondrick/mia/VIBE/generalizationpeople/val_Squat.txt" Enter'
os.system(command)

time.sleep(20)