import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_std_ElbowPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/train.py \
--name correctsruthi_ood_cond_ElbowPunch_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_ElbowPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_ElbowPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_ElbowPunch.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_std_FrontKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name correctsruthi_ood_cond_FrontKick_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_FrontKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_FrontKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_FrontKick.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_FrontPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/train.py \
--name correctsruthi_ood_cond_FrontPunch_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_FrontPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_FrontPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_FrontPunch.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_HighKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/train.py \
--name correctsruthi_ood_cond_HighKick_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_HighKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_HighKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_HighKick.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_HookPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/train.py \
--name correctsruthi_ood_cond_HookPunch_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_HookPunch.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_HookPunch.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_HookPunch.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_JumpingJack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/train.py \
--name correctsruthi_ood_cond_JumpingJack_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_JumpingJack.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_JumpingJack.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_JumpingJack.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_LegBack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/train.py \
--name correctsruthi_ood_cond_LegBack_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_LegBack.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_LegBack.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_LegBack.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_KneeKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/train.py \
--name correctsruthi_ood_cond_KneeKick_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_KneeKick.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_KneeKick.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_KneeKick.txt" Enter'
os.system(command)

time.sleep(10)

