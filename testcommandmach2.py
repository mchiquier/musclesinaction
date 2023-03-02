import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_std_LegCross \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/train.py \
--name correctsruthi_ood_cond_LegCross_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_LegCross.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_LegCross.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_LegCross.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_std_RonddeJambe \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/train.py \
--name correctsruthi_ood_cond_RonddeJambe_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_RonddeJambe.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_RonddeJambe.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_RonddeJambe.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Running \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/train.py \
--name correctsruthi_ood_cond_Running_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_Running.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_Running.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_Running.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Shuffle \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/train.py \
--name correctsruthi_ood_cond_Shuffle_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_Shuffle.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_Shuffle.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_Shuffle.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_SideLunges \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/train.py \
--name correctsruthi_ood_cond_SideLunges_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_SideLunges.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_SideLunges.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_SideLunges.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_SlowSkater \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/train.py \
--name correctsruthi_ood_cond_SlowSkater_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_SlowSkater.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_SlowSkater.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_SlowSkater.txt" Enter'
os.system(command)

time.sleep(20)

command = 'tmux new-session -d -s my_session_std_Squat \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/train.py \
--name correctsruthi_ood_cond_Squat_std \
--data_path_train ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_train_Squat.txt \
--data_path_val_ood ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valout_Squat.txt \
--data_path_val ../../../vondrick/mia/VIBE/correctsruthioodfiles/sruthi_cond_ood_valrest_Squat.txt" Enter'
os.system(command)


time.sleep(10)

