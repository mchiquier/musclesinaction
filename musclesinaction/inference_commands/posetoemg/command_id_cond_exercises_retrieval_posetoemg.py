import os
import pdb
subject = "Sruthi"
#exercises = os.listdir("SmallMIADataset/train/" + subject)
import time
command = 'tmux new-session -d -s my_session_clean_Running \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_Running_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valRunning.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_RonddeJambe \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_RonddeJambe_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valRonddeJambe.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_LegCross \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_LegCross_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valLegCross.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_LegBack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_LegBack_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valLegBack.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_KneeKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_KneeKick_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valKneeKick.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_JumpingJack \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_JumpingJack_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valJumpingJack.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_HookPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=6 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_HookPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valHookPunch.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_HighKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=7 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_HighKick_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valHighKick.txt" Enter'
os.system(command)
time.sleep(20)


command = 'tmux new-session -d -s my_session_clean_FrontPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=0 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_FrontPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valFrontPunch.txt" Enter'
os.system(command)
time.sleep(20)



command = 'tmux new-session -d -s my_session_clean_FrontKick \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_FrontKick_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valFrontKick.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_ElbowPunch \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=1 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_ElbowPunch_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valElbowPunch.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_Shuffle \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=2 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_Shuffle_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valShuffle.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSideLunges \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=3 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_SideLunges_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSideLunges.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSlowSkater \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=4 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_SlowSkater_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSlowSkater.txt" Enter'
os.system(command)
time.sleep(20)

command = 'tmux new-session -d -s my_session_clean_valSquat \; send-keys \
"conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; \
send-keys "CUDA_VISIBLE_DEVICES=5 python musclesinaction/inference_scripts/retrieval_id_exercises_posetoemg.py \
--name generalization_test_cond_Squat_clean_baseline_perex \
--std False \
--threed True \
--predemg True \
--data_path_train musclesinaction/ablation/generalization_ID_cond_exercises_nn/train.txt \
--data_path_val musclesinaction/ablation/generalization_ID_cond_exercises_nn/valSquat.txt" Enter'
os.system(command)
time.sleep(20)
