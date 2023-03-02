# This is an example pipeline of training and evaluating a DL model.
# You don't necessarily have to execute this script; separate commands can also be copy-pasted in
# the shell, such that this file just helps maintain a clear oversight.

subject = "Sruthi"
exercises = os.listdir("SmallMIADataset/train/" + subject)

command = 'tmux new-session -d -s my_session_three \; send-keys "conda activate /proj/vondrick4/mia/condaenvs/vibe-env2" Enter \; send-keys "cd musclesinaction" Enter'
os.system(command)
print("here")
