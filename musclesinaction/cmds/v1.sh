# This is an example pipeline of training and evaluating a DL model.
# You don't necessarily have to execute this script; separate commands can also be copy-pasted in
# the shell, such that this file just helps maintain a clear oversight.

python train.py --name v1

python test.py --resume v1 --name t1
