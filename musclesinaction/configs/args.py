'''
Handling of parameters that can be passed to training and testing scripts.
'''

#from __init__ import *
from benedict import benedict
import pdb
import argparse
import os

def train_args():
    #pdb.set_trace()
    thedict = benedict.from_yaml('musclesinaction/configs/train.yaml')
    #pdb.set_trace()
    parser = argparse.ArgumentParser()

    for key in thedict.keys():
        parser.add_argument("--" + key, default=thedict[key])
    args = parser.parse_args()
    args.bs = int(args.bs)
    args.learn_rate = float(args.learn_rate)
    #movie = args.data_path_train
    # movie = movie.split("/")[-1].split(".txt")[0].split("_")[2]    
    movie = 'all'
    #args.name = "oct20_" + movie + "_" + str(args.learn_rate) + "_" + args.modelname
    args.modelname = args.modelname.split("_")[0]
    #args.break_test = float(args.break_test)
    #args.break_train = float(args.break_train)
    return args

def inference_args():
    #pdb.set_trace()
    thedict = benedict.from_yaml('musclesinaction/configs/inference.yaml')
    #pdb.set_trace()
    parser = argparse.ArgumentParser()

    for key in thedict.keys():
        parser.add_argument("--" + key, default=thedict[key])
    args = parser.parse_args()
    args.bs = int(args.bs)
    args.learn_rate = float(args.learn_rate)
    #movie = args.data_path_train
    # movie = movie.split("/")[-1].split(".txt")[0].split("_")[2]    
    movie = 'all'
    #args.name = "oct20_" + movie + "_" + str(args.learn_rate) + "_" + args.modelname
    args.modelname = args.modelname.split("_")[0]
    #args.break_test = float(args.break_test)
    #args.break_train = float(args.break_train)
    return args

def _str2bool(v): 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _arg2str(arg_value):
    if isinstance(arg_value, bool):
        return '1' if arg_value else '0'
    else:
        return str(arg_value)



def verify_args(args, is_train=False):

    assert args.device in ['cuda', 'cpu']

    args.is_debug = args.name.startswith('d')

    if is_train:

        # Handle allowable options.
        assert args.optimizer in ['sgd', 'adam', 'adamw', 'lamb']

    if args.num_workers < 0:
        if is_train:
            if args.is_debug:
                args.num_workers = max(int(mp.cpu_count() * 0.45) - 6, 4)
            else:
                args.num_workers = max(int(mp.cpu_count() * 0.95) - 8, 4)
        else:
            args.num_workers = max(mp.cpu_count() * 0.25 - 4, 4)
        args.num_workers = min(args.num_workers, 116)
    args.num_workers = int(args.num_workers)

    # If we have no name (e.g. for smaller scripts in eval), assume we are not interested in logging
    # either.
    if args.name != '':

        if is_train:
            # For example, --name v1.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.name)
            args.train_log_path = os.path.join(args.log_root, args.name)

            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(args.train_log_path, exist_ok=True)

        if args.resume != '':
            # Train example: --resume v3 --name dbg4.
            # Test example: --resume v1 --name t1.
            # NOTE: In case of train, --name will mostly be ignored.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.resume)
            args.train_log_path = os.path.join(args.log_root, args.resume)

            if args.epoch >= 0:
                args.resume = os.path.join(args.checkpoint_path, f'model_{args.epoch}.pth')
                args.name += f'_e{args.epoch}'
            else:
                args.resume = os.path.join(args.checkpoint_path, 'checkpoint.pth')

            assert os.path.exists(args.checkpoint_path) and os.path.isdir(args.checkpoint_path)
            assert os.path.exists(args.train_log_path) and os.path.isdir(args.train_log_path)
            assert os.path.exists(args.resume) and os.path.isfile(args.resume)

        if not(is_train):
            assert args.resume != ''
            args.test_log_path = os.path.join(args.train_log_path, 'test_' + args.name)
            args.log_path = args.test_log_path
            os.makedirs(args.test_log_path, exist_ok=True)

        else:
            args.log_path = args.train_log_path
