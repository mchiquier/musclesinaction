import numpy as np
import torch
import random
import os
import time
import musclesinaction.dataloader.data as data
import musclesinaction.configs.args as args
import musclesinaction.vis.logvis as logvis
import pdb
import tqdm

def main(args, logger):

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Instantiate datasets.
    start_time = time.time()
    (train_loader, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)

    for cur_step, data_retval in enumerate(tqdm.tqdm(train_loader)):
        pdb.set_trace()


if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()
    
    argstrain = args.train_args()
    logger = logvis.MyLogger(argstrain, context='train')
    
    main(argstrain, logger)
