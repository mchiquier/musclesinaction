'''
Training + validation oversight and recipe configuration.
'''


# Internal imports.
import numpy as np
import torch 
import torchvision
import random
import os
import time
import tqdm
from pathlib import Path
import pdb
import musclesinaction.models.modelemgtopose as transmodelemgtopose
import musclesinaction.models.model as transmodelposetoemg
import musclesinaction.models.modelbert as transmodelbert
import musclesinaction.models.basicconv as convmodel

import musclesinaction.configs.args as args
import musclesinaction.dataloader.data as data
import musclesinaction.losses.loss as loss
import musclesinaction.models.model as model
import vis.logvis as logvis
import musclesinaction.utils.utils as utils
import pipeline as pipeline

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _inference_one_epoch(args, train_pipeline, phase, epoch, optimizer,
                     lr_scheduler, train_data_loader, val_data_loader,device, logger):
    #assert phase in ['train', 'val', 'val_aug', 'val_noaug']

    log_str = f'Epoch (1-based): {epoch + 1} / {args.num_epochs}'
    logger.info()
    logger.info('=' * len(log_str))
    logger.info(log_str)
    if phase == 'train':
        logger.info(f'===> Train ({phase})')
        logger.report_scalar(phase + '/learn_rate', _get_learning_rate(optimizer), step=epoch)
    else:
        logger.info(f'===> Validation ({phase})')

    train_pipeline[1].set_phase(phase)

    steps_per_epoch = len(train_data_loader) + len(val_data_loader)
    total_step_base = steps_per_epoch * epoch  # This has already happened so far.
    if phase != 'train':
        total_step_base = total_step_base + len(train_data_loader)
    start_time = time.time()
    num_exceptions = 0
    if phase == 'train':
        data_loader = train_data_loader
    else:
        data_loader = val_data_loader
    list_of_total_loss = []

    for cur_step, data_retval in enumerate(tqdm.tqdm(data_loader)):

        framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]

        if 'Jo' in framelist[0] and 'SlowSkater' in framelist[0]:

            if cur_step == 0:
                logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

            total_step = cur_step + total_step_base  # For continuity in wandb.
            try:
                # First, address every example independently.
                # This part has zero interaction between any pair of GPUs.
                (model_retval, loss_retval) = train_pipeline[0](data_retval, cur_step, total_step)
                ignoremovie = None
                print(loss_retval)
                loss_retval = train_pipeline[1].process_entire_batch(
                    data_retval, model_retval, loss_retval, ignoremovie, cur_step, total_step)
                total_loss = loss_retval['total']
                print(total_loss)
                list_of_total_loss.append(total_loss.item())

            except Exception as e:
                num_exceptions += 1
                if num_exceptions >= 7:
                    raise e
                else:
                    logger.exception(e)
                    continue

            if phase=='eval':
                #print("hi")
                logger.handle_val_step(device, epoch, phase, cur_step, total_step, steps_per_epoch,data_retval, model_retval, loss_retval,"original")
                #pdb.set_trace()


    if phase == 'train':
        lr_scheduler.step()
    print(np.mean(np.array(list_of_total_loss)),"here")
    return total_loss


def _inference(args, train_pipeline, optimizer, lr_scheduler, start_epoch, train_loader, train_loader_noshuffle,
                      val_aug_loader, val_noaug_loader, device, logger, checkpoint_fn):

    logger.info('Start training loop...')
    start_time = time.time()
    list_of_val_vals = []

        
    _inference_one_epoch(
        args, train_pipeline, 'eval', start_epoch, optimizer,
        lr_scheduler, val_aug_loader, val_aug_loader, device, logger)


    total_time = time.time() - start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')


def main(args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.checkpoint_path = args.checkpoint_path + "/" + args.name

    logger.info('Checkpoint path: ' + args.checkpoint_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_val_data_loaders(args, logger)
    logger.info(f'Took {time.time() - start_time:.3f}s')

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks.

    if args.modelname == 'transf':
        model_args = {'threed': args.threed,
            'num_tokens': int(args.num_tokens),
        'dim_model': int(args.dim_model),
        'num_classes': int(args.num_classes),
        'num_heads': int(args.num_heads),
        'classif': args.classif,
        'num_encoder_layers':int(args.num_encoder_layers),
        'num_decoder_layers':int(args.num_decoder_layers),
        'dropout_p':float(args.dropout_p),
        'device': args.device,
        'embedding': args.embedding,
        'step': int(args.step)}
        if args.predemg == 'True':
            model = transmodelposetoemg.TransformerEnc(**model_args)
        else:
            model = transmodelemgtopose.TransformerEnc(**model_args)
    elif args.modelname == 'old':
        model_args = {
        'device': args.device}
        model = convmodel.OldBasicConv(**model_args)
    else:
        model_args = {
        'device': args.device}
        model = convmodel.BasicConv(**model_args)

    # Bundle networks into a list.
    networks = [model]
    for i in range(len(networks)):
        networks[i] = networks[i].to(device)
    networks_nodp = [net for net in networks]

    # Instantiate encompassing pipeline for more efficient parallelization.
    train_pipeline = pipeline.MyTrainPipeline(args, logger, networks, device)
    train_pipeline = train_pipeline.to(device)
    train_pipeline_nodp = train_pipeline
    if args.device == 'cuda':
        train_pipeline = torch.nn.DataParallel(train_pipeline)

    # Instantiate optimizer & learning rate scheduler.
    optimizer = torch.optim.AdamW(train_pipeline.parameters(), lr=args.learn_rate)
    milestones = [(args.num_epochs * 2) // 5,
                  (args.num_epochs * 3) // 5,
                  (args.num_epochs * 4) // 5]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=args.lr_decay)

    # Load weights from checkpoint if specified.
    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        networks_nodp[0].load_state_dict(checkpoint['my_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    logger.info(f'Took {time.time() - start_time:.3f}s')

    # Define logic for how to store checkpoints.
    def save_model_checkpoint(epoch):
        if args.checkpoint_path:
            logger.info(f'Saving model checkpoint to {args.checkpoint_path}...')
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'train_args': args,
                'dset_args': dset_args,
                'model_args': model_args,
            }
            checkpoint['my_model'] = networks_nodp[0].state_dict()
            torch.save(checkpoint,
                       os.path.join(args.checkpoint_path, 'model_{}.pth'.format(epoch)))
            torch.save(checkpoint,
                       os.path.join(args.checkpoint_path, 'checkpoint.pth'))
            logger.info()

    # Start eval loop.
    _inference(
        args, (train_pipeline, train_pipeline_nodp), optimizer, lr_scheduler, start_epoch,
        train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, device, logger, save_model_checkpoint)


if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = args.inference_args()

    logger = logvis.MyLogger(args, args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')