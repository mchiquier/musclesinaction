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

def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _inference_one_epoch(args, train_pipeline, phase, train_data_loader, val_data_loader,device, loggerone, loggertwo):
    #assert phase in ['train', 'val', 'val_aug', 'val_noaug']

 
    train_pipeline[1].set_phase(phase)
    train_pipeline[0].set_phase(phase)

    steps_per_epoch = len(train_data_loader) + len(val_data_loader)

    start_time = time.time()
    num_exceptions = 0
    if phase == 'train':
        data_loader = train_data_loader
    else:
        data_loader = val_data_loader
    list_of_total_loss = []
    for cur_step, data_retval in enumerate(tqdm.tqdm(data_loader)):


        total_step = cur_step
        j=0
        framelist = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]

        #if 'ElbowPunch' in framelist[0] and 'Me' in framelist[0]:
        if 'Samir' in framelist[0]:
            if 'RonddeJambeGood' in framelist[0]:
                try:
                    # First, address every example independently.
                    # This part has zero interaction between any pair of GPUs.
                    #pdb.set_trace()
                    arms = torch.tensor(np.load("prednp13.npy"))
                    #data_retval['emg_values'][:,2,:] = arms[:,2,:]
                    #data_retval['emg_values'][:,3,:] = arms[:,3,:]
                    #data_retval['emg_values'][:,6,:] = arms[:,6,:]
                    #data_retval['emg_values'][:,7,:] = arms[:,7,:]
                    (model_retval, loss_retval) = train_pipeline[0](data_retval, cur_step, total_step)
                    loggerone.handle_val_step(device,0, phase, cur_step, total_step, steps_per_epoch,data_retval, model_retval, model_retval, "original")
                    data_retval['old_emg_values'] = data_retval['emg_values']

                    #pdb.set_trace()
                    """current = model_retval['emg_output'][:,1,:] #ME, SQUAT
                    current[current>torch.min(current)] = current[current>torch.min(current)]*3

                    current = model_retval['emg_output'][:,5,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]*3

                    current = model_retval['emg_output'][:,2,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]*2

                    current = model_retval['emg_output'][:,6,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]*2

                    current = model_retval['emg_output'][:,0,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]*5

                    current = model_retval['emg_output'][:,4,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]*5

                    current = model_retval['emg_output'][:,0,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]/10

                    current = model_retval['emg_output'][:,1,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]/10

                    current = model_retval['emg_output'][:,4,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]/10

                    current = model_retval['emg_output'][:,5,:] 
                    current[current>torch.min(current)] = current[current>torch.min(current)]/10"""

                    

                    #model_retval['emg_output'][:,5,:] = model_retval['emg_output'][:,5,:]*2#squatjonny
                    #model_retval['emg_output'][:,0,:] = model_retval['emg_output'][:,0,:]*3#squatjonny
                    #model_retval['emg_output'][:,4,:] = model_retval['emg_output'][:,4,:]*2#squatjonny
                    #current = model_retval['emg_output'][:,6,:]
                    #current[current>torch.min(current)] = current[current>torch.min(current)]*20
                    #model_retval['emg_output'][:,6,:] = current#model_retval['emg_output'][:,6,:]*10#miadecreaselateralspunch
                    #current = model_retval['emg_output'][:,2,:]
                    #current[current>torch.min(current)] = current[current>torch.min(current)]*15
                    #model_retval['emg_output'][:,2,:] = current #model_retval['emg_output'][:,2,:]*10#miadecreaselateralspunch
                    
                    #model_retval['emg_output'][:,1,:] = model_retval['emg_output'][:,1,:]/10#slowskatersamir
                    #model_retval['emg_output'][:,5,:] = model_retval['emg_output'][:,5,:]/10#slowskatersamir


                    #model_retval['emg_output'][:,1,:] = model_retval['emg_output'][:,1,:]*3#squatjonny
                    #model_retval['emg_output'][:,5,:] = model_retval['emg_output'][:,5,:]*2#squatjonny
                    #model_retval['emg_output'][:,0,:] = model_retval['emg_output'][:,0,:]*3#squatjonny
                    #model_retval['emg_output'][:,4,:] = model_retval['emg_output'][:,4,:]*2#squatjonny
                    #pdb.set_trace()
                    #model_retval['emg_output'][:,0,:] = model_retval['emg_output'][:,0,:]*2#ronddejambesruthi
                    #model_retval['emg_output'][:,6,:] = model_retval['emg_output'][:,6,:]*10#ronddejambesruthi
                    #model_retval['emg_output'][:,2,:] = model_retval['emg_output'][:,2,:]*10#ronddejambesruthi
                    #model_retval['emg_output'][:,4,:] = model_retval['emg_output'][:,4,:]*2#ronddejambesruthi
                    #loggerone.handle_val_step(device,0, phase, cur_step, total_step, steps_per_epoch,data_retval, model_retval, model_retval, "edited")
                    #model_retval['emg_output'][:,6,:] = model_retval['emg_output'][:,6,:]/10#miadecreaselateralspunch
                    #model_retval['emg_output'][:,2,:] = model_retval['emg_output'][:,2,:]/10#miadecreaselateralspunch
                    #model_retval['emg_output'][:,2,:] = arms[:,2,:]
                    #model_retval['emg_output'][:,3,:] = arms[:,3,:]
                    #model_retval['emg_output'][:,6,:] = arms[:,6,:]
                    #model_retval['emg_output'][:,7,:] = arms[:,7,:]
                    model_retval['emg_output'][:,0,:] = arms[:,0,:]
                    model_retval['emg_output'][:,1,:] = arms[:,1,:]
                    model_retval['emg_output'][:,4,:] = arms[:,4,:]
                    model_retval['emg_output'][:,5,:] = arms[:,5,:]
                    loggerone.handle_val_step(device,0, phase, cur_step, total_step, steps_per_epoch,data_retval, model_retval, model_retval, "edited")
                    #model_retval['emg_output'][:,0,:] = arms[:,0,:]
                    #model_retval['emg_output'][:,1,:] = arms[:,1,:]
                    #model_retval['emg_output'][:,4,:] = arms[:,4,:]
                    #model_retval['emg_output'][:,5,:] = arms[:,5,:]
                    data_retval['emg_values'] = model_retval['emg_output']
                    (model_retval, loss_retval) = train_pipeline[1](data_retval, cur_step, total_step)
                    ignoremovie = None
                    device = torch.device(args.device)
                    loggertwo.handle_val_step(device,0, phase, cur_step, total_step, steps_per_epoch,data_retval, model_retval, model_retval,"editedpred")
                    #pdb.set_trace()
                    print("test")

                        

                except Exception as e:
                    num_exceptions += 1
                    if num_exceptions >= 7:
                        raise e
                    else:
                        loggerone.exception(e)
                        continue


    print(np.mean(np.array(list_of_total_loss)),"here")
    return 


def _inference(args, train_pipeline,  train_loader, train_loader_noshuffle,
                      val_aug_loader, val_noaug_loader, device, loggerone, loggertwo):

    start_time = time.time()
    list_of_val_vals = []

        
    _inference_one_epoch(
        args, train_pipeline, 'eval',val_aug_loader, val_aug_loader, device, loggerone, loggertwo)


    total_time = time.time() - start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')


def main(args, argstwo, loggerone, loggertwo):



    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.checkpoint_path = args.checkpoint_path + "/" + args.name

    # Instantiate datasets.
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_val_data_loaders(args, loggertwo)

    start_time = time.time()

    # Instantiate networks.
    args.predemg='True'
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
        
    modelposetoemg = transmodelposetoemg.TransformerEnc(**model_args)
    
    argstwo.predemg='False'
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
        
    modelemgtopose = transmodelemgtopose.TransformerEnc(**model_args)
        

    # Bundle networks into a list.
    networks = [modelposetoemg,modelemgtopose]
    for i in range(len(networks)):
        networks[i] = networks[i].to(device)
    networks_nodp = [net for net in networks]

    checkpoint = torch.load('checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth', map_location='cpu')
    networks_nodp[0].load_state_dict(checkpoint['my_model'])

    checkpoint = torch.load('checkpoints/generalization_new_cond_clean_emgtopose_threed/model_100.pth', map_location='cpu')
    networks_nodp[1].load_state_dict(checkpoint['my_model'])

    #pdb.set_trace()

    

    args.predemg='True'
    # Instantiate encompassing pipeline for more efficient parallelization.
    train_pipeline_posetoemg = pipeline.MyTrainPipeline(args, loggerone, [networks[0]], device)
    train_pipeline_posetoemg = train_pipeline_posetoemg.to(device)

    argstwo.predemg='False'

    train_pipeline_emgtopose = pipeline.MyTrainPipeline(argstwo, loggertwo, [networks[1]], device)
    train_pipeline_emgtopose = train_pipeline_emgtopose.to(device)


   
    # Start eval loop.
    _inference(
        args, (train_pipeline_posetoemg, train_pipeline_emgtopose), 
        train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, device, loggerone, loggertwo)


if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    argsone = args.inference_args()
    argstwo = args.inference_args()

    loggerone = logvis.MyLogger(argsone, argstwo, context='train')
    loggertwo = logvis.MyLogger(argstwo, argsone, context='train')

    

    main(argsone, argstwo, loggerone, loggertwo)

    