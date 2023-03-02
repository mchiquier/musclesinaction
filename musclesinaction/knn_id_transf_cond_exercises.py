import numpy as np
import pdb
import torch
import musclesinaction.configs.args as args
import vis.logvis as logvis
import musclesinaction.dataloader.data as data
import time
import os
import random
import torch
import musclesinaction.models.modelemgtopose as transmodelemgtopose
import musclesinaction.models.model as transmodelposetoemg
import tqdm
import musclesinaction.models.modelbert as transmodel
import musclesinaction.models.model as model
import musclesinaction.models.basicconv as convmodel


def perspective_projection(points, rotation, translation,
                        focal_length, camera_center):

    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    #pdb.set_trace()
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1], points

def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)
    #pdb.set_trace()
    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, distance='L2'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros((num_test,240), dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            #print(i,num_test)
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            if distance == 'L1':
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            # using the L2 distance (sum of absolute value differences)
            if distance == 'L2':
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i,:] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred

def main(args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
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
            my_model = transmodelposetoemg.TransformerEnc(**model_args)
        else:
            my_model = transmodelemgtopose.TransformerEnc(**model_args)
    elif args.modelname == 'old':
        model_args = {
        'device': args.device}
        my_model = convmodel.OldBasicConv(**model_args)
    else:
        model_args = {
        'device': args.device}
        my_model = convmodel.BasicConv(**model_args)

    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        my_model.load_state_dict(checkpoint['my_model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    device = torch.device(args.device)
    my_model.to(device)



    list_of_val_emg = []
    list_of_pred_emg = []

    logger.info('Initializing data loaders...')

    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)


    for cur_step, data_retval in enumerate(tqdm.tqdm(val_aug_loader)):
            
        threedskeleton = data_retval['3dskeleton']
        twodskeleton = data_retval['2dskeleton']
        emggroundtruth = data_retval['emg_values']
        emggroundtruth = data_retval['emg_values']
        emggroundtruth = emggroundtruth
        cond = data_retval['condval'].to(device)
        badcond = data_retval['condvalbad'].to(device)
        #bined_left_quad = bined_left_quad.to(self.device)-1
        skeleton = threedskeleton
        skeleton = skeleton.reshape(skeleton.shape[0],skeleton.shape[1],-1).to(device)
        cur2 = time.time()
        #print(cur2-cur,"2")
        cur = cur2
        if args.modelname == 'transf':
            if args.predemg == 'True':
                emg_output = my_model(skeleton,cond) 
                emg_output_wrong_condition = my_model(skeleton, badcond)
            else:
                pose_pred = my_model(emggroundtruth[:,:,:],cond)
                pose_pred_wrong_condition = my_model(emggroundtruth[:,:,:],badcond)
        else:
            emg_output = my_model(torch.unsqueeze(skeleton.permute(0,2,1),dim=1),cond)
        #pdb.set_trace()
        #torch.unsqueeze(twodkpts.permute(0,2,1),dim=1)
        #pdb.set_trace()
        #emg_output = my_model(twodkpts)
        #emg_output = emg_output.permute(0,2,1)
        #pdb.set_trace()
        list_of_pred_emg.append(emg_output.reshape(-1).detach().cpu().numpy())
        list_of_val_emg.append(emggroundtruth.cpu().reshape(-1).numpy())
        #list_of_val_skeleton.append(twodkpts.reshape(-1).numpy())
       
    gtemg = np.concatenate(list_of_val_emg,axis=0)
    predemg = np.concatenate(list_of_pred_emg,axis=0)
    msel = torch.nn.MSELoss()
    pdb.set_trace()
    print(torch.sqrt(msel(torch.tensor(gtemg),torch.tensor(predemg))))

    #list_of_results.append(msel(torch.tensor(ypred)*100,torch.tensor(np_val_emg)*100).numpy())
    #list_of_resultsnn.append(msel(torch.tensor(np_pred_emg)*100,torch.tensor(np_val_emg)*100).numpy())
    #print(list_of_results)
    #print(list_of_resultsnn)
    #print(np.mean(np.array(list_of_results)))
    #print(np.mean(np.array(list_of_resultsnn)))
        
        
    #print(np.mean(np.sqrt(np.sum(np.square(ypred - np_val_emg), axis=1))), trainpath)
    #pdb.set_trace()

#TRAIN SKELETON MATRIX FLATTENED OVER TIME 
#TRAIN EMG MATRIX FLATTENED OVER TIME 
if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    args = args.inference_args()
    args.bs = 1

    logger = logvis.MyLogger(args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')


#TEST SKELETON MATRIX FLATTENED OVER TIME
#TEST EMG MATRIX FLATTENED OVER TIME
