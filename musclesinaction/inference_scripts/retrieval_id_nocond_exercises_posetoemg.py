import numpy as np
import pdb
import torch
import musclesinaction.configs.args as args
import musclesinaction.vis.logvis as logvis
import musclesinaction.dataloader.data as data
import time
import os
import random
import torch
import tqdm


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



    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)


    list_of_train_emg = []
    list_of_train_skeleton = []
    list_of_train_id = []
    list_of_val_emg = []
    list_of_val_skeleton = []
    list_of_val_id = []
    dict_ex_to_id = {
        'ElbowPunch': 0,
        'FrontPunch':1,
        'HookPunch':2,
        'FrontKick': 3,
        'KneeKick':4,
        'LegBack':5,
        'HighKick':6,
        'LegCross':7,
        'SlowSkater':8,
        'JumpingJack':9,
        'Running':10,
        'Shuffle':11,
        'RonddeJambe':12,
        'RonddeJambeGood':12,
        'RonddeJambeBad':12,
        'BadSquat':13,
        'GoodSquat':13,
        'SquatBad':13,
        'SquatGood':13,
        'SideLunges':14,
        'SideLunge':14,
        'Squat':15
    }


    for cur_step, data_retval in enumerate(tqdm.tqdm(train_loader)):
        
        threedskeleton = data_retval['3dskeleton']
        #pdb.set_trace()
        twodskeleton = data_retval['2dskeleton']
        emggroundtruth = data_retval['emg_values']
        exname = data_retval['frame_paths'][0][0].split("/")[8]
        theid = dict_ex_to_id[exname]
        list_of_train_id.append(theid)
        
        list_of_train_emg.append(emggroundtruth.reshape(-1).numpy())
        list_of_train_skeleton.append(threedskeleton.reshape(-1).numpy())


    for cur_step, data_retval in enumerate(tqdm.tqdm(val_aug_loader)):
            
        threedskeleton = data_retval['3dskeleton']
        twodskeleton = data_retval['2dskeleton']
        emggroundtruth = data_retval['emg_values']
        exname = data_retval['frame_paths'][0][0].split("/")[8]
        #pdb.set_trace()
        #torch.unsqueeze(twodkpts.permute(0,2,1),dim=1)
        #pdb.set_trace()
        #emg_output = my_model(twodkpts)
        #emg_output = emg_output.permute(0,2,1)
        #pdb.set_trace()
        list_of_val_skeleton.append(threedskeleton.reshape(-1).numpy())
        list_of_val_emg.append(emggroundtruth.reshape(-1).numpy())
        #list_of_val_skeleton.append(twodkpts.reshape(-1).numpy())

    np_val_emg = np.array(list_of_val_emg)
    np_train_emg = np.array(list_of_train_emg)
    np_val_skeleton = np.array(list_of_val_skeleton)
    np_train_skeleton = np.array(list_of_train_skeleton)
    np_train_id = np.array(list_of_train_id)
    #pdb.set_trace()
    nn = NearestNeighbor()
    nn.train(np_train_skeleton,np_train_emg)
    #nn.train(np_train_skeleton,np_train_id)
    print("here")
    
    ypred = nn.predict(np_val_skeleton)
    pdb.set_trace()
    #np.save(exname + "_final.npy", ypred)
    msel = torch.nn.MSELoss()
    print(torch.sqrt(msel(torch.tensor(ypred),torch.tensor(np_val_emg))))
    #pdb.set_trace()
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

    args = args.train_args()
    args.bs = 1

    logger = logvis.MyLogger(args, args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')


#TEST SKELETON MATRIX FLATTENED OVER TIME
#TEST EMG MATRIX FLATTENED OVER TIME
