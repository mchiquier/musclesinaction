import numpy as np
import pdb
import torch
import musclesinaction.configs.args as args
import vis.logvis as logvis
import musclesinaction.dataloader.data2 as data2
import time
import os
import random
import torch
import tqdm
import musclesinaction.models.modelbert as transmodel
import musclesinaction.models.modelemgtopose as transmodelemgtopose
import musclesinaction.models.model as transmodelposetoemg
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
        Ypred = np.zeros((num_test,self.ytr.shape[1]), dtype=self.ytr.dtype)

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

    networks = [model]
    for i in range(len(networks)):
        networks[i] = networks[i].to(device)
    networks_nodp = [net for net in networks]


    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        networks_nodp[0].load_state_dict(checkpoint['my_model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0



    # Instantiate optimizer & learning rate scheduler.

 


    logger.info('Initializing data loaders...')
    args.bs = 1
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data2.create_train_val_data_loaders(args, logger)

    list_of_gt = {}
    list_of_pred = {}
    subj = 'Sonia'

    for cur_step, data_retval in enumerate(tqdm.tqdm(val_aug_loader)):

        threedskeleton = data_retval['3dskeleton']
        bboxes = data_retval['bboxes']
        predcam = data_retval['predcam']
        filename = data_retval['frame_paths'][0][0]
        exercise = filename.split("/")[8]
        print
        cur_subj = filename.split("/")[7]
        if subj == cur_subj:
            if exercise not in list_of_gt.keys():
                list_of_gt[exercise] = []
                list_of_pred[exercise] = []
            cur = time.time()
            twodskeleton = data_retval['2dskeleton']
            twodskeleton = twodskeleton.reshape(twodskeleton.shape[0],twodskeleton.shape[1],-1)


            threedskeleton = data_retval['3dskeleton']
            bboxes = data_retval['bboxes']
            predcam = data_retval['predcam']
            proj = 5000.0
            
            height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1]).to(device)
            center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1).to(device)
            focal=torch.tensor([[proj]]).to(device).repeat(height.shape[0],1)
            predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1).to(device)
            translation = convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
            reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3]).to(device)
            rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(reshapethreed.shape[0],1,1).to(device)
            focal=torch.tensor([[proj]]).to(device).repeat(translation.shape[0],1)
            imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(reshapethreed.shape[0],1).to(device)
            twodkpts, skeleton = perspective_projection(reshapethreed, rotation, translation.float(),focal[:,0], imgdimgs)
            #skeleton = skeleton.reshape(twodskeleton.shape[0],30,skeleton.shape[1],skeleton.shape[2])
            #skeleton = skeleton.reshape(skeleton.shape[0],skeleton.shape[1],-1)
            twodkpts = twodkpts.reshape(twodskeleton.shape[0],(args.step),twodkpts.shape[1],twodkpts.shape[2])
            divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodkpts.shape[0],twodkpts.shape[1],twodkpts.shape[2],1).to(device)
            twodkpts = twodkpts/divide
            #pdb.set_trace()
            #
            #if self.train_args.modelname != 'transf':
            twodkpts = twodkpts.reshape(twodskeleton.shape[0],twodkpts.shape[1],-1)
        
            bined_left_quad = data_retval['bined_left_quad']-1
            emggroundtruth = data_retval['emg_values'].to(device)
            cond = data_retval['cond'].to(device)
            emggroundtruth = emggroundtruth/100.0

            leftquad = data_retval['left_quad'].to(device)
            leftquad = leftquad/100.0
            leftquad[leftquad > 1.0] = 1.0
            bins = data_retval['bins']
            cond = data_retval['condval']
            #bined_left_quad = bined_left_quad.to(self.device)-1
            twodskeleton = twodskeleton.to(device)
            cur2 = time.time()
            #print(cur2-cur,"2")
            cur = cur2
            if args.modelname == 'transf':
                if args.predemg == 'True':
                    emg_output = model(twodkpts,cond) 
                else:
                    pose_pred = model(emggroundtruth,cond)
            else:
                emg_output = model(torch.unsqueeze(twodkpts.permute(0,2,1),dim=1))
            
            list_of_gt[exercise].append(emggroundtruth.reshape(-1).cpu().numpy())
            list_of_pred[exercise].append(emg_output.reshape(-1).detach().cpu().numpy())
            msel = torch.nn.MSELoss()
            #print(msel(torch.tensor(list_of_gt[exercise]),torch.tensor(list_of_pred[exercise])),exercise)

    msel = torch.nn.MSELoss()
    for exercise_key in list_of_gt.keys():
        print(torch.sqrt(msel(torch.tensor(list_of_gt[exercise_key])*100,torch.tensor(list_of_pred[exercise_key])*100)),exercise_key)


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


    logger = logvis.MyLogger(args, args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')


#TEST SKELETON MATRIX FLATTENED OVER TIME
#TEST EMG MATRIX FLATTENED OVER TIME