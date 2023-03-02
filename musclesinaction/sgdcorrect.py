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
import matplotlib.pyplot as plt
import cv2
import musclesinaction.models.modelbert as transmodel
import musclesinaction.models.model as model
import musclesinaction.models.basicconv as convmodel
from torch.autograd import Variable

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

def plot_skel(keypoints, img, currentpath, i, markersize=5, linewidth=2, alpha=0.7):
    plt.figure()
    #pdb.set_trace()
    keypoints = keypoints[:25]
    plt.imshow(img)
    limb_seq = [([17, 15], [238, 0, 255]),
                ([15, 0], [255, 0, 166]),
                ([0, 16], [144, 0, 255]),
                ([16, 18], [65, 0, 255]),
                ([0, 1], [255, 0, 59]),
                ([1, 2], [255, 77, 0]),
                ([2, 3], [247, 155, 0]),
                ([3, 4], [255, 255, 0]),
                ([1, 5], [158, 245, 0]),
                ([5, 6], [93, 255, 0]),
                ([6, 7], [0, 255, 0]),
                ([1, 8], [255, 21, 0]),
                ([8, 9], [6, 255, 0]),
                ([9, 10], [0, 255, 117]),
                # ([10, 11]], [0, 252, 255]),  # See comment above
                ([10, 24], [0, 252, 255]),
                ([8, 12], [0, 140, 255]),
                ([12, 13], [0, 68, 255]),
                ([13, 14], [0, 14, 255]),
                # ([11, 22], [0, 252, 255]),
                # ([11, 24], [0, 252, 255]),
                ([24, 22], [0, 252, 255]),
                ([24, 24], [0, 252, 255]),
                ([22, 23], [0, 252, 255]),
                ([14, 19], [0, 14, 255]),
                ([14, 21], [0, 14, 255]),
                ([19, 20], [0, 14, 255])]

    colors_vertices = {0: limb_seq[4][1],
                    1: limb_seq[11][1],
                    2: limb_seq[5][1],
                    3: limb_seq[6][1],
                    4: limb_seq[7][1],
                    5: limb_seq[8][1],
                    6: limb_seq[9][1],
                    7: limb_seq[10][1],
                    8: limb_seq[11][1],
                    9: limb_seq[12][1],
                    10: limb_seq[13][1],
                    11: limb_seq[14][1],
                    12: limb_seq[15][1],
                    13: limb_seq[16][1],
                    14: limb_seq[17][1],
                    15: limb_seq[1][1],
                    16: limb_seq[2][1],
                    17: limb_seq[0][1],
                    18: limb_seq[3][1],
                    19: limb_seq[21][1],
                    20: limb_seq[23][1],
                    21: limb_seq[22][1],
                    22: limb_seq[18][1],
                    23: limb_seq[20][1],
                    24: limb_seq[19][1]}

    alpha = alpha
    for vertices, color in limb_seq:
        if keypoints[vertices[0]].mean() != 0 and keypoints[vertices[1]].mean() != 0:
            #pdb.set_trace()
            plt.plot([keypoints[vertices[0]][0], keypoints[vertices[1]][0]],
                    [keypoints[vertices[0]][1], keypoints[vertices[1]][1]], linewidth=linewidth,
                    color=[j / 255 for j in color], alpha=alpha)
    # plot kp
    #set_trace()
    for k in range(len(keypoints)):
        if keypoints[k].mean() != 0:
            plt.plot(keypoints[k][0], keypoints[k][1], 'o', markersize=markersize,
                    color=[j / 255 for j in colors_vertices[k]], alpha=alpha)
    #pdb.set_trace()
    plt.axis('off')
    plt.savefig( currentpath + "/" + 'skeletonimgs/' + str(i).zfill(6) + ".png",bbox_inches='tight', pad_inches = 0)


def main(args, logger):


    
    #trainpaths = ['../../../vondrick/mia/VIBE/ignore/train_ignore_2109.txt']

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
    model_args = {'num_tokens': int(args.num_tokens),
        'dim_model': int(args.dim_model),
        'num_classes': int(args.num_classes),
        'num_heads': int(args.num_heads),
        'classif': args.classif,
        'num_encoder_layers':int(args.num_encoder_layers),
        'num_decoder_layers':int(args.num_decoder_layers),
        'dropout_p':float(args.dropout_p),
        'device': args.device,
        'step': int(args.step),
        'embedding': args.embedding}

    #model_args = {
    #    'device': args.device}
    #my_model = convmodel.BasicConv(**model_args)
    my_model = model.TransformerEnc(**model_args)
    checkpoint = torch.load(args.resume, map_location='cuda')
    my_model.load_state_dict(checkpoint['my_model'])
    my_model.eval()
    my_model = my_model.to('cuda')
    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data2.create_train_val_data_loaders(args, logger)

    list_of_resultsnn = []
    list_of_results = []
    list_of_resultsnn = []
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
    data2.create_train_val_data_loaders(args, logger)

    list_of_train_emg = []
    list_of_train_skeleton = []
    list_of_val_emg = []
    list_of_val_skeleton = []
    list_of_pred_emg = []
    """for cur_step, data_retval in enumerate(tqdm.tqdm(train_loader)):
        
        threedskeleton = data_retval['3dskeleton']
        bboxes = data_retval['bboxes']
        predcam = data_retval['predcam']
        proj = 5000.0
        
        #pdb.set_trace()
        height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1])
        center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1)
        focal=torch.tensor([[proj]]).repeat(height.shape[0],1)
        predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1)
        translation = convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
        reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3])
        rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(reshapethreed.shape[0],1,1)
        focal=torch.tensor([[proj]]).repeat(translation.shape[0],1)
        imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(reshapethreed.shape[0],1)
        twodkpts, skeleton = perspective_projection(reshapethreed, rotation, translation.float(),focal[:,0], imgdimgs)
        twodkpts = twodkpts.reshape(threedskeleton.shape[0],30,twodkpts.shape[1],twodkpts.shape[2])
        divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodkpts.shape[0],twodkpts.shape[1],twodkpts.shape[2],1)

            #pdb.set_trace()
        twodkpts = twodkpts/divide
        #twodkpts = twodkpts.reshape(threedskeleton.shape[0],twodkpts.shape[1],-1)
        emggroundtruth = data_retval['emg_values']
        emggroundtruth = emggroundtruth/100.0
        list_of_train_emg.append(emggroundtruth.reshape(-1).numpy())
        list_of_train_skeleton.append(twodkpts.reshape(-1).numpy())"""
    
    list_of_train_emg = []
    list_of_train_skeleton = []
    
    list_of_val_emg = []
    list_of_val_emg_notleft = []
    list_of_val_skeleton = []
    list_of_pred_emg = []
    list_of_pred_emg_notleft = []
    ignoremovie = '2099'
    #my_model.load_state_dict(checkpoint['my_model'])
    #my_model.eval()
    
        

    for cur_step, data_retval in enumerate(tqdm.tqdm(val_aug_loader)):

        if ignoremovie in data_retval['frame_paths'][0][0]:
            framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
            threedskeleton = data_retval['3dskeleton']
            bboxes = data_retval['bboxes']
            predcam = data_retval['predcam']
            proj = 5000.0
        
            #pdb.set_trace()
            height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1])
            center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1)
            focal=torch.tensor([[proj]]).repeat(height.shape[0],1)
            predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1)
            translation = convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
            reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3])
            rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(reshapethreed.shape[0],1,1)
            focal=torch.tensor([[proj]]).repeat(translation.shape[0],1)
            imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(reshapethreed.shape[0],1)
            twodkpts, skeleton = perspective_projection(reshapethreed, rotation, translation.float(),focal[:,0], imgdimgs)
            
            twodkpts = twodkpts.reshape(threedskeleton.shape[0],30,twodkpts.shape[1],twodkpts.shape[2])
            divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodkpts.shape[0],twodkpts.shape[1],twodkpts.shape[2],1)
                
            #pdb.set_trace()
            twodkpts = twodkpts/divide
            
            
            emggroundtruth = data_retval['emg_values']
            emggroundtruth = emggroundtruth/100.0

            gt = [0.880, 0.840, 0.590, 0.500, 0.380, 0.300, 0.370, 0.460, 0.550, 0.590,
            0.490, 0.420, 0.370, 0.440, 0.780, 1.030, 0.990, 0.750, 0.580, 0.450,
            0.340, 0.520, 0.530, 0.590, 0.640, 0.540, 0.540, 0.570, 0.800, 1.060]
            
            #torch.unsqueeze(twodkpts.permute(0,2,1),dim=1)
            #pdb.set_trace()
            oldtwodkpts = twodkpts

            outgt = torch.sum(torch.tensor(gt).to('cuda'))
            twodkpts = twodkpts.to('cuda')
            msel = torch.nn.MSELoss()
            for i in range(10000):
            
                
                added = Variable(torch.rand((1,30,6,2)).to('cuda'),requires_grad=True)*0.000001
                #optimizer = torch.optim.SGD([added], lr=0.000001)
                
                added.retain_grad()
                current = twodkpts.detach()
                current[:,:,12:15,:] = current[:,:,12:15,:] + added[:,:,:3,:]
                current[:,:,19:22,:] = current[:,:,19:22,:]+ added[:,:,3:,:]

                #twodkpts = twodkpts.reshape(threedskeleton.shape[0],twodkpts.shape[1],-1)
                emg_output = my_model(current.reshape(threedskeleton.shape[0],twodkpts.shape[1],-1))
                
                outpred = torch.sum(emg_output[0,1,:])
                
                
                loss = msel(emg_output[0,1,:],torch.tensor(gt).to('cuda'))
                loss.backward()
                #pdb.set_trace()
                twodkpts[:,:,12:15,:] = twodkpts[:,:,12:15,:] - 0.001*(added.grad.detach()[:,:,:3,:])
                twodkpts[:,:,19:22,:] = twodkpts[:,:,19:22,:] - 0.001*(added.grad.detach()[:,:,3:,:])

                
                #optimizer.step()
                print(loss)

            pdb.set_trace()
            j=3
            twodkpts = twodkpts.reshape(threedskeleton.shape[0],30,25,2)
            multiply = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodkpts.shape[0],twodkpts.shape[1],twodkpts.shape[2],1).to('cuda')
            twodkpts = twodkpts*multiply
            for j in range(len(framelist)):
                img=cv2.imread(framelist[j])
                img = (img*1.0).astype('int')
            
                twodskeleton = twodkpts[0]
                cur_skeleton = twodskeleton[j].cpu().numpy()
                #blurimg = img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100]
                #if int(cur_skeleton[0][1]) - 100 > 0 and int(cur_skeleton[0][1]) + 100 < img.shape[0] and int(cur_skeleton[0][0]) - 100 > 0 and int(cur_skeleton[0][0]) + 100 <img.shape[1]:
                #    blurred_part = cv2.blur(blurimg, ksize=(50, 50))
                #    img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100] = blurred_part
                plot_skel(cur_skeleton, img, "test", j)
            pdb.set_trace()
            #emg_output = my_model(torch.unsqueeze(twodkpts.permute(0,2,1),dim=1))#twodkpts)
            #emg_output = emg_output.permute(0,2,1)
            #pdb.set_trace()

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

    logger = logvis.MyLogger(args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')