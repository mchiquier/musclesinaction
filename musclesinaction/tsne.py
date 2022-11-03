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
import tqdm
import musclesinaction.models.modelbert as transmodel
import musclesinaction.models.basicconv as convmodel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

    trainpaths = [
        '../../../vondrick/mia/VIBE/ignore/train_ignore_2096.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2097.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2098.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2099.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2100.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2101.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2103.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2104.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2105.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2107.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2108.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2109.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2110.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2111.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2112.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2113.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2125.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2126.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2129.txt',
    '../../../vondrick/mia/VIBE/ignore/train_ignore_2131.txt',
    ]
    

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
    """model_args = {'num_tokens': int(args.num_tokens),
        'dim_model': int(args.dim_model),
        'num_classes': int(args.num_classes),
        'num_heads': int(args.num_heads),
        'classif': args.classif,
        'num_encoder_layers':int(args.num_encoder_layers),
        'num_decoder_layers':int(args.num_decoder_layers),
        'dropout_p':float(args.dropout_p),
        'device': args.device,
        'embedding': args.embedding}"""

    model_args = {
        'device': args.device}
    my_model = convmodel.BasicConv(**model_args)
    #my_model = transmodel.TransformerEnc(**model_args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    my_model.load_state_dict(checkpoint['my_model'])
    my_model.eval()
    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)

    list_of_resultsnn = []
    list_of_results = []
    list_of_resultsnn = []
    (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args) = \
    data.create_train_val_data_loaders(args, logger)

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
    list_of_class = []
    #for trainpath in trainpaths:
    #    args.data_path_train = trainpath
    #    trainpath= args.data_path_train #= trainpath
    for cur_step, data_retval in enumerate(tqdm.tqdm(train_loader)):
        #pdb.set_trace()
        #ignoremovie = trainpath
        ignoremovie = data_retval['frame_paths'][0][0]
        ignoremovie =  ignoremovie.split("/")[-2].split("_")[1]
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

        twodkpts = twodkpts.reshape(threedskeleton.shape[0],twodkpts.shape[1],-1)
        emggroundtruth = data_retval['emg_values']
        emggroundtruth = emggroundtruth/100.0
        
        list_of_val_emg.append(emggroundtruth.numpy())
        list_of_val_skeleton.append(twodkpts.reshape(-1).numpy())
        #pdb.set_trace()
        list_of_class.append(int(ignoremovie))

    np_val_emg = np.array(list_of_val_emg)
    np_val_skeleton = np.array(list_of_val_skeleton)
    reshape_emg = np_val_emg.reshape(np_val_emg.shape[0],-1)
    reshape_skeleton = np_val_skeleton.reshape(np_val_skeleton.shape[0],-1)
    np_list_of_class = np.array(list_of_class)
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(reshape_emg)
    X_2d_skeleton = tsne.fit_transform(reshape_skeleton)
    names = sorted(set(list_of_class))
    cm = plt.get_cmap('gist_rainbow')
    markers = ['s','o','x']
    dict_color = {}
    dict_markers = {}
    for i in range(20):
        dict_color[names[i]] = cm(1.*(i+1)/20)
        dict_markers[names[i]] = markers[i%3]
        
    
    #pdb.set_trace()
    dict_of_muscles={}
    dict_of_muscles[2108]='Cross Leg Front'
    dict_of_muscles[2109]='Elbow Punch'
    dict_of_muscles[2096]='Jumping Jack'
    dict_of_muscles[2112]='Side Lunges'
    dict_of_muscles[2110]='Side Shuffle'
    dict_of_muscles[2104] = 'Slow Skater'
    dict_of_muscles[2099] = 'Front Kick'
    dict_of_muscles[2100] = 'Front Punch'
    dict_of_muscles[2097] = 'High Kick'
    dict_of_muscles[2101] = 'Hook Punch'
    dict_of_muscles[2129] = 'Knee Kick'
    dict_of_muscles[2098] = 'Leg Back'
    dict_of_muscles[2113] = 'Running'
    dict_of_muscles[2107] = 'Squats'
    dict_of_muscles[2125] = 'Ball Throw'
    dict_of_muscles[2111] = 'Baseball Bat'
    dict_of_muscles[2126] = 'Bowling'
    dict_of_muscles[2105] = 'Feet Cross'
    dict_of_muscles[2103] = 'Pirouette'
    dict_of_muscles[2131] = 'Woodchop'
    plt.figure()
    for i in range(X_2d.shape[0]):
        c = dict_color[list_of_class[i]]
        mrk = dict_markers[list_of_class[i]]
        plt.scatter(X_2d[i, 0], X_2d[i, 1], color=c, marker = mrk, label=list_of_class[i],s=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    newlabel = [dict_of_muscles[int(x)] for x in [*by_label]]
    plt.legend(by_label.values(), newlabel,loc='center left', bbox_to_anchor=(1, 0.5))
    print("here")
    


    listofkeys = [*by_label]
    plt.savefig("testtsne_emg3.png", bbox_inches='tight')

    plt.figure()
    for i in range(X_2d.shape[0]):
        c = dict_color[list_of_class[i]]
        mrk = dict_markers[list_of_class[i]]
        plt.scatter(X_2d_skeleton[i, 0], X_2d[i, 1], color=c, marker = mrk, label=list_of_class[i],s=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    newlabel = [dict_of_muscles[int(x)] for x in [*by_label]]
    #pdb.set_trace()
    plt.legend(by_label.values(), newlabel,loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig("testtsne_skeleton3.png", bbox_inches='tight')
    #pdb.set_trace()
    
    

    #np_train_emg = np.array(list_of_train_emg)
    np_val_skeleton = np.array(list_of_val_skeleton)
    np_val_class = np.array(list_of_class)
        #np_train_skeleton = np.array(list_of_train_skeleton)
    #nn = NearestNeighbor()
    #nn.train(np_train_skeleton,np_train_emg)
    #ypred = nn.predict(np_val_skeleton)
    #print(msel(torch.tensor(ypred)*100,torch.tensor(np_val_emg)*100))
    #list_of_results.append(msel(torch.tensor(ypred)*100,torch.tensor(np_val_emg)*100).numpy())
    pdb.set_trace()
    list_of_resultsnn.append(msel(torch.tensor(np_pred_emg)*100,torch.tensor(np_val_emg)*100).numpy())
    print(list_of_results)
    #print(list_of_resultsnn)
    print(np.mean(np.array(list_of_results)))
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

    logger = logvis.MyLogger(args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')


#TEST SKELETON MATRIX FLATTENED OVER TIME
#TEST EMG MATRIX FLATTENED OVER TIME