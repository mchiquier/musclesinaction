import numpy as np
import pdb
import torch
import musclesinaction.configs.args as args
import vis.logvis as logvis
import musclesinaction.dataloader.data as data
import time
import os
import musclesinaction.models.modelemgtopose as transmodelemgtopose
import musclesinaction.models.model as transmodelposetoemg
import random
import torch
import tqdm
from collections import OrderedDict
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
    my_model = transmodelposetoemg.TransformerEnc(**model_args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    my_model.load_state_dict(checkpoint['my_model'])
    my_model.to('cuda')

    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    my_model.transformer0.layers[-2].register_forward_hook(get_activation('encoder_penultimate_layer'))
    activation = {}


    list_of_train_emg = []
    list_of_train_skeleton = []
    list_of_val_emg = []
    list_of_val_skeleton = []
    list_of_pred_emg = []
    
    list_of_class = []
    #for trainpath in trainpaths:
    #    args.data_path_train = trainpath
    #    trainpath= args.data_path_train #= trainpath
    for cur_step, data_retval in enumerate(tqdm.tqdm(val_aug_loader)):
        #pdb.set_trace()
        #ignoremovie = trainpath
        framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
        ex = framelist[0].split("/")[8]
        if 'RonddeJambe' in ex:
            ex = 'RonddeJambe'
        if 'Squat' in ex:
            ex = 'Squat'
        threedskeleton = data_retval['3dskeleton']
        bboxes = data_retval['bboxes']
        predcam = data_retval['predcam']

        emggroundtruth = data_retval['emg_values']
        emggroundtruth = emggroundtruth/100.0

        #threedskeleton = threedskeleton.reshape(threedskeleton.shape[0],-1)

        emggroundtruth = data_retval['emg_values']
        emggroundtruth = emggroundtruth
        cond = data_retval['condval']
        badcond = data_retval['condvalbad']
        #bined_left_quad = bined_left_quad.to(self.device)-1
        skeleton = threedskeleton
        cur2 = time.time()
        #print(cur2-cur,"2")
        cur = cur2
        skeleton = skeleton.reshape(skeleton.shape[0],skeleton.shape[1],-1)
        #pdb.set_trace()
        activation = {}
    
        emg_output = my_model(skeleton.to('cuda'),cond.to('cuda')) 
           
        
        list_of_val_emg.append(activation['encoder_penultimate_layer'].reshape(-1).detach().cpu().numpy())
        list_of_val_skeleton.append(threedskeleton.reshape(-1).numpy())
        #pdb.set_trace()
        list_of_class.append(ex)

    np_val_emg = np.array(list_of_val_emg)
    np_val_skeleton = np.array(list_of_val_skeleton)
    reshape_emg = np_val_emg.reshape(np_val_emg.shape[0],-1)
    reshape_skeleton = np_val_skeleton.reshape(np_val_skeleton.shape[0],-1)
    tsne = TSNE(n_components=2, random_state=100)
    X_2d = tsne.fit_transform(reshape_emg)
    X_2d_skeleton = tsne.fit_transform(reshape_skeleton)
    names = sorted(set(list_of_class))
    cm = plt.get_cmap('gist_rainbow')
    markers = ['s','o','x']
    dict_color = {}


    ex_type = {}
    dict_markers = {}
    for i in range(15):
        dict_color[names[i]] = cm(1.*(i+1)/15)
        ex_type[names[i]] = names[i]
        
    
    plt.figure()
    for i in range(X_2d.shape[0]):
        c = dict_color[list_of_class[i]]
        plt.scatter(X_2d[i, 0], X_2d[i, 1], color=c, label=ex_type[list_of_class[i]],s=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left',bbox_to_anchor=(1, 0.5))
    #plt.legend(by_label.values(), newlabel,loc='center left', bbox_to_anchor=(1, 0.5))
    print("here")
    


    listofkeys = [*by_label]
    plt.savefig("tsne_final_exercises_features.png", bbox_inches='tight')
    pdb.set_trace()

    plt.figure()
    for i in range(X_2d.shape[0]):
        c = dict_color[list_of_class[i]]
        plt.scatter(X_2d_skeleton[i, 0], X_2d_skeleton[i, 1], color=c, label=ex_type[list_of_class[i]],s=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left',bbox_to_anchor=(1, 0.5))
    
    plt.savefig("tsne_final_exercises_classes.png", bbox_inches='tight')
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

    args = args.inference_args()
    args.bs = 1

    logger = logvis.MyLogger(args, args, context='train')

    try:

        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')


#TEST SKELETON MATRIX FLATTENED OVER TIME
#TEST EMG MATRIX FLATTENED OVER TIME