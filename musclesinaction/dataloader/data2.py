'''
Data loading and processing logic.
'''

import numpy as np
import random
import musclesinaction.utils.augs as augs
import torch
import os

import matplotlib.pyplot as plt
from matplotlib import animation

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def _read_image_robust(img_path, no_fail=False):
    '''
    Loads and returns an image that meets conditions along with a success flag, in order to avoid
    crashing.
    '''
    try:
        image = plt.imread(img_path).copy()
        success = True
        if (image.ndim != 3 or image.shape[2] != 3
                or np.any(np.array(image.strides) < 0)):
            # Either not RGB or has negative stride, so discard.
            success = False
            if no_fail:
                raise RuntimeError(f'ndim: {image.ndim}  '
                                   f'shape: {image.shape}  '
                                   f'strides: {image.strides}')

    except IOError as e:
        # Probably corrupt file.
        image = None
        success = False
        if no_fail:
            raise e

    return image, success


def _seed_worker(worker_id):
    '''
    Ensures that every data loader worker has a separate seed with respect to NumPy and Python
    function calls, not just within the torch framework. This is very important as it sidesteps
    lack of randomness- and augmentation-related bugs.
    '''
    worker_seed = torch.initial_seed() % (2 ** 32)  # This is distinct for every worker.
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''

    # TODO: Figure out noaug val dataset args as well.
    #my_transform = augs.get_train_transform(args.image_dim)
    dset_args = dict()
    dset_args['percent'] = args.percent
    dset_args['std'] = args.std
    dset_args['step'] = int(args.step)
    dset_args['cond'] = args.cond
    #dset_args['transform'] = my_transform

    train_dataset = MyMuscleDataset(
        args.data_path_train, logger, 'train', **dset_args)

    #validations = os.listdir(args.data_path_val)
    
    
    val_aug_dataset = MyMuscleDataset(
        args.data_path_val, logger, 'val', **dset_args)
    val_aug_loader = torch.utils.data.DataLoader(
    val_aug_dataset, batch_size=args.bs, num_workers=args.num_workers,
    shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)


    #first = int(len(dataset)*0.8)
    #second = len(dataset) - first
    #train_dataset, val_aug_dataset = torch.utils.data.random_split(dataset, [first, second])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
       
    return (train_loader, train_loader, val_aug_loader, val_aug_loader, dset_args)

def create_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''

    # TODO: Figure out noaug val dataset args as well.
    #my_transform = augs.get_train_transform(args.image_dim)
    dset_args = dict()
    dset_args['percent'] = args.percent
    dset_args['std'] = args.std
    dset_args['step'] = int(args.step)
    dset_args['cond'] = args.cond
    #dset_args['transform'] = my_transform
    #validations = os.listdir(args.data_path_val)
    
    
    val_aug_dataset = MyMuscleDataset(
        args.data_path_val, logger, 'val', **dset_args)
    val_aug_loader = torch.utils.data.DataLoader(
    val_aug_dataset, batch_size=args.bs, num_workers=args.num_workers,
    shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
       
    return (val_aug_loader, val_aug_loader, val_aug_loader, val_aug_loader, dset_args)


def create_test_data_loader(train_args, test_args, train_dset_args, logger):
    '''
    return (test_loader, test_dset_args).
    '''

    my_transform = augs.get_test_transform(train_args.image_dim)

    test_dset_args = copy.deepcopy(train_dset_args)
    test_dset_args['transform'] = my_transform

    test_dataset = MyImageDataset(
        test_args.data_path, logger, 'test', **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)

    return (test_loader, test_dset_args)


class MyMuscleDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class that returns uniformly random images of a labelled or unlabelled image
    dataset.
    '''

    def __init__(self, dataset_root, logger, phase, percent,  step, std, cond, transform=None):
        '''
        :param dataset_root (str): Path to dataset (with or without phase).
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        :param transform: Data transform to apply on every image.
        '''
        # Get root and phase directories.
        phase_dir = os.path.join(dataset_root, phase)
        if not os.path.exists(phase_dir):
            # We may already be pointing to a phase directory (which, in the interest of
            # flexibility, is not necessarily the same as the passed phase argument).
            phase_dir = dataset_root
            #dataset_root = str(pathlib.Path(dataset_root).parent)

        self.phase = phase
        with open(dataset_root) as f:
            lines = f.readlines()
            self.all_files = lines
            file_count = len(self.all_files)
            print('File count:', file_count)
            self.dset_size = file_count
            self.file_count = file_count

        self.std = std
        self.dataset_root = dataset_root
        self.logger = logger
        self.phase = phase
        self.phase_dir = phase_dir
        self.cond = cond
        self.transform = transform
        self.percent = float(percent)
        self.step = int(step)
        self.maxemg = 100
        self.bins = np.linspace(0, self.maxemg, 20)
        self.log_dir = 'training_viz_digitized'
        self.plot = False
        self.muscles=['rightquad','leftquad','rightham','leftham','rightglutt','leftglutt','leftbicep','rightbicep']


    def __len__(self):
        return int((self.dset_size)*self.percent)

    def animate(self, list_of_data, labels, part, trialnum, current_path):
    
        #pdb.set_trace()
        t = np.linspace(0, len(list_of_data[0])/10.0, len(list_of_data[0]))
        numDataPoints = len(t)
        colors = ['b','r','c','g']
        
            #ax.set_ylabel('y')

        def animate_func(num):
            print(num, "hi")
            ax.clear()  # Clears the figure to update the line, point,   
            for i,limb in enumerate(list_of_data):
                ax.plot(t[:num],limb[:num], c=colors[i], label=labels[i])
            #ax.plot(t[:num],dataSetlefttricep[:num], c='red', label='right tricep')
            ax.legend(loc="upper left")

            #ax.plot(t[0],dataSet[0],     
            #        c='black', marker='o')
            # Adding Figure Labels
            ax.set_title('Trajectories of ' + part + ' \nTime = ' + str(np.round(t[num],    
                        decimals=2)) + ' sec')
            ax.set_xlabel('x')
            ax.set_ylim([0, self.maxemg])
            
        fig, ax = plt.subplots()
        print(numDataPoints)
        line_ani = animation.FuncAnimation(fig, animate_func, interval=100,   
                                        frames=numDataPoints)
        FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'h264_v4l2m2m'])
        line_ani.save(current_path + "/" + str(part) + '_emg.mp4')

    
    def __getitem__(self, index):
        
        filepath = self.all_files[index].split("\n")[0]
        if self.std == "True":
            emg_values = np.load("../../../vondrick/mia/VIBE/" + filepath + "/emgvaluesstd.npy")
        else:
            emg_values = np.load("../../../vondrick/mia/VIBE/" + filepath + "/emgvalues.npy")
        twod_joints = np.load("../../../vondrick/mia/VIBE/" + filepath + "/joints2d.npy")
        threed_joints = np.load("../../../vondrick/mia/VIBE/" + filepath + "/joints3d.npy")
        predcam = np.load("../../../vondrick/mia/VIBE/" + filepath + "/predcam.npy")
        origcam = np.load("../../../vondrick/mia/VIBE/" + filepath + "/origcam.npy")
        bboxes = np.load("../../../vondrick/mia/VIBE/" + filepath + "/bboxes.npy")
        pose = np.load("../../../vondrick/mia/VIBE/" + filepath + "/pose.npy")
        twodskeletonsmpl = np.load("../../../vondrick/mia/VIBE/" + filepath + "/joints2dsmpl.npy")
        betas = np.load("../../../vondrick/mia/VIBE/" + filepath + "/betas.npy")
        verts = np.load("../../../vondrick/mia/VIBE/" + filepath + "/verts.npy")

        file = open("../../../vondrick/mia/VIBE/" + filepath + "/frame_paths.txt", 'r')
        line = file.readlines()[0]
        frame_paths = line.rstrip().split(',')
            
        person = filepath.split("/")[2]

        if person == 'David':
            themax = np.array([226.0,159.0,283.0,233.0,406.0,139.0,276.0,235.0])
            themin = np.array([7.0,8.0,9.0,2.0,9.0,10.0,8.0,2.0])
        elif person == 'Ishaan':
            themax = np.array([355.0,231.0,242.0,128.0,473.0,183.0,197.0,98.0])
            themin = np.array([3.0,2.0,2.0,2.0,2.0,3.0,2.0,2.0])
        elif person == "Jo":
            themax = np.array([119.0,178.0,83.0,102.0,176.0,106.0,95.0,75.0])
            themin = np.array([2.0,2.0,2.0,1.0,4.0,3.0,2.0,1.0])
        elif person == "Jonny":
            themax = np.array([207.0,97.0,154.0,112.0,182.0,122.0,176.0,123.0])
            themin = np.array([2.0,3.0,2.0,1.0,3.0,3.0,3.0,2.0])
        elif person == "Lionel":
            themax = np.array([177.0,125.0,85.0,167.0,176.0,110.0,130.0,199.0])
            themin = np.array([2.0,3.0,0.0,1.0,1.0,2.0,2.0,3.0])
        elif person == "Me":
            themax = np.array([213.0,115.0,192.0,128.0,207.0,147.0,218.0,114.0])
            themin = np.array([2.0,1.0,2.0,1.0,2.0,2.0,4.0,1.0])
        elif person == "Samir":
            themax = np.array([289.0,141.0,116.0,179.0,452.0,174.0,135.0,177.0])
            themin = np.array([1.0,3.0,2.0,7.0,2.0,1.0,3.0,4.0])
        elif person == "Serena":
            themax = np.array([177.0,120.0,175.0,146.0,147.0,94.0,243.0,209.0])
            themin = np.array([3.0,2.0,2.0,6.0,2.0,0.0,2.0,6.0])
        elif person == "Sonia":
            themax = np.array([154.0,53.0,74.0,102.0,183.0,59.0,152.0,135.0])
            themin = np.array([2.0,3.0,2.0,2.0,3.0,1.0,2.0,2.0])
        else:
            themax = np.array([174.0,134.0,125.0,151.0,170.0,161.0,119.0,137.0])
            themin = np.array([3.0,14.0,4.0,3.0,7.0,5.0,8.0,5.0])

        if self.cond=='True':
            if person == 'David':
                condval = np.array([1.0])
                condvalbad = np.array([0.1])
            elif person == 'Ishaan':
                condval = np.array([0.9]) 
                condvalbad = np.array([0.3])
            elif person == "Jo":
                condval = np.array([0.8]) 
                condvalbad = np.array([0.6])
            elif person == "Jonny":
                condval = np.array([0.7])
                condvalbad = np.array([0.6]) 
            elif person == "Lionel":
                condval = np.array([0.6])
                condvalbad = np.array([0.1]) 
            elif person == "Me":
                condval = np.array([0.5])
                condvalbad = np.array([0.6]) 
            elif person == "Samir":
                condval = np.array([0.4])
                condvalbad = np.array([0.6]) 
            elif person == "Serena":
                condval = np.array([0.3])
                condvalbad = np.array([0.6]) 
            elif person == "Sonia":
                condval = np.array([0.2])
                condvalbad = np.array([0.6]) 
            else:
                condval = np.array([0.1]) 
                condvalbad = np.array([0.9]) 
        else:
            condval = np.array([0.1])
            condvalbad = np.array([0.9])
           
        result = {'condval':condval,
                'condvalbad':condvalbad,
                'max': themax,
                'min':themin,
                  'emg_values': emg_values.transpose(1,0),
                  'predcam': predcam,
                  'origcam': origcam,
                  'bboxes': bboxes,
                  'pose':pose,
                  'betas': betas,
                  'verts': verts,
                  '2dskeletonsmpl': twodskeletonsmpl,
                  '2dskeleton': twod_joints[:,:25,:],
                  '3dskeleton': threed_joints[:,:25,:],
                  'frame_paths': frame_paths,
                  'indexpath': filepath.split("/")[-1]}
        return result

