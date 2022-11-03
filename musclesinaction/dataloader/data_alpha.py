'''
Data loading and processing logic.
'''

import numpy as np
import random
import musclesinaction.utils.augs as augs
import utils
import pdb
import torch
import os
import pathlib
import json
import cv2
import matplotlib.pyplot as plt
import joblib
from matplotlib import animation
import time

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
    dset_args['step'] = int(args.step)
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
    train_loader_noshuffle = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    
   
    return (train_loader, train_loader_noshuffle, val_aug_loader, val_aug_loader, dset_args)


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

    def __init__(self, dataset_root, logger, phase, percent,  step,transform=None):
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

        # Load all file paths beforehand.
        # NOTE: This method call handles subdirectories recursively, but also creates extra files.
        #all_files = utils.cached_listdir(phase_dir, allow_exts=['jpg', 'jpeg', 'png'],
        #                                 recursive=True)
        self.phase = phase
        with open(dataset_root) as f:
            lines = f.readlines()
            self.all_files = lines
            file_count = len(self.all_files)
            print('Image file count:', file_count)
            self.dset_size = file_count
            self.file_count = file_count

        self.dataset_root = dataset_root
        self.logger = logger
        self.phase = phase
        self.phase_dir = phase_dir
        self.transform = transform
        self.percent = float(percent)
        self.step = int(step)
        self.maxemg = 100
        self.bins = np.linspace(0, self.maxemg, 20)
        self.log_dir = 'training_viz_digitized'
        self.plot = False
        self.muscles=['rightquad','leftquad','rightham','leftham','rightglutt','leftglutt','leftbicep','rightbicep']
        """self.videos = ['IMG_2419_30.MOV','IMG_2420_30.MOV','IMG_2422_30.MOV', 'IMG_2404_30.MOV',
        'IMG_2426_30.MOV','IMG_2405_30.MOV','IMG_2409_30.MOV','IMG_2410_30.MOV','IMG_2413_30.MOV','IMG_2424_30.MOV','IMG_2415_30.MOV','IMG_2406_30.MOV',
        'IMG_2407_30.MOV','IMG_2408_30.MOV','IMG_2416_30.MOV','IMG_2423_30.MOV',
        'IMG_2425_30.MOV','IMG_2096_30.MOV','IMG_2097_30.MOV','IMG_2099_30.MOV','IMG_2100_30.MOV','IMG_2101_30.MOV',
        'IMG_2104_30.MOV','IMG_2105_30.MOV','IMG_2108_30.MOV','IMG_2109_30.MOV','IMG_2110_30.MOV',
        'IMG_2111_30.MOV','IMG_2112_30.MOV','IMG_2125_30.MOV','IMG_2129_30.MOV','IMG_2098_30.MOV',
        'IMG_2103_30.MOV','IMG_2107_30.MOV','IMG_2113_30.MOV','IMG_2126_30.MOV','IMG_2131_30.MOV',
        'IMG_2403_30.MOV','IMG_2411_30.MOV','IMG_2412_30.MOV', 'IMG_2414_30.MOV',
        'IMG_2415_30.MOV']"""
        #self.videos = ['IMG_2403_30.MOV','IMG_2411_30.MOV','IMG_2412_30.MOV', 'IMG_2414_30.MOV',
        #'IMG_2415_30.MOV']
        """self.videos = ['IMG_2096_30.MOV','IMG_2097_30.MOV','IMG_2099_30.MOV','IMG_2100_30.MOV','IMG_2101_30.MOV',
        'IMG_2104_30.MOV','IMG_2105_30.MOV','IMG_2108_30.MOV','IMG_2109_30.MOV','IMG_2110_30.MOV',
        'IMG_2111_30.MOV','IMG_2112_30.MOV','IMG_2125_30.MOV','IMG_2129_30.MOV','IMG_2098_30.MOV',
        'IMG_2103_30.MOV','IMG_2107_30.MOV','IMG_2113_30.MOV','IMG_2126_30.MOV','IMG_2131_30.MOV']"""
        """self.videos = ['IMG_2419_30.MOV','IMG_2420_30.MOV','IMG_2422_30.MOV', 'IMG_2404_30.MOV',
        'IMG_2426_30.MOV','IMG_2405_30.MOV','IMG_2409_30.MOV','IMG_2410_30.MOV',
        'IMG_2411_30.MOV','IMG_2412_30.MOV','IMG_2403_30.MOV','IMG_2413_30.MOV',
        'IMG_2414_30.MOV','IMG_2424_30.MOV','IMG_2415_30.MOV','IMG_2406_30.MOV',
        'IMG_2407_30.MOV','IMG_2408_30.MOV','IMG_2416_30.MOV','IMG_2423_30.MOV',
        'IMG_2425_30.MOV'] """
        """self.videos = ['IMG_2478_30.MOV','IMG_2479_30.MOV','IMG_2480_30.MOV',
        'IMG_2487_30.MOV','IMG_2488_30.MOV','IMG_2489_30.MOV',
        'IMG_2471_30.MOV','IMG_2472_30.MOV','IMG_2473_30.MOV','IMG_2474_30.MOV',
        'IMG_2483_30.MOV','IMG_2484_30.MOV','IMG_2485_30.MOV','IMG_2486_30.MOV']"""
        self.videos = ['IMG_squatright_30.MOV', 'IMG_squatwrong_30.MOV']
        self.pickledict = {}
        for elem in self.videos:
            f = open('../squatdataset/' + elem.replace(".","_") + '/alphapose-results.json')
            self.pickledict[elem.split("_")[1]] = json.load(f)
            

        #self.pathtopklone = '../../../vondrick/mia/VIBE/' + 'output/IMG_1196_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkltwo = '../../../vondrick/mia/VIBE/' + 'output/IMG_1197_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkthree = '../../../vondrick/mia/VIBE/' + 'output/IMG_1203_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkfour = '../../../vondrick/mia/VIBE/' + 'output/IMG_1902_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkfive = '../../../vondrick/mia/VIBE/' + 'output/IMG_1903_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopksix = '../../../vondrick/mia/VIBE/' + 'output/IMG_1905_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkseven = '../../../vondrick/mia/VIBE/' + 'output/IMG_1906_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkeight = '../../../vondrick/mia/VIBE/' + 'output/IMG_1919_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopknine = '../../../vondrick/mia/VIBE/' + 'output/IMG_1918_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkten = '../../../vondrick/mia/VIBE/' + 'output/IMG_1920_30.MOV/vibe_output.pkl'#filepath[1]
        #self.pathtopkeleven= '../../../vondrick/mia/VIBE/' + 'output/IMG_1921_30.MOV/vibe_output.pkl'#filepath[1]
        
        """self.totalone = joblib.load(self.pathtopklone)
        self.totaltwo = joblib.load(self.pathtopkltwo)
        self.totalthree = joblib.load(self.pathtopkthree)
        self.totalfour = joblib.load(self.pathtopkfour)
        self.totalfive = joblib.load(self.pathtopkfive)
        self.totalsix = joblib.load(self.pathtopksix)
        self.totalseven = joblib.load(self.pathtopkseven)
        self.totaleight = joblib.load(self.pathtopkeight)
        self.totalnine = joblib.load(self.pathtopknine)
        self.totalten = joblib.load(self.pathtopkten)
        self.totaleleven = joblib.load(self.pathtopkeleven)"""

    def __len__(self):
        return int((self.dset_size-30)*self.percent)

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

    def visualize_video(self,index):
        #index = 5100
        current_path = self.log_dir + "/" + str(index)
        cur = time.time()
        os.makedirs(current_path, 0o777, exist_ok=True)
        file_idx = index
        filepath = self.all_files[file_idx].split(",")
        cur2 = time.time()
        #print(cur2-cur, "1")
        cur = cur2
        
        pathtoframes = '../../../vondrick/mia/VIBE/' + filepath[0]
        #print(filepath[0],filepath[1])
        
        cur2 = time.time()
        #print(cur2-cur, "2")
        cur = cur2
        list_of_emg_values_rightquad = []
        list_of_emg_values_rightham = []
        list_of_emg_values_rightbicep = []
        list_of_emg_values_leftquad = []
        list_of_emg_values_leftham = []
        list_of_emg_values_leftbicep = []
        list_of_emg_values_righttricep = []
        list_of_emg_values_lefttricep = []
        list_of_2d_joints = []
        list_of_3d_joints = []
        list_of_frames = []
        list_of_bboxes = []
        list_of_predcam = []
        list_of_frame_paths = []
        list_of_orig_cam = []
        list_of_verts = []
        filepath = self.all_files[index].split(",")
        for i in range(self.step):
            frame1=pathtoframes + "/" + filepath[2+i*13].zfill(6) + ".png"
            frame2=pathtoframes +  "/" + filepath[3+i*13].zfill(6) + ".png"
            frame3=pathtoframes +  "/" + filepath[4+i*13].zfill(6) + ".png"
            list_of_frame_paths.append(frame1)
            pickleframe1= int(filepath[5+i*13].split("/")[-1])
            pickleframe2=int(filepath[6+i*13].split("/")[-1])
            pickleframe3=int(filepath[7+i*13].split("/")[-1])
            emgvalues = filepath[8+i*13:13+i*13]
            emgvalues[-1] = emgvalues[-1].split("\n")[0]
            list_of_emg_values_rightquad.append(float(emgvalues[0]))
            list_of_emg_values_rightham.append(float(emgvalues[2]))
           
            list_of_emg_values_leftquad.append(float(emgvalues[1]))
            list_of_emg_values_leftham.append(float(emgvalues[3]))
        

            total = self.pickledict[pathtoframes.split("/")[-1].split("_")[1]]
            
            """if '1197' in pathtoframes:
                total = self.totaltwo
            if '1196' in pathtoframes:
                total = self.totalone
            if '1203' in pathtoframes:
                total = self.totalthree
            if '1902' in pathtoframes:
                total = self.totalfour
            if '1903' in pathtoframes:
                total = self.totalfive
            if '1905' in pathtoframes:
                total = self.totalsix
            if '1906' in pathtoframes:
                total = self.totalseven
            if '1919' in pathtoframes:
                total = self.totaleight
            if '1918' in pathtoframes:
                total = self.totalnine
            if '1920' in pathtoframes:
                total = self.totalten
            if '1921' in pathtoframes:
                total = self.totaleleven"""
            
            
            firstjoints2dframe= total[pickleframe1]['keypoints']
            firstjoints2dframe = np.array(firstjoints2dframe).reshape(26,3)[:,:2]
            order = [0,18,6,8,10,5,7,9,19,19,14,16,19,13,15,2,1,4,3,20,22,24,21,23,25]
            openposeorder = np.array([firstjoints2dframe[i,:] for i in order])
            list_of_2d_joints.append(openposeorder)
            second2djoints2dframe = total[pickleframe2]['keypoints']
            third2djoints2dframe = total[pickleframe3]['keypoints']


        emg_values = [list_of_emg_values_rightquad,
        list_of_emg_values_leftquad]

        #emg_values = [list_of_emg_values_rightquad,list_of_emg_values_rightham,
        #list_of_emg_values_leftquad,list_of_emg_values_leftham]


        #emg_values.pop(5)
        #emg_values.pop(1)
        return (emg_values,list_of_2d_joints, list_of_frame_paths)

    def __getitem__(self, index):
        
        cur = time.time()
        (list_of_emg_values, twod_joints, list_of_frame_paths) = self.visualize_video(index)
        cur2 = time.time()
        #print(cur2-cur,"0")
        cur = cur2
        #list_of_frames=np.array(list_of_frames)
        twod_joints=np.array(twod_joints)
        
        emg_values_right_quad = np.array(list_of_emg_values)[0]
        bined_right_quad = np.digitize(emg_values_right_quad,self.bins)
        emg_values_left_quad = np.array(list_of_emg_values)[0]
        bined_left_quad = np.digitize(emg_values_left_quad,self.bins)

        # Return results.
        cur2 = time.time()
        #print(cur2-cur,"1")
        name = list_of_frame_paths[0].split("/")[-2].split("_")[1]
        if name[2] == '4':
            cond = np.array([0.0]) 
        else:
            cond = np.array([1.0])
        
        cur = cur2
        result = {'bined_left_quad': bined_left_quad,  
                  'bined_right_quad': bined_right_quad,
                  'left_quad': emg_values_left_quad,
                  'emg_values': np.array(list_of_emg_values),
                  'right_quad':emg_values_right_quad,  
                  '2dskeleton': twod_joints,
                  'cond': cond,
                  'frame_paths': list_of_frame_paths}
        return result

