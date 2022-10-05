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
    #dset_args['transform'] = my_transform

    train_dataset = MyMuscleDataset(
        args.data_path_train, logger, 'train', **dset_args)

    val_aug_dataset = MyMuscleDataset(
        args.data_path_val, logger, 'val', **dset_args)
    #first = int(len(dataset)*0.8)
    #second = len(dataset) - first
    #train_dataset, val_aug_dataset = torch.utils.data.random_split(dataset, [first, second])
    val_noaug_dataset = val_aug_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    train_loader_noshuffle = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_aug_loader = torch.utils.data.DataLoader(
        val_aug_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_noaug_loader = torch.utils.data.DataLoader(
        val_noaug_dataset, batch_size=args.bs, num_workers=args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) 
    return (train_loader, train_loader_noshuffle, val_aug_loader, val_noaug_loader, dset_args)


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

    def __init__(self, dataset_root, logger, phase, percent, transform=None):
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
        self.step = 30
        self.maxemg = 100
        self.bins = np.linspace(0, self.maxemg, 20)
        self.log_dir = 'training_viz_digitized'
        self.plot = False
        self.muscles=['rightquad','leftquad','rightham','leftham','rightglutt','leftglutt','leftbicep','rightbicep']
        self.pathtopklone = '../../../vondrick/mia/VIBE/' + 'output/IMG_1196_30.MOV/vibe_output.pkl'#filepath[1]
        self.pathtopkltwo = '../../../vondrick/mia/VIBE/' + 'output/IMG_1197_30.MOV/vibe_output.pkl'#filepath[1]
        self.pathtopkthree = '../../../vondrick/mia/VIBE/' + 'output/IMG_1203_30.MOV/vibe_output.pkl'#filepath[1]
        self.pathtopkeleven= '../../../vondrick/mia/VIBE/' + 'output/IMG_1921_30.MOV/vibe_output.pkl'#filepath[1]
        
        self.totalone = joblib.load(self.pathtopklone)
        self.totaltwo = joblib.load(self.pathtopkltwo)
        self.totalthree = joblib.load(self.pathtopkthree)
        self.totaleleven = joblib.load(self.pathtopkeleven)

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
        list_of_emg_values_rightglutt = []
        list_of_emg_values_leftquad = []
        list_of_emg_values_leftham = []
        list_of_emg_values_leftglutt = []
        list_of_emg_values_leftbicep = []
        list_of_emg_values_lefttricep = []
        list_of_2d_joints = []
        list_of_3d_joints = []
        list_of_frames = []
        list_of_bboxes = []
        list_of_predcam = []
        list_of_frame_paths = []
        filepath = self.all_files[index].split(",")
        for i in range(self.step):
            frame1=pathtoframes + "/" + filepath[2+i*17].zfill(6) + ".png"
            frame2=pathtoframes +  "/" + filepath[3+i*17].zfill(6) + ".png"
            frame3=pathtoframes +  "/" + filepath[4+i*17].zfill(6) + ".png"
            list_of_frame_paths.append(frame1)
            pickleframe1= int(filepath[5+i*17].split("/")[-1])
            pickleframe2=int(filepath[6+i*17].split("/")[-1])
            pickleframe3=int(filepath[7+i*17].split("/")[-1])
            emgvalues = filepath[8+i*17:17+i*17]
            emgvalues[-1] = emgvalues[-1].split("\n")[0]
            list_of_emg_values_rightquad.append(float(emgvalues[0]))
            list_of_emg_values_rightham.append(float(emgvalues[2]))
            list_of_emg_values_rightglutt.append(float(emgvalues[4]))

            list_of_emg_values_leftquad.append(float(emgvalues[1]))
            list_of_emg_values_leftham.append(float(emgvalues[3]))
            list_of_emg_values_leftglutt.append(float(emgvalues[5]))

            list_of_emg_values_leftbicep.append(float(emgvalues[6]))
            list_of_emg_values_lefttricep.append(float(emgvalues[7]))
            
            if '1197' in pathtoframes:
                total = self.totaltwo
            if '1196' in pathtoframes:
                total = self.totalone
            if '1203' in pathtoframes:
                total = self.totalthree
            if '1921' in pathtoframes:
                total = self.totaleleven
            
            firstjoints2dframe= total[1]['joints2d_img_coord'][pickleframe1]
            list_of_2d_joints.append(firstjoints2dframe)
            second2djoints2dframe = total[1]['joints2d_img_coord'][pickleframe2]
            third2djoints2dframe = total[1]['joints2d_img_coord'][pickleframe3]

            firstjoints3dframe= total[1]['joints3d'][pickleframe1]
            firstbboxes= total[1]['bboxes'][pickleframe1]
            firstpredcam = total[1]['pred_cam'][pickleframe1]
            list_of_3d_joints.append(firstjoints3dframe)
            list_of_bboxes.append(firstbboxes)
            list_of_predcam.append(firstpredcam)
            second2djoints3dframe = total[1]['joints3d'][pickleframe2]
            third2djoints3dframe = total[1]['joints3d'][pickleframe3]
            #if i==0:
                #cur2 = time.time()
                #print(cur2-cur,(cur2-cur)*30, "3")
                #cur = cur2
            """if self.phase != 'train':
                img=cv2.imread(frame1)
                img = img[...,::-1]
                list_of_frames.append(img)"""
            
            if self.plot:
                plt.figure()
                plt.imshow(img)
                plt.scatter(firstjoints2dframe[:,0],firstjoints2dframe[:,1],s=40)
                plt.savefig(current_path + "/" + str(index+i) + ".png")

        emg_values = [list_of_emg_values_rightquad,list_of_emg_values_rightham,
        list_of_emg_values_rightglutt,list_of_emg_values_leftquad,
        list_of_emg_values_leftham,list_of_emg_values_leftglutt,
        list_of_emg_values_leftbicep,list_of_emg_values_lefttricep]

        digitized_emg_values=[]

        cur2 = time.time()
        #print(cur2-cur, "4")
        cur = cur2
        for muscle in emg_values:
            digitized_emg_values.append(np.digitize(muscle,self.bins))
        
        cur2 = time.time()
        #print(cur2-cur, "5")
        cur = cur2

        list_of_emg_values_right_leg = [digitized_emg_values[0],digitized_emg_values[2],digitized_emg_values[4]]
        list_of_emg_values_left_leg = [digitized_emg_values[1],digitized_emg_values[3],digitized_emg_values[5]]
        list_of_emg_values_left_arm = [digitized_emg_values[6],digitized_emg_values[7]]

        if self.plot:
            self.animate(list_of_emg_values_right_leg, [self.muscles[0],self.muscles[2],self.muscles[4]], 'rightlegdig', '2',current_path)
            self.animate(list_of_emg_values_left_leg, [self.muscles[1],self.muscles[3],self.muscles[5]], 'leftlegdig', '2',current_path)
            self.animate(list_of_emg_values_left_arm, [self.muscles[6],self.muscles[7]], 'leftarmdig', '2',current_path)
        return (emg_values,list_of_2d_joints, list_of_3d_joints, list_of_frame_paths, list_of_bboxes, list_of_predcam)

    def __getitem__(self, index):
        
        cur = time.time()
        (list_of_emg_values, twod_joints, list_of_threed_joints, list_of_frame_paths, list_of_bboxes, list_of_predcam) = self.visualize_video(index)
        cur2 = time.time()
        #print(cur2-cur,"0")
        cur = cur2
        #list_of_frames=np.array(list_of_frames)
        twod_joints=np.array(twod_joints)
        bboxes = np.array(list_of_bboxes)
        predcam = np.array(list_of_predcam)
        threed_joints=np.array(list_of_threed_joints)
        #twod_joints = twod_joints.reshape(twod_joints.shape[0],-1)
        emg_values_right_quad = np.array(list_of_emg_values)[0]
        bined_right_quad = np.digitize(emg_values_right_quad,self.bins)
        emg_values_left_quad = np.array(list_of_emg_values)[0]
        bined_left_quad = np.digitize(emg_values_left_quad,self.bins)

        # Return results.
        cur2 = time.time()
        #print(cur2-cur,"1")
        cur = cur2
        result = {'bined_left_quad': bined_left_quad,  
                  'bined_right_quad': bined_right_quad,
                  'left_quad': emg_values_left_quad,
                  'right_quad':emg_values_right_quad,  
                  '2dskeleton': twod_joints,
                  '3dskeleton': threed_joints,
                  'bboxes': bboxes,
                  'predcam': predcam,
                  #'frames': list_of_frames,
                  'frame_paths': list_of_frame_paths,
                  'bins': np.linspace(0, self.maxemg, 20)}
        return result

