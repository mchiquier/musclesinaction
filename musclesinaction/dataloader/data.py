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

    def visualize_video(self,index):
        #index = 5100
        current_path = self.log_dir + "/" + str(index)
        cur = time.time()
        os.makedirs(current_path, 0o777, exist_ok=True)
        file_idx = index
        filepath = self.all_files[index]
        emgvalues = np.load(filepath + "/emgvaluesstd.npy")
        twodjoints = np.load(filepath + "/2djoints.npy")
        threedjoints = np.load(filepath + "/3djoints.npy")
 
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
            list_of_emg_values_rightbicep.append(float(emgvalues[4]))

            list_of_emg_values_leftquad.append(float(emgvalues[1]))
            list_of_emg_values_leftham.append(float(emgvalues[3]))
            list_of_emg_values_leftbicep.append(float(emgvalues[5]))

            list_of_emg_values_righttricep.append(float(emgvalues[6]))
            list_of_emg_values_lefttricep.append(float(emgvalues[7]))

            total = self.pickledict[pathtoframes.split("/")[-3] + "/" +pathtoframes.split("/")[-2]]
            
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
            
            
            firstjoints2dframe= total[1]['joints2d_img_coord'][pickleframe1]
            list_of_2d_joints.append(firstjoints2dframe)
            second2djoints2dframe = total[1]['joints2d_img_coord'][pickleframe2]
            third2djoints2dframe = total[1]['joints2d_img_coord'][pickleframe3]

            origcam = total[1]['orig_cam'][pickleframe1]
            verts = total[1]['verts'][pickleframe1]
            list_of_orig_cam.append(origcam)
            list_of_verts.append(verts)
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
        list_of_emg_values_rightbicep,list_of_emg_values_righttricep,list_of_emg_values_leftquad,
        list_of_emg_values_leftham,list_of_emg_values_leftbicep,list_of_emg_values_lefttricep]

        #emg_values = [list_of_emg_values_rightquad,list_of_emg_values_rightham,
        #list_of_emg_values_leftquad,list_of_emg_values_leftham]

        digitized_emg_values=[]

        cur2 = time.time()
        #print(cur2-cur, "4")
        cur = cur2
        for muscle in emg_values:
            digitized_emg_values.append(np.digitize(muscle,self.bins))
        
        cur2 = time.time()
        #print(cur2-cur, "5")
        cur = cur2

        #emg_values.pop(5)
        #emg_values.pop(1)
        return (emg_values,list_of_2d_joints, list_of_3d_joints, list_of_frame_paths, list_of_bboxes, list_of_predcam, list_of_orig_cam, list_of_verts)

    def __getitem__(self, index):
        
        filepath = self.all_files[index].split("\n")[0]
        if self.std == "True":
            emg_values = np.load("../../../vondrick/mia/VIBE/" + filepath + "/emgvaluesstd.npy")
        else:
            emg_values = np.load("../../../vondrick/mia/VIBE/" + filepath + "/emgvalues.npy")
        twod_joints = np.load("../../../vondrick/mia/VIBE/" + filepath + "/joints2d.npy")
        threed_joints = np.load("../../../vondrick/mia/VIBE/" + filepath + "/joints3d.npy")
        predcam = np.load("../../../vondrick/mia/VIBE/" + filepath + "/predcam.npy")
        bboxes = np.load("../../../vondrick/mia/VIBE/" + filepath + "/bboxes.npy")

        file = open("../../../vondrick/mia/VIBE/" + filepath + "/frame_paths.txt", 'r')
        line = file.readlines()[0]
        frame_paths = line.rstrip().split(',')
            
        #print(cur2-cur,"0")



        #list_of_frames=np.array(list_of_frames)
        person = filepath.split("/")[2]
        if self.cond=='True':
            if person == 'David':
                #condval = torch.tensor([0]).to(torch.int64) #.to(self.device)
                condval = np.array([1.0])
                condvalbad = np.array([0.1])
                #mean = np.array([20.477754532775453,22.77594142259414,8.943654114365412,10.712970711297071,25.561506276150627,31.569735006973502,10.12510460251046,8.890306834030683,])
                #std = np.array([9.843984859529295,7.8700795426556445,2.184267197160608,4.016428843990136,13.291629562251334,10.801735697194195,2.4458652432669696,3.6610053859341303])
            #condvalbad = torch.tensor([1]).to(torch.int64)
            elif person == 'Ishaan':
                #condval = torch.tensor([1]).to(torch.int64)
                #condvalbad = torch.tensor([2]).to(torch.int64)
                condval = np.array([0.9]) #np.array([0.9]) #np.array([0.6])
                condvalbad = np.array([0.3])
                #mean = np.array([12.693347193347194,9.420859320859321,15.787179487179488,24.37186417186417,7.33083853083853,8.572765072765073,24.766389466389466,28.63139293139293,])
                #std = np.array([5.0676918939207605,4.278709410152058,8.025951618999134,16.31777423074967,4.699630460205841,4.770252420447796,15.780496877387819,19.274093253782787])
            elif person == "Jo":
                #condval = torch.tensor([2]).to(torch.int64)
                #condvalbad = torch.tensor([3]).to(torch.int64)
                condval = np.array([0.8]) #np.array([0.9]) #np.array([0.3])
                condvalbad = np.array([0.6])
                #mean= np.array([5.376230661040788,8.166526019690577,27.226019690576653,31.612869198312236,14.103867791842475,13.167158931082982,22.12011251758087,20.867088607594937])
                #std = np.array([3.651494944259142,5.772440868944948,16.608860878596072,22.613865413206977,4.348816765789181,6.718509726195761,10.779130379832406,13.83257533754815])
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
                #condval = torch.tensor([3]).to(torch.int64)
                #condvalbad = torch.tensor([0]).to(torch.int64)
                condval = np.array([0.1]) #np.array([0.9]) #np.array([0.1])
                condvalbad = np.array([0.9]) #np.array([0.9])
                #mean = np.array([18.297543859649124,23.576771929824563,9.026315789473685,10.502315789473684,22.304070175438596,27.723438596491228,16.91621052631579,8.84280701754386])
                #std = np.array([14.925368522965195,18.504676528899612,4.521381030901245,3.331491669265,20.79705016857905,21.666831424218724,9.523643543394451,2.714327000937558])
        else:
            if person == 'Samir':
                #condval = torch.tensor([0]).to(torch.int64) #.to(self.device)
                condval = np.array([0.1])
                condvalbad = np.array([0.9])
                mean = np.array([20.477754532775453,22.77594142259414,8.943654114365412,10.712970711297071,25.561506276150627,31.569735006973502,10.12510460251046,8.890306834030683,])
                std = np.array([9.843984859529295,7.8700795426556445,2.184267197160608,4.016428843990136,13.291629562251334,10.801735697194195,2.4458652432669696,3.6610053859341303])
            
            #condvalbad = torch.tensor([1]).to(torch.int64)
            elif person == 'Sonia':
                #condval = torch.tensor([1]).to(torch.int64)
                #condvalbad = torch.tensor([2]).to(torch.int64)
                condval = np.array([0.1]) #np.array([0.9]) #np.array([0.6])
                condvalbad = np.array([0.9])
                mean = np.array([12.693347193347194,9.420859320859321,15.787179487179488,24.37186417186417,7.33083853083853,8.572765072765073,24.766389466389466,28.63139293139293,])
                std = np.array([5.0676918939207605,4.278709410152058,8.025951618999134,16.31777423074967,4.699630460205841,4.770252420447796,15.780496877387819,19.274093253782787])
            elif person == "Sruthi":
                #condval = torch.tensor([2]).to(torch.int64)
                #condvalbad = torch.tensor([3]).to(torch.int64)
                condval = np.array([0.1]) #np.array([0.9]) #np.array([0.3])
                condvalbad = np.array([0.9])
                mean= np.array([5.376230661040788,8.166526019690577,27.226019690576653,31.612869198312236,14.103867791842475,13.167158931082982,22.12011251758087,20.867088607594937])
                std = np.array([3.651494944259142,5.772440868944948,16.608860878596072,22.613865413206977,4.348816765789181,6.718509726195761,10.779130379832406,13.83257533754815])

            else:
                #condval = torch.tensor([3]).to(torch.int64)
                #condvalbad = torch.tensor([0]).to(torch.int64)
                condval = np.array([0.1]) #np.array([0.9]) #np.array([0.1])
                condvalbad = np.array([0.9]) #np.array([0.9])
                mean = np.array([18.297543859649124,23.576771929824563,9.026315789473685,10.502315789473684,22.304070175438596,27.723438596491228,16.91621052631579,8.84280701754386])
                std = np.array([14.925368522965195,18.504676528899612,4.521381030901245,3.331491669265,20.79705016857905,21.666831424218724,9.523643543394451,2.714327000937558])
        
  
        result = {'condval':condval,
                    'condvalbad':condvalbad,
                    #'mean': mean,
                    #'std':std,
                  'emg_values': emg_values.transpose(1,0),
                  'predcam': predcam,
                  'bboxes': bboxes,
                  '2dskeleton': twod_joints[:,:25,:],
                  '3dskeleton': threed_joints[:,:25,:],
                  'frame_paths': frame_paths,
                  'indexpath': filepath.split("/")[-1]}
                  #'frames': list_of_frames,
                  #'frame_paths': list_of_frame_paths[0],
                  #'bins': np.linspace(0, self.maxemg, 20
        return result

