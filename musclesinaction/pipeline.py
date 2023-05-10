'''
Entire training pipeline logic.
'''

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import time
# Internal imports.
import musclesinaction.losses.loss as loss
import musclesinaction.utils.utils as utils


class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleList(networks)
        self.my_model = networks[0]
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().
        self.crossent = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        

    def set_phase(self, phase):
        self.phase = phase
        self.losses = loss.MyLosses(self.train_args, self.logger, phase)

        if phase == 'train':
            self.train()
            for net in self.networks:
                if net is not None:
                    net.train()
            torch.set_grad_enabled(True)

        else:
            self.eval()
            for net in self.networks:
                if net is not None:
                    net.eval()
            torch.set_grad_enabled(False)

    def perspective_projection(self, points, rotation, translation,
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
            self, pare_cam, bbox_height, bbox_center,
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

    def forward(self, data_retval, cur_step, total_step):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param data_retval (dict): Data loader elements.
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :return (model_retval, loss_retval)
            model_retval (dict): All output information.
            loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        '''
        cur = time.time()
        twodskeleton = data_retval['2dskeleton']
        #twodskeleton = twodskeleton.reshape(twodskeleton.shape[0],twodskeleton.shape[1],-1)

        #mean = data_retval['mean']
        #std = data_retval['std']
        #mean = torch.unsqueeze(mean,dim=2)
        #std = torch.unsqueeze(std,dim=2)
        threedskeleton = data_retval['3dskeleton']
        #pose = data_retval['pose']


        if self.train_args.threed == "True":
            skeleton = threedskeleton
        else:
            skeleton = twodskeleton
        """bboxes = data_retval['bboxes']
        predcam = data_retval['predcam']
        proj = 5000.0
        
        height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1])
        center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1)
        focal=torch.tensor([[proj]]).to(self.device).repeat(height.shape[0],1)
        predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1)
        translation = self.convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
        reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3])
        rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(reshapethreed.shape[0],1,1).to(self.device)
        focal=torch.tensor([[proj]]).to(self.device).repeat(translation.shape[0],1)
        imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(reshapethreed.shape[0],1).to(self.device)
        twodkpts, skeleton = self.perspective_projection(reshapethreed, rotation, translation.float(),focal[:,0], imgdimgs)
        #skeleton = skeleton.reshape(twodskeleton.shape[0],30,skeleton.shape[1],skeleton.shape[2])
        #skeleton = skeleton.reshape(skeleton.shape[0],skeleton.shape[1],-1)
        twodkpts = twodkpts.reshape(twodskeleton.shape[0],(self.train_args.step),twodkpts.shape[1],twodkpts.shape[2])"""
        if self.train_args.threed != "True":
            divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(skeleton.shape[0],skeleton.shape[1],skeleton.shape[2],1).to(self.device)
            skeleton= skeleton/divide
        #divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodskeleton.shape[0],twodskeleton.shape[1],twodskeleton.shape[2],1).to(self.device)
        #twodskeleton= twodskeleton/divide
        #pdb.set_trace()
        #
        #if self.train_args.modelname != 'transf':
        skeleton = skeleton.reshape(skeleton.shape[0],skeleton.shape[1],-1)
    
        emggroundtruth = data_retval['emg_values'].to(self.device)
        emggroundtruth = emggroundtruth
        cond = data_retval['condval'].to(self.device)
        badcond = data_retval['condvalbad'].to(self.device)
        #bined_left_quad = bined_left_quad.to(self.device)-1
        skeleton = skeleton.to(self.device)
        cur2 = time.time()
        #print(cur2-cur,"2")
        cur = cur2
        if self.train_args.modelname == 'transf':
            if self.train_args.predemg == 'True':
                emg_output = self.my_model(skeleton,cond) 
                emg_output_wrong_condition = self.my_model(skeleton, badcond)
            else:
                pose_pred = self.my_model(emggroundtruth[:,:,:],cond)
                pose_pred_wrong_condition = self.my_model(emggroundtruth[:,:,:],badcond)
        else:
            emg_output = self.my_model(torch.unsqueeze(skeleton.permute(0,2,1),dim=1),cond)
        cur2 = time.time()
        #print(cur2-cur,"3")
        cur = cur2
        model_retval = dict()
        
  
        if self.train_args.modelname != 'transf':
            loss = self.mse(emg_output[:,:,0,:], (emggroundtruth[:,:,:]).type(torch.cuda.FloatTensor))
            #print(leftquad[:,:])
            model_retval['emg_output'] = emg_output[:,:,0,:]
        else:
            if self.train_args.predemg == 'True':
                
                if self.phase == "eval" or self.phase == "evalood":
                    """if self.train_args.std=="True":
                        emg_output_std = (emg_output*std) + mean
                        emg_output_wrong_condition_std = (emg_output_wrong_condition*std) + mean
                        model_retval['emg_output'] = emg_output_std[:,:,:]  
                        model_retval['emg_output_wrong_condition'] = emg_output_wrong_condition_std[:,:,:] 
                        total_loss = self.mse(emg_output_std, (emggroundtruth).type(torch.cuda.FloatTensor))
                        bad_cond_loss = self.mse(emg_output_wrong_condition_std, (emggroundtruth).type(torch.cuda.FloatTensor))
                    else:"""
                    #pdb.set_trace()
                    model_retval['emg_output'] = emg_output[:,:,:]  
                    model_retval['emg_output_wrong_condition'] = emg_output_wrong_condition[:,:,:]   
                    bad_cond_loss = self.mse(emg_output_wrong_condition, (emggroundtruth).type(torch.cuda.FloatTensor))
                    total_loss = self.mse(emg_output, (emggroundtruth).type(torch.cuda.FloatTensor))
                else:
                    model_retval['emg_output'] = emg_output[:,:,:]  
                    model_retval['emg_output_wrong_condition'] = emg_output_wrong_condition[:,:,:]   
                    bad_cond_loss = self.mse(emg_output_wrong_condition, (emggroundtruth).type(torch.cuda.FloatTensor))
                    total_loss = self.mse(emg_output, (emggroundtruth).type(torch.cuda.FloatTensor))
            else:
                #pdb.set_trace()
                total_loss = self.mse(pose_pred, (skeleton[:,:,:]).type(torch.cuda.FloatTensor))
                bad_cond_loss = self.mse(pose_pred_wrong_condition, (skeleton[:,:,:]).type(torch.cuda.FloatTensor))
                model_retval['pose_output'] = pose_pred
                model_retval['pose_output_wrong_condition'] = pose_pred_wrong_condition
        

        model_retval['emg_gt'] = emggroundtruth
        
        loss_retval = dict()
        loss_retval['cross_ent'] = total_loss 
        loss_retval['bad_loss'] = bad_cond_loss
        cur2 = time.time()
        #print(cur2-cur,"4")
        cur = cur2
        #loss_retval = self.losses.per_example(data_retval, model_retval)

        return (model_retval, loss_retval)

    def process_entire_batch(self, data_retval, model_retval, loss_retval, ignoremovie, cur_step, total_step):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(data_retval, model_retval, loss_retval,ignoremovie, total_step)

        return loss_retval
