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

        divide = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.tensor([1080.0,1920.0]),dim=0),dim=0),dim=0).repeat(twodskeleton.shape[0],twodskeleton.shape[1],twodskeleton.shape[2],1).to(self.device)
        twodkpts = twodskeleton/divide
        #pdb.set_trace()
        #
        #if self.train_args.modelname != 'transf':
        twodkpts = twodkpts.reshape(twodskeleton.shape[0],twodkpts.shape[1],-1).to(self.device)
    
        bined_left_quad = data_retval['bined_left_quad']-1
        emggroundtruth = data_retval['emg_values'].to(self.device)
        cond = data_retval['cond'].to(self.device)
        emggroundtruth = emggroundtruth/100.0

        if self.train_args.modelname == 'transf':
            emg_output = self.my_model(twodkpts) #, cond
        else:
            emg_output = self.my_model(torch.unsqueeze(twodkpts.permute(0,2,1),dim=1))
        cur2 = time.time()
        #print(cur2-cur,"3")
        cur = cur2
        model_retval = dict()
        
        meangt = torch.unsqueeze(torch.mean(emggroundtruth,dim=2),dim=2).repeat(1,1,10)

        if self.train_args.modelname != 'transf':
            loss = self.mse(emg_output[:,:,0,:], (emggroundtruth[:,:,:]).type(torch.cuda.FloatTensor))
            #print(leftquad[:,:])
            model_retval['emg_output'] = emg_output[:,:,0,:]
        else:
            total_loss_r = self.mse(emg_output[:,4,:], (emggroundtruth[:,1,:]).type(torch.cuda.FloatTensor))
            total_loss_l = self.mse(emg_output[:,0,:], (emggroundtruth[:,0,:]).type(torch.cuda.FloatTensor))
            total_loss = total_loss_r + total_loss_l
            model_retval['emg_output'] = emg_output[:,:,:]

        model_retval['emg_gt'] = emggroundtruth
        
        loss_retval = dict()
        loss_retval['cross_ent'] = total_loss 
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
