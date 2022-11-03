'''
Objective functions.
'''

import torch
import pdb
class MyLosses():
    '''
    Wrapper around the loss functionality such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, phase):
        self.train_args = train_args
        self.logger = logger
        self.phase = phase
        self.l1_lw = train_args.l1_lw
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def my_l1_loss(self, rgb_output, rgb_target):
        '''
        :param rgb_output (B, H, W, 3) tensor.
        :param rgb_target (B, H, W, 3) tensor.
        :return loss_l1 (tensor).
        '''
        loss_l1 = self.l1_loss(rgb_output, rgb_target)
        return loss_l1

    def per_example(self, data_retval, model_retval):
        '''
        Loss calculations that *can* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :return loss_retval (dict): Preliminary loss information.
        '''
        (B, H, W, _) = data_retval['rgb_input'].shape

        loss_l1 = []

        # Loop over every example.
        for i in range(B):

            rgb_input = data_retval['rgb_input'][i:i + 1]
            rgb_output = model_retval['rgb_output'][i:i + 1]
            rgb_target = data_retval['rgb_target'][i:i + 1]

            # Calculate loss terms.
            cur_l1 = self.my_l1_loss(rgb_output, rgb_target)
            
            # Update lists.
            if self.l1_lw > 0.0:
                loss_l1.append(cur_l1)
            
        # Sum & return losses + other informative metrics across batch size within this GPU.
        # Prefer sum over mean to be consistent with dataset size.
        loss_l1 = torch.sum(torch.stack(loss_l1)) if len(loss_l1) else None

        # Return results.
        result = dict()
        result['l1'] = loss_l1
        return result

    def entire_batch(self, data_retval, model_retval, loss_retval,ignoremovie, total_step):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :return loss_retval (dict): All loss information.
        '''

        # Sum all terms across batch size.
        # Prefer sum over mean to be consistent with dataset size.

        list_of_movienames = []
    
        """v = loss_retval['cross_ent']
        print(v.shape)
        if torch.is_tensor(v):
            for i in range(v.shape[0]):
                print(len(data_retval['frame_paths']),data_retval['frame_paths'][0].shape)
                moviename = data_retval['frame_paths'][0][i].split("/")[-2].split("_")[1]
                list_of_movienames.append(moviename)
                if moviename in loss_retval.keys():
                    loss_retval[moviename].append(v[i].detach().cpu().numpy().item())
                else:
                    loss_retval[moviename] = [v[i].detach().cpu().numpy().item()]"""

        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                if k == 'cross_ent':
                    loss_retval[k] = torch.mean(v)

        # Obtain total loss. 
        loss_total = loss_retval['cross_ent'] #* self.l1_lw
        #video = data_retval['frame_paths'][0][0].split("/")[-2]
        
        # Convert loss terms (just not the total) to floats for logging.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = v.item()

        # Report all loss values.
        if self.phase != 'eval':
            """for moviename in sorted(set(list_of_movienames)):
                if ignoremovie in moviename:
                    self.logger.report_scalar(
                    self.phase + '/loss_total_' + moviename, loss_retval[moviename], step=total_step)"""
            self.logger.report_scalar(
                self.phase + '/loss_total', loss_total.item(), step=total_step)
            for i in range(model_retval['emg_gt'].shape[1]):
                self.logger.report_scalar(
                    self.phase + '/emggt' + str(i), torch.mean(model_retval['emg_gt'][:,i,:]).item(), step=total_step,remember=False,commit_histogram=True)
                self.logger.report_scalar(
                    self.phase + '/emgpred' + str(i), torch.mean(model_retval['emg_output'][:,i,:]).item(), step=total_step,remember=False,commit_histogram=True)

        #self.logger.report_scalar(
        #    self.phase + '/loss_l1', loss_retval['l1'], remember=True)

        # Return results, i.e. append to the existing loss_retval dictionary.
        loss_retval['total'] = loss_total
        return loss_retval
