'''
Objective functions.
'''

import torch

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

    def entire_batch(self, data_retval, model_retval, loss_retval):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :return loss_retval (dict): All loss information.
        '''

        # Sum all terms across batch size.
        # Prefer sum over mean to be consistent with dataset size.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = torch.sum(v)

        # Obtain total loss. 
        loss_total = loss_retval['l1'] * self.l1_lw
        
        # Convert loss terms (just not the total) to floats for logging.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = v.item()

        # Report all loss values.
        self.logger.report_scalar(
            self.phase + '/loss_total', loss_total.item(), remember=True)
        self.logger.report_scalar(
            self.phase + '/loss_l1', loss_retval['l1'], remember=True)

        # Return results, i.e. append to the existing loss_retval dictionary.
        loss_retval['total'] = loss_total
        return loss_retval
