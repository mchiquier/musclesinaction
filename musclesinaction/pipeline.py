'''
Entire training pipeline logic.
'''

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

# Internal imports.
import template.losses.loss as loss
import template.utils.utils as utils


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

        rgb_input = data_retval['rgb_input']
        rgb_input = rgb_input.to(self.device)

        rgb_output = self.my_model(rgb_input)

        model_retval = dict()
        model_retval['rgb_output'] = rgb_output

        loss_retval = self.losses.per_example(data_retval, model_retval)

        return (model_retval, loss_retval)

    def process_entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(data_retval, model_retval, loss_retval)

        return loss_retval
