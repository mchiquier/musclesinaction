'''
Neural network architecture description.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np

# Internal imports.
import utils


class MyModel(torch.nn.Module):
    '''
    NOTE: This is just an example architecture capable of autoencoding images.
    https://github.com/lucidrains/perceiver-pytorch
    '''

    def __init__(self, logger):
        '''
        :param image_dim (int): Size of entire input or output image.
        :param patch_dim (int): Size of one image patch.
        :param emb_dim (int): Internal feature embedding size.
        :param depth (int): Perceiver IO depth.
        '''
        super().__init__()
        self.logger = logger

        self.net = torch.nn.Conv2d(3, 3, 1)

    def forward(self, rgb_input):
        '''
        :param rgb_input (B, 3, Hi, Wi) tensor.
        :return rgb_output (B, 3, Hi, Wi) tensor.
        '''

        rgb_output = self.net(rgb_input)

        return rgb_output
