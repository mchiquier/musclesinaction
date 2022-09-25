'''
Data augmentation logic.
'''

import torch
# Library imports.
from torchvision import transforms


# NOTE: This template is tailored to ImageNet as one of many possible starting points.
# It is however not exactly the same as the "official" ImageNet bag of tricks, so use with caution!
# Adapted from:
# https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/fastai_imagenet.py

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    '''
    Lighting noise (AlexNet - style PCA - based noise).
    '''

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def get_train_transform(size):
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4, .4, .4),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        # normalize,
    ])
    return my_transform


def get_test_transform(size):
    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        # normalize,
    ])
    return my_transform
