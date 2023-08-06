'''
Logging and visualization logic.
'''

import musclesinaction.vis.logvisgen as logvisgen
from musclesinaction.vis.renderer import Renderer

import pdb
import csv
import wandb
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import torch
import cv2
import subprocess
import time
import multiprocessing as mp
from smplx import SMPL as _SMPL
from smplx.utils import ModelOutput, SMPLOutput
from smplx.lbs import vertices2joints

# This script is the extended version of https://github.com/nkolot/SPIN/blob/master/smplify/losses.py to deal with
# sequences inputs.

import torch

JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """
    # pose_prior_weight = 1.
    # shape_prior_weight = 1.
    # angle_prior_weight = 1.
    # sigma = 10.

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss
    print(f'joints: {reprojection_loss[0].sum().item():.2f}, '
          f'pose_prior: {pose_prior_loss[0].item():.2f}, '
          f'angle_prior: {angle_prior_loss[0].item():.2f}, '
          f'shape_prior: {shape_prior_loss[0].item():.2f}')

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


def temporal_body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                               joints_2d, joints_conf, pose_prior,
                               focal_length=5000, sigma=100, pose_prior_weight=4.78,
                               shape_prior_weight=5, angle_prior_weight=15.2,
                               smooth_2d_weight=0.01, smooth_3d_weight=1.0,
                               output='sum'):
    """
    Loss function for body fitting
    """
    # pose_prior_weight = 1.
    # shape_prior_weight = 1.
    # angle_prior_weight = 1.
    # sigma = 10.

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    # Smooth 2d joint loss
    joint_conf_diff = joints_conf[1:]
    joints_2d_diff = projected_joints[1:] - projected_joints[:-1]
    smooth_j2d_loss = (joint_conf_diff ** 2) * joints_2d_diff.abs().sum(dim=-1)
    smooth_j2d_loss = torch.cat(
        [torch.zeros(1, smooth_j2d_loss.shape[1], device=body_pose.device), smooth_j2d_loss]
    ).sum(dim=-1)
    smooth_j2d_loss = (smooth_2d_weight ** 2) * smooth_j2d_loss

    # Smooth 3d joint loss
    joints_3d_diff = model_joints[1:] - model_joints[:-1]
    # joints_3d_diff = joints_3d_diff * 100.
    smooth_j3d_loss = (joint_conf_diff ** 2) * joints_3d_diff.abs().sum(dim=-1)
    smooth_j3d_loss = torch.cat(
        [torch.zeros(1, smooth_j3d_loss.shape[1], device=body_pose.device), smooth_j3d_loss]
    ).sum(dim=-1)
    smooth_j3d_loss = (smooth_3d_weight ** 2) * smooth_j3d_loss

    total_loss += smooth_j2d_loss + smooth_j3d_loss

    # print(f'joints: {reprojection_loss[0].sum().item():.2f}, '
    #       f'pose_prior: {pose_prior_loss[0].item():.2f}, '
    #       f'angle_prior: {angle_prior_loss[0].item():.2f}, '
    #       f'shape_prior: {shape_prior_loss[0].item():.2f}, '
    #       f'smooth_j2d: {smooth_j2d_loss.sum().item()}, '
    #       f'smooth_j3d: {smooth_j3d_loss.sum().item()}')

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def temporal_camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                                 focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [JOINT_IDS[joint] for joint in op_joints]
    # gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    # gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    # reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
    #                          projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op).sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


VIBE_DATA_DIR = 'musclesinaction/vibe_data'

JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(VIBE_DATA_DIR, 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = osp.join(VIBE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = VIBE_DATA_DIR
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output


def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces


# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys
import os

import time
import pickle

import numpy as np

import torch
import torch.nn as nn

DEFAULT_DTYPE = torch.float32

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def create_prior(prior_type, **kwargs):
    if prior_type == 'gmm':
        prior = MaxMixturePrior(**kwargs)
    elif prior_type == 'l2':
        return L2Prior(**kwargs)
    elif prior_type == 'angle':
        return SMPLifyAnglePrior(**kwargs)
    elif prior_type == 'none' or prior_type is None:
        # Don't use any pose prior
        def no_prior(*args, **kwargs):
            return 0.0
        prior = no_prior
    else:
        raise ValueError('Prior {}'.format(prior_type) + ' is not implemented')
    return prior


class SMPLifyAnglePrior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):
    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


def arrange_betas(pose, betas):
    batch_size = pose.shape[0]
    num_video = betas.shape[0]

    video_size = batch_size // num_video
    betas_ext = torch.zeros(batch_size, betas.shape[-1], device=betas.device)
    for i in range(num_video):
        betas_ext[i*video_size:(i+1)*video_size] = betas[i]

    return betas_ext

class TemporalSMPLify():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 use_lbfgs=True,
                 device=torch.device('cuda'),
                 max_iter=20):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size
        self.max_iter = max_iter
        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=VIBE_DATA_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.use_lbfgs = use_lbfgs
        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.max_iter,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    camera_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints


                    loss = temporal_camera_fitting_loss(model_joints, camera_translation,
                                               init_cam_t, camera_center,
                                               joints_2d, joints_conf, focal_length=self.focal_length)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_camera_fitting_loss(model_joints, camera_translation,
                                           init_cam_t, camera_center,
                                           joints_2d, joints_conf, focal_length=self.focal_length)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints

                    loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()
                # scheduler.step(epoch=i)

        # Get final loss value

        with torch.no_grad():
            betas_ext = arrange_betas(body_pose, betas)
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas_ext)
            model_joints = smpl_output.joints
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation,
                                                           camera_center,
                                                           joints_2d, joints_conf, self.pose_prior,
                                                           focal_length=self.focal_length,
                                                           output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        # Back to weak perspective camera
        camera_translation = torch.stack([
            2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
            camera_translation[:,0], camera_translation[:,1]
        ], dim=-1)

        betas = betas.repeat(pose.shape[0],1)
        output = {
            'theta': torch.cat([camera_translation, pose, betas], dim=1),
            'verts': vertices,
            'kp_3d': joints,
        }

        return output, reprojection_loss
        # return vertices, joints, pose, betas, camera_translation, reprojection_loss

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss



def smplify_runner(
        pred_rotmat,
        pred_betas,
        pred_cam,
        j2d,
        device,
        batch_size,
        lr=1.0,
        opt_steps=1,
        use_lbfgs=True,
        pose2aa=False
):
    smplify = TemporalSMPLify(
        step_size=lr,
        batch_size=batch_size,
        num_iters=opt_steps,
        focal_length=5000.,
        use_lbfgs=use_lbfgs,
        device=device,
        # max_iter=10,
    )
    # Convert predicted rotation matrices to axis-angle
    if pose2aa:
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach()).reshape(batch_size, -1)
    else:
        pred_pose = pred_rotmat

    # Calculate camera parameters for smplify
    pred_cam_t = pred_cam
    #pred_cam_t = torch.stack([pred_cam[:, 1], pred_cam[:, 2],2 * 5000 / (224 * pred_cam[:, 0] + 1e-9)], dim=-1)

    gt_keypoints_2d_orig = j2d
    # Before running compute reprojection error of the network
    
    opt_joint_loss = smplify.get_fitting_loss(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(30,1).to(device),
        gt_keypoints_2d_orig).mean(dim=-1)

    best_prediction_id = torch.argmin(opt_joint_loss).item()
    pred_betas = pred_betas[best_prediction_id].unsqueeze(0)
    # pred_betas = pred_betas[best_prediction_id:best_prediction_id+2] # .unsqueeze(0)
    # top5_best_idxs = torch.topk(opt_joint_loss, 5, largest=False)[1]
    # breakpoint()

    start = time.time()
    # Run SMPLify optimization initialized from the network prediction
    # new_opt_vertices, new_opt_joints, \
    # new_opt_pose, new_opt_betas, \
    # new_opt_cam_t, \
    output, new_opt_joint_loss = smplify(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(30,1).to(device),
        gt_keypoints_2d_orig,
    )
    new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
    # smplify_time = time.time() - start
    # print(f'Smplify time: {smplify_time}')
    # Will update the dictionary for the examples where the new loss is less than the current one
    update = (new_opt_joint_loss < opt_joint_loss)

    new_opt_vertices = output['verts']
    new_opt_cam_t = output['theta'][:,:3]
    new_opt_pose = output['theta'][:,3:75]
    new_opt_betas = output['theta'][:,75:]
    new_opt_joints3d = output['kp_3d']

    return_val = [
        update, new_opt_vertices.cpu(), new_opt_cam_t.cpu(),
        new_opt_pose.cpu(), new_opt_betas.cpu(), new_opt_joints3d.cpu(),
        new_opt_joint_loss, opt_joint_loss,
    ]

    return return_val

class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, argstwo, context):
        
        self.step_interval = 1
        self.num_exemplars = 4  # To increase simultaneous examples in wandb during train / val.
        self.maxemg = 300
        self.classif = args.classif
        self.args = args
        self.argstwo = argstwo
        self.renderer = Renderer(resolution=(1080, 1920), orig_img=True, wireframe=False)
        super().__init__(args.log_path, context, args.name)

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
        points = torch.einsum('bij,bkj->bki', rotation.repeat(points.shape[0],1,1), points)
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

        cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        return cam_t

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval):

        if cur_step % self.step_interval == 0:
            
            j=0

            exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars

            total_loss = loss_retval['total']
            framelist = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]

            #pdb.set_trace()
            current_path = self.visualize_video(framelist,data_retval['2dskeleton'][j],cur_step,j,phase,framelist[0].split("/")[-2])
            threedskeleton = data_retval['3dskeleton'][j]
            current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,framelist[0].split("/")[-2])

            #pdb.set_trace()
            
            if self.classif:
                values = data_retval['bined_left_quad'][j]-1
                bins = data_retval['bins'][j]
                gt_values = torch.index_select(bins.cpu(), 0, values.cpu())
                pred_values = model_retval['emg_bins'][j].cpu()
                
                self.animate([gt_values.numpy()],[pred_values.detach().numpy()],['left_quad'],'leftleg',2,current_path,epoch)
            else:
                gt_values = data_retval['left_quad'][j]
                gt_values[gt_values>1.0] = 1.0
                pred_values = model_retval['emg_output'][j][0].cpu()
                pred_values[pred_values>1.0] = 1.0
                self.animate([gt_values.numpy()],[pred_values.detach().numpy()],['left_quad'],'leftleg',2,current_path,epoch)

            # Print metrics in console.

            command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_159_leftleg_emg.mp4','-filter_complex',
            'hstack=inputs=3', f'{current_path}/total.mp4']
            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                    f'total_loss: {total_loss:.3f}  ')

    def plot_skel(self, keypoints, predkeypoints, img, currentpath, i, markersize=5, linewidth=2, alpha=0.7):
        plt.figure()
        #pdb.set_trace()
        keypoints = keypoints[:25]
        predkeypoints = predkeypoints[:25]
        plt.imshow(img)
        limb_seq = [([17, 15], [238, 0, 255]),
                    ([15, 0], [255, 0, 166]),
                    ([0, 16], [144, 0, 255]),
                    ([16, 18], [65, 0, 255]),
                    ([0, 1], [255, 0, 59]),
                    ([1, 2], [255, 77, 0]),
                    ([2, 3], [247, 155, 0]),
                    ([3, 4], [255, 255, 0]),
                    ([1, 5], [158, 245, 0]),
                    ([5, 6], [93, 255, 0]),
                    ([6, 7], [0, 255, 0]),
                    ([1, 8], [255, 21, 0]),
                    ([8, 9], [6, 255, 0]),
                    ([9, 10], [0, 255, 117]),
                    # ([10, 11]], [0, 252, 255]),  # See comment above
                    ([10, 24], [0, 252, 255]),
                    ([8, 12], [0, 140, 255]),
                    ([12, 13], [0, 68, 255]),
                    ([13, 14], [0, 14, 255]),
                    # ([11, 22], [0, 252, 255]),
                    # ([11, 24], [0, 252, 255]),
                    ([24, 22], [0, 252, 255]),
                    ([24, 24], [0, 252, 255]),
                    ([22, 23], [0, 252, 255]),
                    ([14, 19], [0, 14, 255]),
                    ([14, 21], [0, 14, 255]),
                    ([19, 20], [0, 14, 255])]

        colors_vertices = {0: limb_seq[4][1],
                        1: limb_seq[11][1],
                        2: limb_seq[5][1],
                        3: limb_seq[6][1],
                        4: limb_seq[7][1],
                        5: limb_seq[8][1],
                        6: limb_seq[9][1],
                        7: limb_seq[10][1],
                        8: limb_seq[11][1],
                        9: limb_seq[12][1],
                        10: limb_seq[13][1],
                        11: limb_seq[14][1],
                        12: limb_seq[15][1],
                        13: limb_seq[16][1],
                        14: limb_seq[17][1],
                        15: limb_seq[1][1],
                        16: limb_seq[2][1],
                        17: limb_seq[0][1],
                        18: limb_seq[3][1],
                        19: limb_seq[21][1],
                        20: limb_seq[23][1],
                        21: limb_seq[22][1],
                        22: limb_seq[18][1],
                        23: limb_seq[20][1],
                        24: limb_seq[19][1]}
        """limb_seq = [([17, 15], [238, 0, 255]),
                    ([15, 0], [255, 0, 166]),
                    ([0, 16], [144, 0, 255]),
                    ([16, 18], [65, 0, 255]),
                    ([0, 1], [255, 0, 59]),
                    ([1, 2], [255, 77, 0]),
                    ([2, 3], [247, 155, 0]),
                    ([3, 4], [255, 255, 0]),
                    ([1, 5], [158, 245, 0]),
                    ([5, 6], [93, 255, 0]),
                    ([6, 7], [0, 255, 0]),
                    ([1, 8], [255, 21, 0]),
                    ([8, 9], [6, 255, 0]),
                    ([9, 10], [0, 255, 117]),
                    # ([10, 11]], [0, 252, 255]),  # See comment above
                    ([10, 24], [0, 252, 255]),
                    ([8, 12], [0, 140, 255]),
                    ([12, 13], [0, 68, 255]),
                    ([13, 14], [0, 14, 255]),
                    # ([11, 22], [0, 252, 255]),
                    # ([11, 24], [0, 252, 255]),
                    ([24, 22], [0, 252, 255]),
                    ([24, 24], [0, 252, 255]),
                    ([22, 23], [0, 252, 255]),
                    ([14, 19], [0, 14, 255]),
                    ([14, 21], [0, 14, 255]),
                    ([19, 20], [0, 14, 255])]

        colors_vertices = {0: limb_seq[4][1],
                        1: limb_seq[11][1],
                        2: limb_seq[5][1],
                        3: limb_seq[6][1],
                        4: limb_seq[7][1],
                        5: limb_seq[8][1],
                        6: limb_seq[9][1],
                        7: limb_seq[10][1],
                        8: limb_seq[11][1],
                        9: limb_seq[12][1],
                        10: limb_seq[13][1],
                        11: limb_seq[14][1],
                        12: limb_seq[15][1],
                        13: limb_seq[16][1],
                        14: limb_seq[17][1],
                        15: limb_seq[1][1],
                        16: limb_seq[2][1],
                        17: limb_seq[0][1],
                        18: limb_seq[3][1],
                        19: limb_seq[21][1],
                        20: limb_seq[23][1],
                        21: limb_seq[22][1],
                        22: limb_seq[18][1],
                        23: limb_seq[20][1],
                        24: limb_seq[19][1]}"""

        alpha = alpha
        for vertices, color in limb_seq:
            if predkeypoints[vertices[0]].mean() != 0 and predkeypoints[vertices[1]].mean() != 0:
                plt.plot([predkeypoints[vertices[0]][0], predkeypoints[vertices[1]][0]],
                        [predkeypoints[vertices[0]][1], predkeypoints[vertices[1]][1]], linewidth=linewidth,
                        color=[j / 255 for j in colors_vertices[0]], alpha=alpha)
                
        for vertices, color in limb_seq:
            if keypoints[vertices[0]].mean() != 0 and keypoints[vertices[1]].mean() != 0:
                plt.plot([keypoints[vertices[0]][0], keypoints[vertices[1]][0]],
                        [keypoints[vertices[0]][1], keypoints[vertices[1]][1]], linewidth=linewidth,
                        color=[j / 255 for j in colors_vertices[22]], alpha=alpha)
        # plot kp
        #set_trace()
        for k in range(len(keypoints)):
            if keypoints[k].mean() != 0:
                plt.plot(keypoints[k][0], keypoints[k][1], 'o', markersize=markersize,
                        color=[j / 255 for j in colors_vertices[22]], alpha=alpha)
                
        for k in range(len(predkeypoints)):
            if predkeypoints[k].mean() != 0:
                plt.plot(predkeypoints[k][0], predkeypoints[k][1], 'o', markersize=markersize,
                        color=[j / 255 for j in colors_vertices[0]], alpha=alpha)
        #pdb.set_trace()
        plt.axis('off')
        plt.savefig(currentpath,bbox_inches='tight', pad_inches = 0)

    def visualize_video(self, frames, twodskeleton, twodpred, cur_step,j,phase,current_path, flag):
        #pdb.set_trace()
        if not os.path.isdir(current_path):
            os.makedirs(current_path, 0o777)
        if not os.path.isdir(current_path + '/skeletonimgs'):
            os.makedirs(current_path + '/skeletonimgs', 0o777)
        if not os.path.isdir(current_path + '/' + flag + '_frames'):
            os.makedirs(current_path + '/' + flag + '_frames', 0o777)
        for i in range(30):
            #pdb.set_trace()
            name = frames[i].split("/")[-1]
            img=cv2.imread(frames[i])
            img = (img*1.0).astype('int')

            #pdb.set_trace()
            cur_skeleton = twodskeleton[i].cpu().numpy()
            pred_skeleton = twodpred[i].cpu().numpy()
            blurimg = img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100]
            if int(cur_skeleton[0][1]) - 100 > 0 and int(cur_skeleton[0][1]) + 100 < img.shape[0] and int(cur_skeleton[0][0]) - 100 > 0 and int(cur_skeleton[0][0]) + 100 <img.shape[1]:
                blurred_part = cv2.blur(blurimg, ksize=(50, 50))
                img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100] = blurred_part
            self.plot_skel(cur_skeleton,pred_skeleton, img[...,::-1], current_path + "/" + flag + '_frames/' + str(i).zfill(6) + ".png", i)
            #cv2.imwrite(current_path + "/" + flag + '_frames/' + str(i).zfill(6) + ".png", img)
            #plt.figure()
            #plt.imshow(img)
            #plt.scatter(cur_skeleton[:,0],cur_skeleton[:,1],s=40)
            #plt.savefig(current_path + "/" + str(i).zfill(6) + ".png")


        command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}' + "/" + flag + '_frames/' + '%06d.png',  '-c:v', 'libx264','-vf' ,'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p', f'{current_path}/out' + flag + '.mp4']

        print(f'Running \"{" ".join(command)}\"')
        #pdb.set_trace()
        subprocess.call(command)

        return current_path

    def visualize_skeleton(self, threedskeleton, bboxes, predcam, cur_step,ex,phase,current_path, current_path_parent):

        proj =5000.0
        translation = self.convert_pare_to_full_img_cam(predcam,bboxes[:,2:3],bboxes[:,:2],1080,1920,proj)
        #pdb.set_trace()
        twodkpts, skeleton = self.perspective_projection(threedskeleton, torch.unsqueeze(torch.eye(3),dim=0), translation[0].float(),torch.tensor([[proj]]), torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0))
        #twodkpts=twodkpts[0]
        #skeleton=skeleton[0]

        colors=['b','b','r','r','r','g','g','g','y','r','r','r','g','g','g','b','b','b','b','g','g','g','r',
        'r','r','r','r','r','g','g','g','r','r','r','l','l','l','b','b','y','y','y','b','b','b',
        'b','b','b','b']

        fig = plt.figure(figsize=(15,4.8))
        ax = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122,projection='3d')
        #ax3 = fig.add_subplot(133)
        #current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/'  + movie + "/" + str(cur_step) + "/" + str(ex)
        #current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
        
        for i in range(len(threedskeleton)):
            cur = time.time()
            maxvaly,minvaly = torch.max(skeleton[:,:,2]),torch.min(skeleton[:,:,2])
            maxvalz,minvalz = torch.max(skeleton[:,:,1]),torch.min(skeleton[:,:,1])
            def thistakesalongtime():
                print(i,"i")
                curskeleton = skeleton[i].cpu().numpy()
                
                cur = time.time()
                for j in range(curskeleton.shape[0]):
                    #print(j,"j")
                #plt.figure()
                    c = colors[j]
                    if c == 'b':
                        newc = 'blue' #'#ff0000'
                    elif c=='r':
                        newc= 'red' #'#0000ff'
                    else:
                        newc = '#0f0f0f'
                    if j == 25 or j==30:
                        newc = 'yellow'
                    ax.scatter3D(curskeleton[j][0],curskeleton[j][2],curskeleton[j][1],c=newc)
                    ax2.scatter3D(curskeleton[j][0],curskeleton[j][2],curskeleton[j][1],c=newc)
            #
                #ax.view_init(-75, -90,90)
                    #ax.invert_yaxis()
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                ax.set_xlim3d([-1.0, 2.0])
                ax.set_zlim3d([minvalz, maxvalz])
                ax.set_ylim3d([minvaly, maxvaly])
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                ax.invert_zaxis()
                ax2.set_xlim3d([-1.5, 2.0])
                ax2.set_zlim3d([minvalz, maxvalz])
                ax2.set_ylim3d([minvaly, maxvaly])
                ax2.invert_zaxis()
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo

                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("y")
                
                ax.view_init(0,180)

                ax2.set_xlabel("x")
                ax2.set_ylabel("z")
                ax2.set_zlabel("y")
                ax2.view_init(0,-90)
                curtwo = time.time()
                print(curtwo-cur)
                cur = curtwo
                plt.savefig(current_path + "/3dskeleton" + str(i).zfill(6) + ".png")

                if i == len(threedskeleton)-1:
                    time.sleep(0.5)
                    command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/3dskeleton%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/out3dskeleton.mp4']
                    print(f'Running \"{" ".join(command)}\"')
                    subprocess.call(command)
            proc = mp.Process(target=thistakesalongtime) # , args=(1, 2))
            proc.start()

            # plt.savefig(current_path + "/3dskeleton" + str(i).zfill(6) + ".png")
            curtwo = time.time()
            print(curtwo-cur, "here")
            cur = curtwo

        return current_path

    def visualize_mesh_activation(self,twodskeleton, list_of_verts,list_of_origcam,frames, emg_values,emg_values_pred,current_path):
        #maxvals = torch.ones(emg_values.shape)*200.0

        # DEBUG #
        #pdb.set_trace()
        #if frames[0].split("/")[-2].split("_")[1][2]=='4':
        
        #maxvals = torch.unsqueeze(torch.tensor([139,174,155,127,113,246,84,107]),dim=1).repeat(1,30)
        #maxvals=torch.unsqueeze(torch.tensor([163.0, 243.0, 267.0, 167.0, 162.0, 212.0, 289.0, 237.0]),dim=1).repeat(1,30)
        #else:
        #maxvals = torch.unsqueeze(torch.tensor([139,174,155,127,113,246,84,107]),dim=1).repeat(1,30)
        #maxvals = torch.unsqueeze(torch.tensor([139,174,155,127,113,246,84,107]),dim=1).repeat(1,30)
        #maxvals = torch.unsqueeze(torch.tensor([70,70]),dim=1).repeat(1,30)
        #maxvalspred = torch.unsqueeze(torch.tensor([70,70,70,70,70,70,70,70]),dim=1).repeat(1,30)
        #maxvals=torch.unsqueeze(torch.tensor([163.0, 243.0, 267.0, 167.0, 162.0, 212.0, 289.0, 237.0]),dim=1).repeat(1,30)
        #emg_values_pred = emg_values_pred.permute(1,0)
        #print(emg_values.shape, maxvals.shape, emg_values_pred.shape)
        #pdb.set_trace()
        emg_values = emg_values
        

        emg_values_pred = emg_values_pred
        if not os.path.isdir(current_path):
            os.makedirs(current_path, 0o777)
        if not os.path.isdir(current_path + "/meshimgsfront/" ):
            os.makedirs(current_path + "/meshimgsfront/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsback/" ):
            os.makedirs(current_path + "/meshimgsback/", 0o777)
        if not os.path.isdir(current_path + "/meshimgstruth/" ):
            os.makedirs(current_path + "/meshimgstruth/", 0o777)
        if not os.path.isdir(current_path + "/meshimgspred/" ):
            os.makedirs(current_path + "/meshimgspred/", 0o777)

        if not os.path.isdir(current_path + "/meshimgpredfront/" ):
            os.makedirs(current_path + "/meshimgpredfront/", 0o777)
        if not os.path.isdir(current_path + "/meshimgpredback/" ):
            os.makedirs(current_path + "/meshimgpredback/", 0o777)
        if not os.path.isdir(current_path + "/meshimgtruthfront/" ):
            os.makedirs(current_path + "/meshimgtruthfront/", 0o777)
        if not os.path.isdir(current_path + "/meshimgtruthback/" ):
            os.makedirs(current_path + "/meshimgtruthback/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsnobkgdfront/" ):
            os.makedirs(current_path + "/meshimgsnobkgdfront/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsnobkgdback/" ):
            os.makedirs(current_path + "/meshimgsnobkgdback/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsnobkgdfrontpred/" ):
            os.makedirs(current_path + "/meshimgsnobkgdfrontpred/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsnobkgdbackpred/" ):
            os.makedirs(current_path + "/meshimgsnobkgdbackpred/", 0o777)

        

        for i in range(len(frames)):
            cur_skeleton = twodskeleton[i].cpu().numpy()
            img=cv2.imread(frames[i])
            img = (img*1.0).astype('int')
            if int(cur_skeleton[0][1]) - 100 > 0 and int(cur_skeleton[0][1]) + 100 < img.shape[0] and int(cur_skeleton[0][0]) - 100 > 0 and int(cur_skeleton[0][0]) + 100 <img.shape[1]:
                blurimg = img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100]
                blurred_part = cv2.blur(blurimg, ksize=(50, 50))
                img[int(cur_skeleton[0][1]) - 100:int(cur_skeleton[0][1]) + 100, int(cur_skeleton[0][0]) - 100:int(cur_skeleton[0][0]) + 100] = blurred_part
            #img = img[...,::-1]
            #pdb.set_trace()
            #if frames[i].split("/")[-2].split("_")[1][2]=='4':
            #    back = cv2.imread('sruthibackpic.png')
            #else:
            #    back = cv2.imread('finalback.png')
            #back = cv2.imread('finalback.png')
            #back = cv2.imread('finalback3.png')
            back = cv2.imread('finalback.png')
            back = (back*1.0).astype('int')
            
            #back = img
            
            orig_height, orig_width = img.shape[:2]
            mesh_filename = None
            mc = (0, 0, 0)
            frame_verts = list_of_verts[i]
            frame_cam = list_of_origcam[i]

            fronttruthimg = self.renderer.render(
                img,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=False,
            )

            fronttruthimgnobkgd = self.renderer.render(
                np.ones(img.shape).astype('int')*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=False,
            )

            frontpredimgnobkgd = self.renderer.render(
                np.ones(img.shape).astype('int')*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=True,
            )

            cv2.imwrite(os.path.join(current_path + "/meshimgtruthfront/", str(i).zfill(6) + '.png'), fronttruthimg)

            frontpredimg = self.renderer.render(
                img,
                frame_verts,
                emg_values = emg_values_pred[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=True,
            )

            cv2.imwrite(os.path.join(current_path + "/meshimgpredfront/", str(i).zfill(6) + '.png'), frontpredimg)
            frontimg = np.concatenate([fronttruthimg, frontpredimg], axis=1)

            cv2.imwrite(os.path.join(current_path + "/meshimgsfront/", str(i).zfill(6) + '.png'), frontimg)

            backtruth = self.renderer.render(
                back,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=False,
            )

            backtruthnobkgd = self.renderer.render(
                np.ones(img.shape)*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=False,
            )

            backprednobkgd = self.renderer.render(
                np.ones(img.shape)*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=True,
            )

            cv2.imwrite(os.path.join(current_path + "/meshimgtruthback/", str(i).zfill(6) + '.png'), backtruth)

            backpred = self.renderer.render(
                back,
                frame_verts,
                emg_values = emg_values_pred[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=True,
            )
            cv2.imwrite(os.path.join(current_path + "/meshimgpredback/", str(i).zfill(6) + '.png'), backpred)

            imgback = np.concatenate([backtruth, backpred], axis=1)
            imggt = np.concatenate([fronttruthimg, backtruth], axis=1)
            imgpred = np.concatenate([frontpredimg, backpred], axis=1)
            #imgnobkgd = np.concatenate([fronttruthimgnobkgd, backtruthnobkgd], axis=1)

            cv2.imwrite(os.path.join(current_path + "/meshimgsback/", str(i).zfill(6) + '.png'), imgback)
            cv2.imwrite(os.path.join(current_path + "/meshimgstruth/", str(i).zfill(6) + '.png'), imggt)
            cv2.imwrite(os.path.join(current_path + "/meshimgspred/", str(i).zfill(6) + '.png'), imgpred)
            cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdfront/", str(i).zfill(6) + '.png'), fronttruthimgnobkgd)
            cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdback/", str(i).zfill(6) + '.png'), backtruthnobkgd)
            cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdfrontpred/", str(i).zfill(6) + '.png'), frontpredimgnobkgd)
            cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdbackpred/", str(i).zfill(6) + '.png'), backprednobkgd)

        command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgsfront' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshfront.mp4']
        commandback = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgsback' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshback.mp4']
        commandtruth = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgstruth' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshtruth.mp4']
        commandpred = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgspred' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshpred.mp4']


        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)

        print(f'Running \"{" ".join(commandtruth)}\"')
        subprocess.call(commandtruth)

        print(f'Running \"{" ".join(commandpred)}\"')
        subprocess.call(commandpred)

        print(f'Running \"{" ".join(commandback)}\"')
        subprocess.call(commandback)



    def animate(self, list_of_data, list_of_pred_data, labels, part, trialnum, current_path, epoch):
    
        #pdb.set_trace()
        t = np.linspace(0, len(list_of_data[0])/10.0, len(list_of_data[0]))
        numDataPoints = len(t)
        colors = ['g','c']
        colorspred = ['r','b']

        
            #ax.set_ylabel('y')

        def animate_func(num):
            #print(num, "hi")
            ax.clear()  # Clears the figure to update the line, point,   
            for i,limb in enumerate(list_of_data):
                ax.plot(t[:num],limb[:num], c=colors[i], label=labels[i])
            for i,limb in enumerate(list_of_pred_data):
                ax.plot(t[:num],limb[:num], c=colorspred[i], label=labels[i] + "pred")
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
        #print(numDataPoints)
        line_ani = animation.FuncAnimation(fig, animate_func, interval=100,   
                                        frames=numDataPoints)
        print("saving_animation")
        #FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'h264_v4l2m2m'])
        line_ani.save(current_path + "/" + str(part) + '_emg.mp4')

    def handle_val_step(self, device, epoch, phase, cur_step, total_step, steps_per_epoch,
                            data_retval, model_retval, loss_retval):

            #pdb.set_trace()
            
            if cur_step % self.step_interval == 0:

                exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars
                #pdb.set_trace()

                #total_loss = loss_retval['total']
                for j in range(data_retval['2dskeleton'].shape[0]):
                    

                    framelist = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]
                    movie = framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                    #pdb.set_trace()
                    
                    
                    current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
                    
                    current_path_parent = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/"
                    
                    if not os.path.isdir(current_path):
                        os.makedirs(current_path, 0o777)
                    with open(current_path_parent + "file_list.txt","a") as f:
                        f.write(current_path + " \n")
                    #pdb.set_trace()
                    #temp = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],cur_step,j,phase,current_path, 'truth')
                    #pdb.set_trace()
                    if self.args.predemg!= 'True' and self.args.threed=='False':
                        pred_pose = model_retval['pose_output'][0].reshape(30,25,3)
                        pred_pose[:,:,0] = pred_pose[:,:,0]*1080
                        pred_pose[:,:,1] = pred_pose[:,:,1]*1920

                        temp = self.visualize_video(framelist,pred_pose[:,:25,:],cur_step,j,phase,current_path, 'pred')
                        #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                    if self.args.predemg != 'True':

                        result = smplify_runner(model_retval['pose_output'][0].to(device).type(torch.float32),data_retval['betas'][0].to(device).type(torch.float32),data_retval['predcam'][0].to(device).type(torch.float32),data_retval['2dskeletonsmpl'][0].to(device).type(torch.float32),device,1,lr=1.0,opt_steps=1,use_lbfgs=True,pose2aa=False)
                        predverts = result[1]
                        framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
                        movie = framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                        current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/pred/"
                        #pdb.set_trace()
                        self.visualize_mesh_activation(data_retval['2dskeleton'][0],predverts,data_retval['origcam'][0],framelist, data_retval['emg_values'][0],data_retval['emg_values'][0],current_path)

                        result = smplify_runner(data_retval['pose'][0].to(device).type(torch.float32),data_retval['betas'][0].to(device).type(torch.float32),data_retval['predcam'][0].to(device).type(torch.float32),data_retval['2dskeletonsmpl'][0].to(device).type(torch.float32),device,1,lr=1.0,opt_steps=1,use_lbfgs=True,pose2aa=False)
                        predverts = result[1]
                        framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
                        movie = framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                        current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/gt/"
                        #pdb.set_trace()
                        self.visualize_mesh_activation(data_retval['2dskeleton'][0],predverts,data_retval['origcam'][0],framelist, data_retval['emg_values'][0],data_retval['emg_values'][0],current_path)
                        #threedskeleton = model_retval['pose_output'][j].reshape(30,25,3)
                        #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                        
                        """bboxes = data_retval['bboxes'][:,:30,:]
                        predcam = data_retval['predcam'][:,:30,:]
                        proj = 5000.0
                        #pdb.set_trace()
                        
                        height= bboxes[:,:,2:3].reshape(bboxes.shape[0]*bboxes.shape[1]).to(device)
                        center = bboxes[:,:,:2].reshape(bboxes.shape[0]*bboxes.shape[1],-1).to(device)
                        focal=torch.tensor([[proj]]).to(device).repeat(height.shape[0],1)
                        predcamelong = predcam.reshape(predcam.shape[0]*predcam.shape[1],-1).to(device)
                        #pdb.set_trace()
                        translation = self.convert_pare_to_full_img_cam(predcamelong,height,center,1080,1920,focal[:,0])
                        #pdb.set_trace()
                        #reshapethreed= threedskeleton.reshape(threedskeleton.shape[0]*threedskeleton.shape[1],threedskeleton.shape[2],threedskeleton.shape[3])
                        rotation=torch.unsqueeze(torch.eye(3),dim=0).repeat(1,1,1).to(device)
                        focal=torch.tensor([[proj]]).to(device).repeat(translation.shape[0],1)
                        imgdimgs=torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0).repeat(1,1).to(device)
                        #pdb.set_trace()
                        twodkpts, skeleton = self.perspective_projection(threedskeleton, rotation, translation.float(),focal[:,0], imgdimgs)
                        #pdb.set_trace()
                        #current[:,:,0] = current[:,:,0]*1080
                        #current[:,:,1] = current[:,:,1]*1920
                        #pdb.set_trace()
                        np.save(current_path + "/predpose2d" + str(cur_step) + ".npy",twodkpts.detach().cpu().numpy())
                        np.save(current_path + "/gtpose2d" + str(cur_step) + ".npy",data_retval['2dskeleton'][j][:,:25,:].cpu().numpy())
                        np.save(current_path + "/gtpose3d" + str(cur_step) + ".npy",data_retval['3dskeleton'][j][:,:25,:].cpu().numpy())
                        np.save(current_path + "/predpose3d" + str(cur_step) + ".npy",threedskeleton.detach().cpu().numpy())
                        current_path = self.visualize_video(framelist,twodkpts,data_retval['2dskeleton'][j][:,:25,:],cur_step,j,phase,current_path, 'pred')"""
                    #current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,current_path, current_path_parent)
                        #if self.args.predemg == 'True':
                        #    self.visualize_mesh_activation(data_retval['2dskeleton'][j],data_retval['verts'][j],data_retval['orig_cam'][j],framelist, data_retval['emg_values'][j],model_retval['emg_output'][j].cpu().detach(),current_path)"""
        
                    gtnp = model_retval['emg_gt'].detach().cpu().numpy()
                    if self.args.predemg == 'True':
                        prednp = model_retval['emg_output'].detach().cpu().numpy()
                    else:
                        prednp = gtnp
                    
                    np.save(current_path + "/gtnp" + str(cur_step) + ".npy",gtnp)
                    np.save(current_path + "/prednp" + str(cur_step) + ".npy",prednp)
                    rangeofmuscles=['LeftQuad','LeftHamstring','LeftLateral','LeftBicep','RightQuad','RightHamstring','RightLateral','RightBicep']
                    """for i in range(model_retval['emg_gt'].shape[1]):
                        gt_values = model_retval['emg_gt'][j,i,:].cpu()*100
                        #gt_values[gt_values>100.0] = 100.0
                        pred_values = model_retval['emg_output'][j][i].cpu()*100.0
                        #pdb.set_trace()
                        self.animate([gt_values.numpy()],[pred_values.detach().numpy()],[rangeofmuscles[i]],rangeofmuscles[i],2,current_path,epoch)"""
                    ###DEBUG
                    for i in range(model_retval['emg_gt'].shape[1]):
                        gt_values = model_retval['emg_gt'][j,i,:].cpu()
                        #gt_values = data_retval['old_emg_values'][j,i,:].cpu()
                        if self.args.predemg == 'True':
                            #pred_values = data_retval['emg_values'][j][i].cpu()
                            pred_values = model_retval['emg_output'][j][i].cpu()
                        else:
                            #pred_values = data_retval['emg_values'][j][i].cpu()
                            pred_values = gt_values
                        #pdb.set_trace()
                        self.animate([gt_values.numpy()],[pred_values.detach().numpy()],[rangeofmuscles[i]],rangeofmuscles[i],2,current_path,epoch)
                

                        # Print metrics in console.
                        """command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_206_leftbicep_emg.mp4',
                        'i', f'{current_path}/epoch_206_rightquad_emg.mp4','-filter_complex',
                        'hstack=inputs=4', f'{current_path}/total.mp4']
                        print(f'Running \"{" ".join(command)}\"')
                        subprocess.call(command)"""
                    
                

    def epoch_finished(self, epoch):
        returnval = self.commit_scalars(step=epoch)
        return returnval

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval):

        psnr = inference_retval['psnr']

        # Print metrics in console.
        self.info(f'[Step {cur_step} / {num_steps}]  '
                  f'psnr: {psnr.mean():.2f} ± {psnr.std():.2f}')

        # Save input, prediction, and ground truth images.
        rgb_input = inference_retval['rgb_input']
        rgb_output = inference_retval['rgb_output']
        rgb_target = inference_retval['rgb_target']

        gallery = np.stack([rgb_input, rgb_output, rgb_target])
        gallery = np.clip(gallery, 0.0, 1.0)
        file_name = f'rgb_iogt_s{cur_step}.png'
        online_name = f'rgb_iogt'
        self.save_gallery(gallery, step=cur_step, file_name=file_name, online_name=online_name)
