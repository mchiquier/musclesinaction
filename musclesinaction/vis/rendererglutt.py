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

import math
import trimesh
import pyrender
import subprocess
import numpy as np
from pyrender.constants import RenderFlags
import os
from smplx import SMPL, SMPLH, SMPLX
from matplotlib import cm as mpl_cm, colors as mpl_colors
import json
import sys
from colour import Color
from PIL import ImageColor
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_smpl_faces():
    smpl = SMPL('musclesinaction/vibe_data', batch_size=1, create_transl=False)
    return smpl.faces

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    
    def download_url(self, url, outdir):
        print(f'Downloading files from {url}')
        cmd = ['wget', '-c', url, '-P', outdir]
        subprocess.call(cmd)
        file_path = os.path.join(outdir, url.split('/')[-1])
        return file_path

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def colorFader(self,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mpl.colors.to_rgb('yellow'))
        c2=np.array(mpl.colors.to_rgb('red'))
        return self.hex_to_rgb(mpl.colors.to_hex((1-mix)*c1 + mix*c2))

    def part_segm_to_vertex_colors(self, part_segm, n_vertices, front, emg_values, alpha=1.0, pred=False):
        vertex_labels = np.zeros(n_vertices)
        alpha_values = np.zeros(n_vertices)
        vertex_colors = np.ones((n_vertices, 4))
        red = Color("red")
        colors = list(red.range_to(Color("grey"),10))

        for part_idx, (k, v) in enumerate(part_segm.items()):
            
            if front:
                if k == 'rightUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[1]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[0]*255),255])
                elif k == 'leftUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[0]
                    vertex_labels[v] = 0
                    if pred:
                        vertex_colors[v] = np.array([0,0,int(emg_values[4]*255),255])
                    else:
                        vertex_colors[v] = np.array([0,0,int(emg_values[1]*255),255])
                else:
                    alpha_values[v] = 1.0
                    vertex_labels[v] = 4
                    vertex_colors[v] = np.array([222, 222, 222,255])
                """if k == 'rightUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[1]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[0]*255),255])
                elif k == 'leftUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[0]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[4]*255),255])
                elif k == 'leftArm':
                    alpha_values[v]=0.0 #1-emg_values[5]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[6]*255),255])
                elif k == 'rightArm':
                    alpha_values[v]=0.0 #1-emg_values[4]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[2]*255),255])
                elif k == 'neck':
                    alpha_values[v]=0.0
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,255,255])
                else:
                    alpha_values[v] = 1.0
                    vertex_labels[v] = 4
                    vertex_colors[v] = np.array([222, 222, 222,255])"""
            else:
                if k == 'rightUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[1]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[0]*255),255])
                elif k == 'leftUpLeg':
                    alpha_values[v]=0.0 #1-emg_values[0]
                    vertex_labels[v] = 0
                    if pred:
                        vertex_colors[v] = np.array([0,0,int(emg_values[4]*255),255])
                    else:
                        vertex_colors[v] = np.array([0,0,int(emg_values[1]*255),255])
                else:
                    alpha_values[v] = 1.0
                    vertex_labels[v] = 4
                    vertex_colors[v] = np.array([222, 222, 222,255])
                """
                if k == 'rightUpLeg':
                    alpha_values[v]=1-emg_values[3]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[1]*255),255])
                elif k == 'leftUpLeg':
                    alpha_values[v]=1-emg_values[2]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[5]*255),255])
                elif k == 'leftArm':
                    alpha_values[v]=1-emg_values[7]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[7]*255),255])
                elif k == 'rightArm':
                    alpha_values[v]=1-emg_values[6]
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,int(emg_values[3]*255),255])
                elif k == 'neck':
                    alpha_values[v]=0.0
                    vertex_labels[v] = 0
                    vertex_colors[v] = np.array([0,0,255,255])
                else:
                    alpha_values[v] = 1
                    vertex_labels[v] = 4
                    vertex_colors[v] = np.array([222, 222, 222,255])"""

        #vertex_colors = np.ones((n_vertices, 4))
        #vertex_colors = np.ones((n_vertices, 3))*255
        #vertex_colors[:, 3] = alpha_values
        #out = np.expand_dims(alpha_values,axis=1)*255
        #out = out.astype(int)
        #vertex_colors = vertex_colors.astype(int)
        #vertex_colors = np.concatenate([out,vertex_colors],axis=1)
        #vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

        return vertex_colors


    def render(self, img, verts, emg_values, cam, front=True, angle=None, axis=None, mesh_filename=None, color=[0.0, 0.0, 0.0],pred=False):

        #mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        Rxcam = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        Rxcamtwo = trimesh.transformations.rotation_matrix(math.radians(180), [0, 1, 0])
        
        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=1.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 0.5)
        )
        
        part_segm_filepath = 'musclesinaction/smpl_vert_segmentation.json'
        part_segm = json.load(open(part_segm_filepath))
        #vertices = verts.detach().numpy()

        vertex_colors = self.part_segm_to_vertex_colors(part_segm, verts.shape[0],front,emg_values, pred)
        if not front:
            mesh = trimesh.Trimesh(verts, self.faces, process=False, vertex_colors=vertex_colors)
            #mesh.apply_transform(Rx)
            mesh.apply_transform(Rxcam)
            mesh.apply_transform(Rxcamtwo)
            mesh = pyrender.Mesh.from_trimesh(mesh)
        else:
            mesh = trimesh.Trimesh(verts, self.faces, process=False, vertex_colors=vertex_colors)
            mesh.apply_transform(Rx)
            mesh = pyrender.Mesh.from_trimesh(mesh)
        

        mesh_node = self.scene.add(mesh, 'mesh')
        #jointscene = self.scene.add(joints_pcl, pose=Rx)

        camera_pose = np.eye(4)
        if front:
            cam_node = self.scene.add(camera, pose=camera_pose)
        else:
            cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        #if front:
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)
        #else:
        #    rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        #    output_img = rgb[:, :, :-1] 
        #    image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        #self.scene.remove_node(jointscene)

        return image
