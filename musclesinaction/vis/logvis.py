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
            current_path = self.visualize_skeleton(threedskeleton.detach().cpu(),data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,framelist[0].split("/")[-2])

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

    def plot_skel(self, keypoints, predkeypoints, img, currentpath, i, flag, markersize=3, linewidth=2, alpha=0.7):
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
        if flag=="pred":
            for vertices, color in limb_seq:
                if predkeypoints[vertices[0]].mean() != 0 and predkeypoints[vertices[1]].mean() != 0:
                    plt.plot([predkeypoints[vertices[0]][0], predkeypoints[vertices[1]][0]],
                            [predkeypoints[vertices[0]][1], predkeypoints[vertices[1]][1]], linewidth=linewidth,
                            color=[j / 255 for j in colors_vertices[22]], alpha=alpha) #22,10
                
        for vertices, color in limb_seq:
            if keypoints[vertices[0]].mean() != 0 and keypoints[vertices[1]].mean() != 0:
                plt.plot([keypoints[vertices[0]][0], keypoints[vertices[1]][0]],
                        [keypoints[vertices[0]][1], keypoints[vertices[1]][1]], linewidth=linewidth,
                        color=[j / 255 for j in colors_vertices[8]], alpha=alpha)
                
        # plot kp
        #set_trace()
        if flag=="pred":
            for k in range(len(predkeypoints)):
                if predkeypoints[k].mean() != 0:
                    plt.plot(predkeypoints[k][0], predkeypoints[k][1], 'o', markersize=markersize,
                            color=[j / 255 for j in colors_vertices[22]], alpha=alpha) # 22
                
        for k in range(len(keypoints)):
            if keypoints[k].mean() != 0:
                plt.plot(keypoints[k][0], keypoints[k][1], 'o', markersize=markersize,
                        color=[j / 255 for j in colors_vertices[8]], alpha=alpha)
                
        
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
            self.plot_skel(cur_skeleton,pred_skeleton, img[...,::-1], current_path + "/" + flag + '_frames/' + str(i).zfill(6) + ".png", i, flag)
            
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
        twodkpts, skeleton = self.perspective_projection(threedskeleton.detach().cpu(), torch.unsqueeze(torch.eye(3),dim=0), translation[0].float(),torch.tensor([[proj]]), torch.unsqueeze(torch.tensor([1080.0/2, 1920.0/2]),dim=0))
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
                for vertices, color in limb_seq:
                    ax.plot3D([curskeleton[vertices[0]][0], curskeleton[vertices[1]][0]],
                            [curskeleton[vertices[0]][1], curskeleton[vertices[1]][1]],
                            [curskeleton[vertices[0]][2], curskeleton[vertices[1]][2]], 
                            linewidth=3,
                            color=[j / 255 for j in colors_vertices[8]], alpha=0.7)
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
                ax2.view_init(0,-45)
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
        if not os.path.isdir(current_path + "/meshobj/" ):
            os.makedirs(current_path + "/meshobj/", 0o777)
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

        list_of_vertex_colors =[]

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
            back = cv2.imread('backplain.png')
            back = (back*1.0).astype('int')
            
            #back = img
            
            orig_height, orig_width = img.shape[:2]
            mesh_filename = None
            mc = (0, 0, 0)
            frame_verts = list_of_verts[i]
            frame_cam = list_of_origcam[i]

            fronttruthimg, vertex_colors = self.renderer.render(
                "True",
                current_path + "/meshobj/" + str(i),
                img,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=False,
            )
            list_of_vertex_colors.append(vertex_colors)

            fronttruthimgnobkgd,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
                np.ones(img.shape).astype('int')*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=False,
            )

            frontpredimgnobkgd,vertex_colors_pred = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
                np.ones(img.shape).astype('int')*255,
                frame_verts,
                emg_values = emg_values_pred[:,i],
                cam=frame_cam,
                color=mc,
                front=True,
                mesh_filename=mesh_filename,
                pred=True,
            )

            np.save(current_path + "/vertexcolorspred_" + str(i) + ".npy", vertex_colors_pred)

            cv2.imwrite(os.path.join(current_path + "/meshimgtruthfront/", str(i).zfill(6) + '.png'), fronttruthimg)

            frontpredimg,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
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
            #pdb.set_trace()

            cv2.imwrite(os.path.join(current_path + "/meshimgsfront/", str(i).zfill(6) + '.png'), frontimg)

            backtruth,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
                back,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=False,
            )

            backtruthnobkgd,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
                np.ones(img.shape)*255,
                frame_verts,
                emg_values = emg_values[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=False,
            )

            backprednobkgd,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
                np.ones(img.shape)*255,
                frame_verts,
                emg_values = emg_values_pred[:,i],
                cam=frame_cam,
                color=mc,
                front=False,
                mesh_filename=mesh_filename,
                pred=True,
            )

            cv2.imwrite(os.path.join(current_path + "/meshimgtruthback/", str(i).zfill(6) + '.png'), backtruth)

            backpred,_ = self.renderer.render(
                "False",
                current_path + "/meshobj/" + str(i),
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
            #cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdfront/", str(i).zfill(6) + '.png'), fronttruthimgnobkgd)
            #cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdback/", str(i).zfill(6) + '.png'), backtruthnobkgd)
            #cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdfrontpred/", str(i).zfill(6) + '.png'), frontpredimgnobkgd)
            #cv2.imwrite(os.path.join(current_path + "/meshimgsnobkgdbackpred/", str(i).zfill(6) + '.png'), backprednobkgd)

        #pdb.set_trace()
        np.save(current_path + "/" + "vertexcolors.npy", np.array(list_of_vertex_colors))
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
                            data_retval, model_retval, loss_retval, flag):

            #pdb.set_trace()
            
            if cur_step % self.step_interval == 0:

                exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars
                #pdb.set_trace()

                #total_loss = loss_retval['total']
                for j in range(data_retval['2dskeleton'].shape[0]):
                    

                    framelist = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]
                    filepath = data_retval['filepath']
                    movie = filepath.split("/")[-2] + filepath.split("/")[-1]#framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                    #pdb.set_trace()
                    
                    
                    #current_path = "../../../../local/vondrick/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
                    current_path = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
                    current_path_parent = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/"

                    #current_path_parent = "../../../../local/vondrick/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/"
                    
                    if not os.path.isdir(current_path):
                        os.makedirs(current_path, 0o777)
                    if not os.path.isdir(current_path_parent):
                        os.makedirs(current_path_parent, 0o777)
                    with open(current_path_parent + "file_list.txt","a") as f:
                        f.write(current_path + " \n")
                    #pdb.set_trace()

                    #temp = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],cur_step,j,phase,current_path, 'truth')
                    #pdb.set_trace()
                    #self, frames, twodskeleton, twodpred, cur_step,j,phase,current_path, flag):
                    if self.args.predemg!= 'True' and self.args.threed=='False':
                        pred_pose = model_retval['pose_output'][0].reshape(30,25,2)
                        pred_pose[:,:,0] = pred_pose[:,:,0]*1080
                        pred_pose[:,:,1] = pred_pose[:,:,1]*1920

                        temp = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],pred_pose[:,:25,:],cur_step,j,phase,current_path, 'pred')
                        #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                    elif self.args.predemg != 'True' and self.args.threed=='True':

                        threedskeleton = model_retval['pose_output'][0].reshape(30,25,3)
                        bboxes = data_retval['bboxes'][:,:30,:]
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
                        #current_path = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],twodkpts,cur_step,j,phase,current_path, 'pred')
                        #current_path = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],twodkpts,cur_step,j,phase,current_path, 'gt')
                        #pdb.set_trace()
                        temp = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],twodkpts[:,:25,:],cur_step,j,phase,current_path, 'pred')
                        temp = self.visualize_video(framelist,data_retval['2dskeleton'][j][:,:25,:],twodkpts[:,:25,:],cur_step,j,phase,current_path, 'gt')
                        current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,current_path, current_path_parent)
                        #pdb.set_trace()
                    else:
                        framelist = [data_retval['frame_paths'][i][0] for i in range(len(data_retval['frame_paths']))]
                        filepath = data_retval['filepath']
                        movie = filepath.split("/")[-2] + filepath.split("/")[-1]#framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                        #movie = framelist[0].split("/")[-4] + "/" + framelist[0].split("/")[-3]
                        #current_path = "../../../../local/vondrick/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/gt" + flag + "/"
                        current_path = "muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(0) + "/gt" + flag + "/"

                        themax = torch.unsqueeze(data_retval['max'],dim=2)
                        themin = torch.unsqueeze(data_retval['min'],dim=2)
                        #pdb.set_trace()
                        normalized_gt_emg = (model_retval['emg_gt'].detach().cpu()-themin)/themax
                        normalized_pred_emg = (model_retval['emg_output'].detach().cpu()-themin)/themax
                        #normalized_pred_emg[normalized_pred_emg<0] = 0.0
                        #normalized_pred_emg[normalized_pred_emg>1] = 1.0
                        #pdb.set_trace()
                        if not os.path.isdir(current_path):
                            os.makedirs(current_path, 0o777)
                        #pdb.set_trace()
                        self.visualize_mesh_activation(data_retval['2dskeleton'][j],data_retval['verts'][j],data_retval['origcam'][j],framelist,normalized_gt_emg[j],normalized_pred_emg[j],current_path)
                        #threedskeleton = model_retval['pose_output'][j].reshape(30,25,3)
                        #threedskeleton = data_retval['3dskeleton'][0][:,:25,:].to(device)
                        #current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,current_path, current_path_parent)
                        #if self.args.predemg == 'True':
                        #    self.visualize_mesh_activation(data_retval['2dskeleton'][j],data_retval['verts'][j],data_retval['orig_cam'][j],framelist, data_retval['emg_values'][j],model_retval['emg_output'][j].cpu().detach(),current_path)"""
        
                    gtnp = data_retval['emg_values'].detach().cpu().numpy()
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
