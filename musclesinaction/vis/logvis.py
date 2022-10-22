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
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import cv2
import subprocess
import time
import multiprocessing as mp

class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context):
        
        self.step_interval = 1
        self.num_exemplars = 4  # To increase simultaneous examples in wandb during train / val.
        self.maxemg = args.maxemg
        self.classif = args.classif
        self.args = args
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
                gt_values[gt_values>100.0] = 100.0
                pred_values = model_retval['emg_output'][j][0].cpu()*100.0
                pred_values[pred_values>100.0] = 100.0
                self.animate([gt_values.numpy()],[pred_values.detach().numpy()],['left_quad'],'leftleg',2,current_path,epoch)

            # Print metrics in console.

            command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_159_leftleg_emg.mp4','-filter_complex',
            'hstack=inputs=3', f'{current_path}/total.mp4']
            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                    f'total_loss: {total_loss:.3f}  ')

    def visualize_video(self, frames, twodskeleton, cur_step,j,phase,movie):

        current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/' + movie + "/" + str(cur_step) + "/" + str(j)
        if not os.path.isdir(current_path):
            os.makedirs(current_path, 0o777)
            for i in range(len(frames)):
                img=cv2.imread(frames[i])
                img = img[...,::-1]
                cur_skeleton = twodskeleton[i].cpu().numpy()
                plt.figure()
                plt.imshow(img)
                plt.scatter(cur_skeleton[:,0],cur_skeleton[:,1],s=40)
                plt.savefig(current_path + "/" + str(i).zfill(6) + ".png")

            command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/out.mp4']

            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)

        return current_path

    def visualize_skeleton(self, threedskeleton, bboxes, predcam, cur_step,ex,phase,movie):

        proj =5000.0
        translation = self.convert_pare_to_full_img_cam(predcam,bboxes[:,2:3],bboxes[:,:2],1080,1920,proj)
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
        current_path = "../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/'  + movie + "/" + str(cur_step) + "/" + str(ex)
        with open("../../www-data/mia/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/'  + movie + "/file_list.txt","a") as f:
            path = "../../mia/muscleresultviz/muscleresults/" + self.args.name + '_' + phase + '_viz_digitized/'  + movie + "/" + str(cur_step) + "/" + str(ex)

            f.write(path + " \n")
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

    def visualize_mesh_activation(self,list_of_verts,list_of_origcam,frames, emg_values,emg_values_pred,current_path):
        #maxvals = torch.ones(emg_values.shape)*200.0
        maxvals = torch.unsqueeze(torch.tensor([139,174,155,127,113,246,84,107]),dim=1).repeat(1,30)
        
        emg_values = emg_values/maxvals

        emg_values_pred = emg_values_pred/(maxvals/100)
        if not os.path.isdir(current_path + "/meshimgsfront/" ):
            os.makedirs(current_path + "/meshimgsfront/", 0o777)
        if not os.path.isdir(current_path + "/meshimgsback/" ):
            os.makedirs(current_path + "/meshimgsback/", 0o777)
            for i in range(len(frames)):
                img=cv2.imread(frames[i])
                
                orig_height, orig_width = img.shape[:2]
                mesh_filename = None
                mc = (255, 255, 255)
                frame_verts = list_of_verts[i]
                frame_cam = list_of_origcam[i]

                imgout = self.renderer.render(
                    img,
                    frame_verts,
                    emg_values = emg_values[:,i],
                    cam=frame_cam,
                    color=mc,
                    front=True,
                    mesh_filename=mesh_filename,
                )

                imgback = self.renderer.render(
                    img,
                    frame_verts,
                    emg_values = emg_values_pred[:,i],
                    cam=frame_cam,
                    color=mc,
                    front=True,
                    mesh_filename=mesh_filename,
                )
                img = np.concatenate([imgout, imgback], axis=1)

                cv2.imwrite(os.path.join(current_path + "/meshimgsfront/", str(i).zfill(6) + '.png'), img)

                imgout = self.renderer.render(
                    img,
                    frame_verts,
                    emg_values = emg_values[:,i],
                    cam=frame_cam,
                    color=mc,
                    front=False,
                    mesh_filename=mesh_filename,
                )

                imgback = self.renderer.render(
                    img,
                    frame_verts,
                    emg_values = emg_values_pred[:,i],
                    cam=frame_cam,
                    color=mc,
                    front=False,
                    mesh_filename=mesh_filename,
                )
                imgback = np.concatenate([imgout, imgback], axis=1)

                cv2.imwrite(os.path.join(current_path + "/meshimgsback/", str(i).zfill(6) + '.png'), imgback)

            command = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgsfront' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshfront.mp4']
            commandback = ['ffmpeg', '-framerate', '10', '-i',f'{current_path}/meshimgsback' +'/%06d.png',  '-c:v', 'libx264','-pix_fmt', 'yuv420p', f'{current_path}/' + 'meshback.mp4']

            print(f'Running \"{" ".join(command)}\"')
            subprocess.call(command)

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
        line_ani.save(current_path + "/epoch_" + str(epoch) + "_" + str(part) + '_emg.mp4')

    def handle_val_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                            data_retval, model_retval, loss_retval):

            if cur_step % self.step_interval == 0:

                exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars

                total_loss = loss_retval['total']
                for j in range(data_retval['bboxes'].shape[0]):
                    framelist = [data_retval['frame_paths'][i][j] for i in range(len(data_retval['frame_paths']))]
                    
                    current_path = self.visualize_video(framelist,data_retval['2dskeleton'][j],cur_step,j,phase,framelist[0].split("/")[-2])
                    threedskeleton = data_retval['3dskeleton'][j]
                    current_path = self.visualize_skeleton(threedskeleton,data_retval['bboxes'][j],data_retval['predcam'][j],cur_step,j,phase,framelist[0].split("/")[-2])
                    self.visualize_mesh_activation(data_retval['verts'][j],data_retval['orig_cam'][j],framelist, data_retval['emg_values'][j],model_retval['emg_output'][j].cpu().detach(),current_path)
                    
                    if self.classif:
                        values = data_retval['bined_left_quad'][0]-1
                        bins = data_retval['bins'][0]
                        gt_values = torch.index_select(bins.cpu(), 0, values.cpu())
                        pred_values = model_retval['emg_bins'][0].cpu()
                        
                        self.animate([gt_values.numpy()],[pred_values.detach().numpy()],['left_quad'],'leftleg',2,current_path,epoch)
                    else:
                        rangeofmuscles=['rightquad','rightham','rightbicep','righttricep','leftquad','leftham','leftbicep','lefttricep']
                        for i in range(8):
                            gt_values = model_retval['emg_gt'][j,i,:].cpu()*100
                            #gt_values[gt_values>100.0] = 100.0
                            pred_values = model_retval['emg_output'][j][i].cpu()*100.0
                            self.animate([gt_values.numpy()],[pred_values.detach().numpy()],[rangeofmuscles[i]],rangeofmuscles[i],2,current_path,epoch)
                    

                    # Print metrics in console.
                    """command = ['ffmpeg', '-i', f'{current_path}/out.mp4', '-i',f'{current_path}/out3dskeleton.mp4',  '-i', f'{current_path}/epoch_206_leftbicep_emg.mp4',
                    'i', f'{current_path}/epoch_206_rightquad_emg.mp4','-filter_complex',
                    'hstack=inputs=4', f'{current_path}/total.mp4']
                    print(f'Running \"{" ".join(command)}\"')
                    subprocess.call(command)"""
                    
                    self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                            f'total_loss: {total_loss:.3f}  ')
                

           
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
