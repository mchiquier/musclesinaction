'''
Logging and visualization logic.
'''

import musclesinaction.vis.logvisgen as logvisgen
import pdb
import csv
import wandb
import os
import numpy as np


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context):
        if 'batch_size' in args:
            self.step_interval = 3200 // args.batch_size
        else:
            self.step_interval = 200
        self.num_exemplars = 4  # To increase simultaneous examples in wandb during train / val.
        super().__init__(args.log_path, context, args.name)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval):

        if cur_step % self.step_interval == 0:

            exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars

            total_loss = loss_retval['total']

            # Print metrics in console.
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                      f'total_loss: {total_loss:.3f}  ')

           

    def epoch_finished(self, epoch):
        self.commit_scalars(step=epoch)

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
