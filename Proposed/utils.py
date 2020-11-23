import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

import matplotlib.pyplot as plt
import cv2
import torch
import pdb
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import logging

from multiprocessing import Pool

def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)

    return logdet

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion=None):

        self.exp_path = os.path.join(exp_path, args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_'))
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f'Current Exp Path: {self.exp_path}')

        self.writter = SummaryWriter(os.path.join(self.exp_path, 'logs'))


        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        self.encoding_steps = int(2 * args.sampling_rate)


        self.map_version = None


        self.beta = args.beta
        self.ploss_type = args.ploss_type        
        self.ploss_criterion = ploss_criterion

        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)

        if args.load_ckpt:
            self.load_checkpoint(args.load_ckpt)

        # Other Parameters
        self.best_valid_ade = 1e9
        self.best_valid_fde = 1e9
        self.start_epoch = args.start_epoch


        self.flow_based_decoder = False
        self.num_candidates = 1

    def train(self, num_epochs):
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info("==========================================================================================")

            train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch()
            valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.inference()

            ## unwrapping ADEs/FDEs
            train_minade2, train_avgade2, train_minade3, train_avgade3 = train_ades
            train_minfde2, train_avgfde2, train_minfde3, train_avgfde3 = train_fdes

            valid_minade2, valid_avgade2, valid_minade3, valid_avgade3 = valid_ades
            valid_minfde2, valid_avgfde2, valid_minfde3, valid_avgfde3 = valid_fdes

            self.best_valid_ade = min(valid_avgade3, self.best_valid_ade)
            self.best_valid_fde = min(valid_avgfde3, self.best_valid_fde)
            self.scheduler.step(scheduler_metric)

            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} '
                f'| Train minADE[2/3]: {train_minade2:0.4f} / {train_minade3:0.4f} | Train minFDE[2/3]: {train_minfde2:0.4f} / {train_minfde3:0.4f} '
                f'| Train avgADE[2/3]: {train_avgade2:0.4f} / {train_avgade3:0.4f} | Train avgFDE[2/3]: {train_avgfde2:0.4f} / {train_avgfde3:0.4f}'
            )

            logging_msg2 = (
                f'| Epoch: {epoch:02} | Valid Loss: {valid_loss:0.6f} '
                f'| Valid minADE[2/3]: {valid_minade2:0.4f} / {valid_minade3:0.4f} | Valid minFDE[2/3]: {valid_minfde2:0.4f} /{valid_minfde3:0.4f} '
                f'| Valid avgADE[2/3]: {valid_avgade2:0.4f} / {valid_avgade3:0.4f} | Valid avgFDE[2/3]: {valid_avgfde2:0.4f} /{valid_avgfde3:0.4f} '
                f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n'
            )

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, qloss=valid_qloss, ploss=valid_ploss, ade=valid_minade3, fde=valid_minfde3)

            # Log values to Tensorboard
            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Train_QLoss', train_qloss, epoch)
            self.writter.add_scalar('data/Train_PLoss', train_ploss, epoch)
            self.writter.add_scalar('data/Learning_Rate', self.get_lr(), epoch)

            self.writter.add_scalar('data/Train_minADE2', train_minade2, epoch)
            self.writter.add_scalar('data/Train_minFDE2', train_minfde2, epoch)
            self.writter.add_scalar('data/Train_minADE3', train_minade3, epoch)
            self.writter.add_scalar('data/Train_minFDE3', train_minfde3, epoch)

            self.writter.add_scalar('data/Train_avgADE2', train_avgade2, epoch)
            self.writter.add_scalar('data/Train_avgFDE2', train_avgfde2, epoch)
            self.writter.add_scalar('data/Train_avgADE3', train_avgade3, epoch)
            self.writter.add_scalar('data/Train_avgFDE3', train_avgfde3, epoch)
            self.writter.add_scalar('data/Scheduler_Metric', scheduler_metric, epoch)

            self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Valid_QLoss', valid_qloss, epoch)
            self.writter.add_scalar('data/Valid_PLoss', valid_ploss, epoch)
            self.writter.add_scalar('data/Valid_minADE2', valid_minade2, epoch)
            self.writter.add_scalar('data/Valid_minFDE2', valid_minfde2, epoch)
            self.writter.add_scalar('data/Valid_minADE3', valid_minade3, epoch)
            self.writter.add_scalar('data/Valid_minFDE3', valid_minfde3, epoch)

            self.writter.add_scalar('data/Valid_avgADE2', valid_avgade2, epoch)
            self.writter.add_scalar('data/Valid_avgFDE2', valid_avgfde2, epoch)
            self.writter.add_scalar('data/Valid_avgADE3', valid_avgade3, epoch)
            self.writter.add_scalar('data/Valid_avgFDE3', valid_avgfde3, epoch)

        self.writter.close()
        self.logger.info("Training Complete! ")
        self.logger.info(f'| Best Valid ADE: {self.best_valid_ade} | Best Valid FDE: {self.best_valid_fde} |')


    def train_single_epoch(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        if self.map_version == '2.0':
            """ Make position & distance embeddings for map v2.0"""
            with torch.no_grad():
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))
                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std

            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)

        c1 = -self.decoding_steps * np.log(2 * np.pi)
        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            scene_images, log_prior, \
            future_agent_masks, \
            num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
            num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
            two_mask, three_mask, \
            decode_start_vel, decode_start_pos, \
            scene_id, batch_size = batch

            # Detect dynamic sizes
            #batch_size = scene_images.size(0)
            num_three_agents = torch.sum(three_mask)
            #print(scene_images.shape)
            scene_images = scene_images.to(self.device)


            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)

            future_agents_traj = future_agents_traj.to(self.device)[three_mask]
            future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

            num_future_agents = num_future_agents.to(self.device)
            episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

            future_agent_masks = future_agent_masks.to(self.device)
            agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
            agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            future_agent_masks = future_agent_masks.to(self.device)

            decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

            log_prior = log_prior.to(self.device)

            gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

            gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)

            rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
            rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
            
            num_agents = gen_trajs.size(0)
            num_agents2 = rs_error2.size(0)
            num_agents3 = rs_error3.size(0)

            ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
            fde2 = rs_error2[..., -1]

            minade2, _ = ade2.min(dim=-1) # A X candi >> A
            avgade2 = ade2.mean(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            avgfde2 = fde2.mean(dim=-1)

            batch_minade2 = minade2.mean() # A >> 1
            batch_minfde2 = minfde2.mean()
            batch_avgade2 = avgade2.mean()
            batch_avgfde2 = avgfde2.mean()

            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]

            minade3, _ = ade3.min(dim=-1)
            avgade3 = ade3.mean(dim=-1)
            minfde3, _ = fde3.min(dim=-1)
            avgfde3 = fde3.mean(dim=-1)

            batch_minade3 = minade3.mean()
            batch_minfde3 = minfde3.mean()
            batch_avgade3 = avgade3.mean()
            batch_avgfde3 = avgfde3.mean()


            batch_loss = batch_minade3
            epoch_loss += batch_loss.item()
            batch_qloss = torch.zeros(1)
            batch_ploss = torch.zeros(1)

            # Loss backward
            batch_loss.backward()
            self.optimizer.step()

            print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
            "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
            "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2
            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += num_agents
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3


        epoch_loss /= epoch_agents

        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.
        
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        with torch.no_grad():
            if self.map_version == '2.0':
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
            
                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)
            
            c1 = -self.decoding_steps * np.log(2 * np.pi)
            for b, batch in enumerate(self.valid_loader):

                scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id, batch_size = batch

                # Detect dynamic batch size
                #batch_size = scene_images.size(0)
                num_three_agents = torch.sum(three_mask)


                
                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)[three_mask]
                future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

                num_future_agents = num_future_agents.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

                future_agent_masks = future_agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                log_prior = log_prior.to(self.device)


                gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
                num_agents = gen_trajs.size(0)
                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1)
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()


                batch_loss = batch_minade3
                epoch_loss += batch_loss.item()
                batch_qloss = torch.zeros(1)
                batch_ploss = torch.zeros(1)

                print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
                "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size
                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3
        


        epoch_loss /= epoch_agents

        # 2-Loss
        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2

        # 3-Loss
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
        epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

        scheduler_metric = epoch_avgade3 + epoch_avgfde3 

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric


    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde, qloss=0, ploss=0):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_ploss': ploss,
            'val_qloss': qloss,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, qloss, ploss, ade, fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)
