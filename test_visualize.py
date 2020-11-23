import torch
import os
import argparse
import torch
from torch.utils.data import DataLoader
from Proposed.models import MultiAgentTrajectory
from dataset.nuscenes import NuscenesDataset, nuscenes_collate
import sys
import time
import numpy as np
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import cv2
import pdb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import logging
from multiprocessing import Pool


scene_channels = 3
sampling_rate = 2
nfuture = int(3 * sampling_rate)
velocity_const = 0.5
agent_embed_dim = 128
#num_candidates = 6
num_candidates = 1
att_dropout = 0.1
crossmodal_attention = False
use_scene = True
scene_size = (64, 64)
ploss_type = 'map'
test_partition = 'val'
map_version = '2.0'
sample_stride = 1
multi_agent = 1
num_workers = 20
test_cache = "./caches/nusc_val_cache.pkl"
batch_size = 64
beta = 0.1
decoding_steps = int(3 *  sampling_rate)
out_dir = "./test"
test_ckpt = "./experiment/Trajectory__20_November__05_17_/ck_95_0.0000_0.0000_1.1190_2.2991.pth.tar"

test_render = 1
test_times = 1
render = test_render

_data_dir = './data/nuscenes'
map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x in scene_id]

checkpoint = torch.load(test_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



dataset = NuscenesDataset(test_partition, map_version=map_version, sampling_rate=sampling_rate, sample_stride=sample_stride, use_scene=use_scene, scene_size=scene_size,
    ploss_type=ploss_type, num_workers=num_workers, cache_file=test_cache, multi_agent=multi_agent)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         pin_memory=True,collate_fn=lambda x: nuscenes_collate(x),

                         num_workers=1)


print(f'Test Examples: {len(dataset)}')

model = MultiAgentTrajectory(device=device, embedding_dim = agent_embed_dim, nfuture=nfuture, att_dropout=att_dropout)
model = model.to(device)
model.load_state_dict(checkpoint['model_state'], strict=False)

def dac(gen_trajs, map_file):
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    da_mask = np.any(map_array > 0, axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dac = []

    gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

    stay_in_da_count = [0 for i in range(num_agents)]
    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        stay_in_da = [True for i in range(num_agents)]

        oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
        diregard_mask = oom_mask.sum(axis=-1) > 2
        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x*224+y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
                    stay_in_da[i] = False

        for i in range(num_agents):
            if stay_in_da[i]:
                stay_in_da_count[i] += 1

    for i in range(num_agents):
        if diregard_mask[i]:
            dac.append(0.0)
        else:
            dac.append(stay_in_da_count[i] / num_candidates)

    dac_mask = np.logical_not(diregard_mask)

    return np.array(dac), dac_mask


def dao(gen_trajs, map_file):
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    da_mask = np.any(map_array > 0, axis=-1)

    num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
    dao = [0 for i in range(num_agents)]

    occupied = [[] for i in range(num_agents)]

    gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

    for k in range(num_candidates):
        gen_trajs_k = gen_trajs[:, k]

        oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
        diregard_mask = oom_mask.sum(axis=-1) > 2

        for t in range(decoding_timesteps):
            gen_trajs_kt = gen_trajs_k[:, t]
            oom_mask_t = oom_mask[:, t]
            x, y = gen_trajs_kt.T

            lin_xy = (x*224+y)
            lin_xy[oom_mask_t] = -1
            for i in range(num_agents):
                xi, yi = x[i], y[i]
                _lin_xy = lin_xy.tolist()
                lin_xyi = _lin_xy.pop(i)

                if diregard_mask[i]:
                    continue

                if oom_mask_t[i]:
                    continue

                if lin_xyi in occupied[i]:
                    continue

                if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
                    occupied[i].append(lin_xyi)
                    dao[i] += 1

    for i in range(num_agents):
        if diregard_mask[i]:
            dao[i] = 0.0
        else:
            dao[i] /= da_mask.sum()

    dao_mask = np.logical_not(diregard_mask)

    return np.array(dao), dao_mask


def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
    if '.png' in map_file:
        map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
        map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

    elif '.pkl' in map_file:
        with open(map_file, 'rb') as pnt:
            map_array = pkl.load(pnt)

    def plot_candidate(title, ax, can_idx):
        ax.set_title(title)
        ax.imshow(map_array, extent=[-56, 56, 56, -56])
        ax.set_aspect('equal')
        ax.set_xlim([-56, 56])
        ax.set_ylim([-56, 56])

        # num_tgt_agents, num_candidates = gen_trajs.shape[:2]
        num_tgt_agents, _ = gen_trajs.shape[:2]
        num_candidates = 1
        num_src_agents = len(src_trajs)

        gen_trajs_k = gen_trajs[:, can_idx]

        x_pts_k, y_pts_k = [], []
        for i in range(num_tgt_agents):
            gen_traj_ki = gen_trajs_k[i]
            tgt_len_i = tgt_lens[i]
            x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
            y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])
            ax.plot(gen_traj_ki[:tgt_len_i, 0], gen_traj_ki[:tgt_len_i, 1], c='g', linewidth=3.5)
        """
        x_pts, y_pts = [], []
        for i in range(num_src_agents):
            src_traj_i = src_trajs[i]
            src_len_i = src_lens[i]
            x_pts.extend(src_traj_i[:src_len_i, 0])
            y_pts.extend(src_traj_i[:src_len_i, 1])
            ax.plot(src_traj_i[:src_len_i, 0], src_traj_i[:src_len_i, 1], alpha=0.3, c='r', linewidth=3.5)
        """
        x_pts, y_pts = [], []
        for i in range(num_tgt_agents):
            tgt_traj_i = tgt_trajs[i]
            tgt_len_i = tgt_lens[i]
            x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
            y_pts.extend(tgt_traj_i[:tgt_len_i, 1])
            ax.plot(tgt_traj_i[:tgt_len_i, 0], tgt_traj_i[:tgt_len_i, 1], alpha=0.3, c='r', linewidth=3.5)

        ax.plot([], [], c='r', alpha=0.3, label='ground-truth')
        #ax.plot([], [], c='r', alpha=0.3, label='history')
        ax.plot([], [], c='g', label='estimated')
        ax.legend()

    H, W = map_array.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))


    for a in range(1):
        for b in range(1):
            num = a + b
            plot_candidate('Inference {}'.format(num + 1), ax, num)

    #     plt.show()

    global img_save_count

    fig.savefig('./test/results/scene_{}.jpg'.format(img_save_count), bbox_inches='tight', pad_inches=0.5, dpi=150)
    img_save_count += 1


def run_test():
    print('Starting model test.....')
    model.eval()  # Set model to evaluate mode.

    list_loss = []
    list_qloss = []
    list_ploss = []
    list_minade2, list_avgade2 = [], []
    list_minfde2, list_avgfde2 = [], []
    list_minade3, list_avgade3 = [], []
    list_minfde3, list_avgfde3 = [], []
    list_minmsd, list_avgmsd = [], []

    list_dao = []
    list_dac = []

    for test_time_ in range(test_times):

        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_minmsd, epoch_avgmsd = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        epoch_dao = 0.0
        epoch_dac = 0.0
        dao_agents = 0.0
        dac_agents = 0.0

        H = W = 64
        with torch.no_grad():
            if map_version == '2.0':
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))

                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std

                distance_2d = coordinate_2d - np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))

                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std

                coordinate = coordinate.to(device)
                distance = distance.to(device)

            c1 = -decoding_steps * np.log(2 * np.pi)

            for b, batch in enumerate(data_loader):

                scene_images, log_prior, \
                agent_masks, \
                num_src_trajs, src_trajs, src_lens, src_len_idx, \
                num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                tgt_two_mask, tgt_three_mask, \
                decode_start_vel, decode_start_pos, scene_id, batch_size = batch

                # Detect dynamic batch size

                num_three_agents = torch.sum(tgt_three_mask)
                """
                if map_version == '2.0':
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(device), coordinate_batch, distance_batch), dim=1)
                """
                src_trajs = src_trajs.to(device)
                src_lens = src_lens.to(device)

                tgt_trajs = tgt_trajs.to(device)[tgt_three_mask]
                tgt_lens = tgt_lens.to(device)[tgt_three_mask]

                num_tgt_trajs = num_tgt_trajs.to(device)
                episode_idx = torch.arange(batch_size, device=device).repeat_interleave(num_tgt_trajs)[tgt_three_mask]

                agent_masks = agent_masks.to(device)
                agent_tgt_three_mask = torch.zeros_like(agent_masks)
                agent_masks_idx = torch.arange(len(agent_masks), device=device)[agent_masks][tgt_three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(device)[agent_tgt_three_mask]

                log_prior = log_prior.to(device)

                gen_trajs = model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                gen_trajs = gen_trajs.reshape(num_three_agents, num_candidates, decoding_steps, 2)


                rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(decoding_steps * 2 / 3)]

                diff = gen_trajs - tgt_trajs.unsqueeze(1)
                msd_error = (diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2)

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

                msd = msd_error.mean(-1)
                minmsd, _ = msd.min(dim=-1)
                avgmsd = msd.mean(dim=-1)
                batch_minmsd = minmsd.mean()
                batch_avgmsd = avgmsd.mean()

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

                print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_time_ + 1, test_times, b + 1,
                                                                              len(data_loader)), end='\r')  # +

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

                epoch_minmsd += batch_minmsd.item() * num_agents3
                epoch_avgmsd += batch_avgmsd.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3

                map_files = map_file(scene_id)
                output_files = [out_dir + '/' + x[2] + '_' + x[3] + '.jpg' for x in scene_id]

                cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
                cum_num_src_trajs = [0] + torch.cumsum(num_src_trajs, dim=0).tolist()

                src_trajs = src_trajs.cpu().numpy()
                src_lens = src_lens.cpu().numpy()

                tgt_trajs = tgt_trajs.cpu().numpy()
                tgt_lens = tgt_lens.cpu().numpy()

                zero_ind = np.nonzero(tgt_three_mask.numpy() == 0)[0]
                zero_ind -= np.arange(len(zero_ind))

                tgt_three_mask = tgt_three_mask.numpy()
                agent_tgt_three_mask = agent_tgt_three_mask.cpu().numpy()

                gen_trajs = gen_trajs.cpu().numpy()

                src_mask = agent_tgt_three_mask

                gen_trajs = np.insert(gen_trajs, zero_ind, 0, axis=0)

                tgt_trajs = np.insert(tgt_trajs, zero_ind, 0, axis=0)
                tgt_lens = np.insert(tgt_lens, zero_ind, 0, axis=0)

                for i in range(1):
                    candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                    tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                    tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]

                    src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]
                    src_lens_i = src_lens[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]
                    map_file_i = map_files[i]
                    output_file_i = output_files[i]

                    candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                    tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                    tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]

                    src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]
                    src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]

                    dao_i, dao_mask_i = dao(candidate_i, map_file_i)
                    dac_i, dac_mask_i = dac(candidate_i, map_file_i)

                    epoch_dao += dao_i.sum()
                    dao_agents += dao_mask_i.sum()

                    epoch_dac += dac_i.sum()
                    dac_agents += dac_mask_i.sum()

                    write_img_output(candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i,
                                     'test/img')
            print(1)



        list_loss.append(epoch_loss / epoch_agents)

        # 2-Loss
        list_minade2.append(epoch_minade2 / epoch_agents2)
        list_avgade2.append(epoch_avgade2 / epoch_agents2)
        list_minfde2.append(epoch_minfde2 / epoch_agents2)
        list_avgfde2.append(epoch_avgfde2 / epoch_agents2)

        # 3-Loss
        list_minade3.append(epoch_minade3 / epoch_agents3)
        list_avgade3.append(epoch_avgade3 / epoch_agents3)
        list_minfde3.append(epoch_minfde3 / epoch_agents3)
        list_avgfde3.append(epoch_avgfde3 / epoch_agents3)

        list_minmsd.append(epoch_minmsd / epoch_agents3)
        list_avgmsd.append(epoch_avgmsd / epoch_agents3)

        list_dao.append(epoch_dao / dao_agents)
        list_dac.append(epoch_dac / dac_agents)


    test_ploss = [0.0, 0.0]
    test_qloss = [0.0, 0.0]
    test_loss = [np.mean(list_loss), np.std(list_loss)]

    test_minade2 = [np.mean(list_minade2), np.std(list_minade2)]
    test_avgade2 = [np.mean(list_avgade2), np.std(list_avgade2)]
    test_minfde2 = [np.mean(list_minfde2), np.std(list_minfde2)]
    test_avgfde2 = [np.mean(list_avgfde2), np.std(list_avgfde2)]

    test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
    test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
    test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
    test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]

    test_minmsd = [np.mean(list_minmsd), np.std(list_minmsd)]
    test_avgmsd = [np.mean(list_avgmsd), np.std(list_avgmsd)]

    test_dao = [np.mean(list_dao), np.std(list_dao)]
    test_dac = [np.mean(list_dac), np.std(list_dac)]

    test_ades = (test_minade2, test_avgade2, test_minade3, test_avgade3)
    test_fdes = (test_minfde2, test_avgfde2, test_minfde3, test_avgfde3)

    print("--Final Performane Report--")
    print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(test_minade3[0], test_minade3[1], test_minfde3[0],
                                                                  test_minfde3[1]))
    print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(test_avgade3[0], test_avgade3[1], test_avgfde3[0],
                                                                  test_avgfde3[1]))
    print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0],
                                                          test_dac[1]))
    with open(out_dir + '/metric.pkl', 'wb') as f:
        pkl.dump({"ADEs": test_ades,
                  "FDEs": test_fdes,
                  "Qloss": test_qloss,
                  "Ploss": test_ploss,
                  "DAO": test_dao,
                  "DAC": test_dac}, f)






if __name__ == "__main__":
    img_save_count = 0

    run_test()