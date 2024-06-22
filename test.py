import time

import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataloader import TrajectoryDataset
from model.CVAE import CVAE
from model.sampler import Sampler
from utils.metrics import compute_ADE, compute_FDE, count_miss_samples
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter


parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--sdd_scale', type=float, default=50.0)

# model architecture
parser.add_argument('--pos_concat', type=bool, default=True)
parser.add_argument('--cross_motion_only', type=bool, default=True)

parser.add_argument('--tf_model_dim', type=int, default=256)
parser.add_argument('--tf_ff_dim', type=int, default=512)
parser.add_argument('--tf_nhead', type=int, default=8)
parser.add_argument('--tf_dropout', type=float, default=0.1)

parser.add_argument('--he_tf_layer', type=int, default=2)  # he = history encoder
parser.add_argument('--fe_tf_layer', type=int, default=2)  # fe = future encoder
parser.add_argument('--fd_tf_layer', type=int, default=2)  # fd = future decoder

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=3)
parser.add_argument('--asconv_layer_num', type=int, default=3)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--sample_k', type=int, default=20)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)

# sampler architecture
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

# testing options
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--sample_num', type=int, default=20)

parser.add_argument('--sampler_epoch', type=int, default=200)
parser.add_argument('--vae_epoch', type=int, default=80)


def test(cvae, sampler, loader_test, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()

    # total_cnt = 0
    # miss_cnt = 0

    for cnt, batch in enumerate(loader_test):
        seq_name = batch.pop()[0]
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
        non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch

        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame_idx))
        sys.stdout.flush()

        with torch.no_grad():
            cvae.set_data(obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask)
            dec_motion, _, _, attn_weights = sampler.forward(cvae)  # [N sn T 2]  # testing function

        dec_motion = dec_motion * traj_scale
        traj_gt = pred_traj_gt.transpose(1, 2) * traj_scale  # [N 2 T] -> [N T 2]

        # rearrange dec_motion
        agent_traj = []
        sample_motion = dec_motion.detach().cpu().numpy()  # [7 20 12 2]
        for i in range(sample_motion.shape[0]):  # traverse each person  list -> ped dimension
            agent_traj.append(sample_motion[i, :, :, :])
        traj_gt = traj_gt.detach().cpu().numpy()

        # calculate ade and fde and get the min value for 20 samples
        ade = compute_ADE(agent_traj, traj_gt)
        ade_meter.update(ade, n=cvae.agent_num)

        fde = compute_FDE(agent_traj, traj_gt)
        fde_meter.update(fde, n=cvae.agent_num)

        # miss_sample_num = count_miss_samples(agent_traj, traj_gt)
        # total_cnt += sample_motion.shape[0]
        # miss_cnt += miss_sample_num

    # miss_rate = float(miss_cnt) / float(total_cnt)

    return ade_meter, fde_meter  # , miss_rate


def main(args):

    data_set = './dataset/' + args.dataset + '/'

    prepare_seed(args.seed)

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    traj_scale = 1.0
    if args.dataset == 'eth':
        args.max_train_agent = 32

    if args.dataset == 'sdd':
        traj_scale = args.sdd_scale
        dset_test = SDD_Dataset(
            data_set + 'test/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)
    else:
        dset_test = TrajectoryDataset(
            data_set + 'test/',
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1, traj_scale=traj_scale)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=0)

    cvae = CVAE(args)
    sampler = Sampler(args)
    # load cvae model
    vae_dir = './checkpoints/' + args.dataset + '/vae/'
    all_vae_models = os.listdir(vae_dir)
    if len(all_vae_models) == 0:
        print('VAE model not found!')
        return

    sampler_dir = './checkpoints/' + args.dataset + '/sampler/'
    all_sampler_models = os.listdir(sampler_dir)
    if len(all_sampler_models) == 0:
        print('sampler model not found!')
        return

    minade = 100000
    SAMPLER_EPOCH = 0
    VAE_EPOCH = 0
    minfde =0




    for y in all_sampler_models[-1:]:

        if args.dataset == 'hotel':
            y = 'model_%04d.p' % 27#  80:44  22 70: 42   60:40  #90:27
        elif args.dataset == 'eth':
            y = 'model_%04d.p' % 10  #vae 80  22  28  14  #100vae 24,26,28  # 80vae :31 #70:10
        elif args.dataset == 'zara1':
            y = 'model_%04d.p' % 29 # 46# 58 36   50 # 46 56 36 50 #120:36  40  #80 :29   40 38
        elif args.dataset == 'zara2':
            y = 'model_%04d.p' %28  # 42  48 32  # 60  36  # 60vae 38 36
        elif args.dataset == 'univ':
            y = y#'model_%04d.p' % 24  # zara 60  #hotel 75 45  #135 :32
        else:
            y = 'model_%04d.p' % 102 #y 100
    # if default_sampler_model not in all_sampler_models:
    #     default_sampler_model = all_sampler_models[-1]
    # load sampler model
        sampler_path = os.path.join(sampler_dir, y)
        model_cp = torch.load(sampler_path, map_location='cpu')
        sampler.load_state_dict(model_cp)
        # torch.save(model_cp['model_dict'], cp_path)
        print('loading model from checkpoint: %s' % sampler_path)



        sampler.set_device(device)
        sampler.eval()
        for x in all_vae_models[9:]:
            if args.dataset =='hotel':
                x = x#'model_%04d.p' % 102#    45  # 90#65     #   102  #90:101
            elif args.dataset == 'eth':
                x = x#'model_%04d.p' % 106 # 45 100 106
            elif args.dataset == 'zara1':
                x = x#'model_%04d.p' % 147 # zara 60  #hotel 75 55   #141    115   133
            elif args.dataset == 'zara2':
                x = x#'model_%04d.p' % 80  # zara 60  #hotel 75
            elif args.dataset == 'univ':
                x = 'model_%04d.p' % 60 #72  60  45
            else:
                x = x#'model_%04d.p' % 100  # sdd 100
        # if default_vae_model not in all_vae_models:
        #     x = all_vae_models[-1]
            vae_path = os.path.join(vae_dir, x)
            print('loading model from checkpoint: %s' % vae_path)
            model_cp = torch.load(vae_path, map_location='cpu')
            # torch.save(model_cp['model_dict'], cp_path)
            cvae.load_state_dict(model_cp)

            cvae.set_device(device)
            cvae.eval()

            # run testing
            ade_meter, fde_meter = test(cvae, sampler, loader_test, traj_scale)

            print('-' * 20 + ' STATS ' + '-' * 20)
            print('ADE: %.4f' % ade_meter.avg)
            print('FDE: %.4f' % fde_meter.avg)
            #记录最小的ade以及fde

            if ade_meter.avg <minade:
                minade = ade_meter.avg
                minfde = fde_meter.avg
                print('minADE: %.4f' % minade)
                SAMPLER_EPOCH = sampler_path
                VAE_EPOCH =  vae_path

    print('-' * 30 + ' STATS ' + '-' * 30)
    print('result:minADE: %.4f' % minade)
    print('result:minFDE: %.4f' % minfde)
    print('result:SAMPLER_EPOCH:' +str(SAMPLER_EPOCH))
    print('result:SAMPLER_EPOCH:' +str(VAE_EPOCH))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

