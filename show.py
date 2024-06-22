import torch
from torchvision.models import AlexNet
import netron
from model.CVAE import CVAE
from torch.utils.data import DataLoader
from utils.dataloader import TrajectoryDataset
from torch import nn

import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import TrajectoryDataset
from model.CVAE import CVAE
from model.vaeloss import compute_vae_loss
from utils.sddloader import SDD_Dataset

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter

# maybe need to close
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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

parser.add_argument('--he_tf_layer', type=int, default=2)  # he = history encoder #--he_tf_layer  3  --fe_tf_layer 3  --fd_tf_layer  3
parser.add_argument('--fe_tf_layer', type=int, default=2)  # fe = future encoder
parser.add_argument('--fd_tf_layer', type=int, default=2)  # fd = future decoder

# parser.add_argument('--cross_range', type=int, default=2)
# parser.add_argument('--num_conv_layer', type=int, default=7)

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

# loss config
parser.add_argument('--mse_weight', type=float, default=1.0)
parser.add_argument('--kld_weight', type=float, default=1.0)
parser.add_argument('--kld_min_clamp', type=float, default=2.0)
parser.add_argument('--var_weight', type=float, default=1.0)
parser.add_argument('--var_k', type=int, default=20)

# training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', type=str, default='step')

parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--lr_fix_epochs', type=int, default=10)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=100)

args = parser.parse_args()
data_set = './dataset/' + 'eth' + '/'

dset_train = TrajectoryDataset(
            data_set + 'train/',
            obs_len=8,
            pred_len=12,
            skip=1, traj_scale=1.0)

loader_train = DataLoader(
    dset_train,
    batch_size=1,
    shuffle=True,
    num_workers=0)
class CVAE_show(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cpu')
        self.cvae=CVAE(args)
        self.args = args
    # def set_device(self, device):
    #     self.device = device
    #     self.to(device)

    def forward(self, pre_motion, fut_motion, pre_motion_mask, fut_motion_mask):
        self.cvae.set_data(pre_motion, fut_motion, pre_motion_mask, fut_motion_mask)
        out = self.cvae.forward(self.args.sample_k).cuda()
        return out


device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')


model =  CVAE_show(args)
# model.set_device(device)
for cnt, batch in enumerate(loader_train):
    seq_name = batch.pop()[0]
    frame_idx = int(batch.pop()[0])
    batch = [tensor[0].cuda() for tensor in batch]
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
    non_linear_ped, valid_ped, obs_loss_mask, pred_loss_mask = batch



    input = obs_traj, pred_traj_gt, obs_loss_mask, pred_loss_mask

    torch.onnx.export(model, input,f='cvae.onnx')  # 导出 .onnx 文件
    netron.start('cvae.onnx')  # 展示结构图
    break