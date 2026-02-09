import os
import sys
from os.path import join as pjoin
import torch.multiprocessing as mp # don't chdir before we load this
import argparse
import re

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

# args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-r","--rank",dest="rank",action="store",default=0,type=int)
argParser.add_argument("--num_gpus_per_node",dest="n_gpus_per_node",action="store",default=1,type=int)
argParser.add_argument("--num_nodes",dest="n_nodes",action="store",default=1,type=int)

argParser.add_argument("--instance_data_path",type=str, help="path to save checkpoints and logs")
argParser.add_argument("--batch_size",type=int)
argParser.add_argument("--colmap_data_folders", type=str, nargs="+", default=[])
argParser.add_argument("--focal_lengths", type=float, nargs="+", default=[])
argParser.add_argument("--real_wl_focal_length", type=float, default=None)
argParser.add_argument("--real_wr_focal_length", type=float, default=None)
argParser.add_argument('--task', type=str, choices=['push', 'stack', 'pour', 'hang', 'umi', 'handover_item', 'coordinated_lift_ball', 'coordinated_put_item_in_drawer', 'coordinated_put_item_in_drawer_easy', 'dual_push_buttons', 'bimanual_straighten_rope', 'coordinated_lift_tray_easy5', 'coordinated_push_box_easy4', 'coordinated_push_box_easy2', 'coordinated_lift_tray_easy6', 'coordinated_push_box', 'lift_drawer'])
argParser.add_argument('--sfm_method', type=str, choices=['colmap', 'grabber_orbslam', 'umi', 'orbslam_bimanual'])
argParser.add_argument("--max_epoch",type=int, default=10000)
argParser.add_argument("--left_right_pose_cond", action="store_true", help="Whether to use pose conditioning for aTc and cTa in CrossAttnBlock")
argParser.add_argument("--gripper_state", action="store_true", help="Whether to include gripper state in the input")
argParser.add_argument("--external_gpu", action="store_true", help="Whether to use the external GPU to train")
argParser.add_argument("--stable_diffusion_encoder", action="store_true", help="Whether to use stable diffusion encoder rather than the VQGAN")
argParser.add_argument("--n_kept_checkpoints",type=int, default=5)
argParser.add_argument("--checkpoint_interval",type=int, default=500)
argParser.add_argument("--sample_param_lb",type=int, default=5)
argParser.add_argument("--sample_param_ub",type=int, default=15)
argParser.add_argument("--checkpoint_retention_interval",type=int, default=200)
argParser.add_argument("--wandb", action="store_true", help="wandb logging")


cli_args = argParser.parse_args()

from utils import *

globals.instance_data_path = cli_args.instance_data_path
globals.ckpt_path = os.path.join(cli_args.instance_data_path, "checkpoints")
globals.batch_size = cli_args.batch_size
globals.colmap_data_folders = cli_args.colmap_data_folders
globals.focal_lengths = cli_args.focal_lengths
globals.real_wl_focal_length = cli_args.real_wl_focal_length
globals.real_wr_focal_length = cli_args.real_wr_focal_length
globals.task = cli_args.task
globals.sfm_method = cli_args.sfm_method
globals.n_kept_checkpoints = cli_args.n_kept_checkpoints
globals.checkpoint_interval = cli_args.checkpoint_interval
globals.max_epoch = cli_args.max_epoch
globals.checkpoint_retention_interval = cli_args.checkpoint_retention_interval
globals.sample_param_lb = cli_args.sample_param_lb
globals.sample_param_ub = cli_args.sample_param_ub

import torch.distributed as dist
import utils.trainers
import socket
import signal
import psutil

def worker(local_rank,node_rank,n_gpus_per_node,n_nodes):
    # set multiprocessing to fork for dataloader workers (might avoid shared memory issues)
    try:
        mp.set_start_method('spawn') # fork
    except:
        pass

    # initialize distributed training
    rank = node_rank*n_gpus_per_node + local_rank
    world_size = n_nodes*n_gpus_per_node

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    trainer = utils.trainers.Score_sde_trainer(local_rank,node_rank,n_gpus_per_node,n_nodes,cli_args)

    # load checkpoint if it exists
    ckpt_dir = globals.ckpt_path
    checkpoints = os.listdir(ckpt_dir)
    checkpoints = [x for x in checkpoints if x.endswith('.pth')]
    checkpoints.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    if len(checkpoints) > 0:
        latest_checkpoint = pjoin(ckpt_dir,checkpoints[-1])
        if rank == 0: tlog(f'Resuming from {latest_checkpoint}','note')
        trainer.load_checkpoint(latest_checkpoint)

    # attach signal handler for graceful termination
    def termination_handler(sig_num,frame):
        trainer.termination_requested = True
    signal.signal(signal.SIGTERM,termination_handler)

    # do training
    try:
        trainer.train()
    except Exception as e:
        # print which worker failed
        tlog(f'Exception on {socket.gethostname()}, rank {rank}: {e}','error')
        raise e

if __name__ == '__main__':
    if cli_args.rank < 0:
        node_rank = int(os.environ['SLURM_PROCID'])
    else:
        node_rank = cli_args.rank
    tlog(f'{socket.gethostname()}: {node_rank}')
    if node_rank == 0:
        init_instance_dirs(['logs','checkpoints'])

    def main_exit_handler(sig_num,frame):
        parent = psutil.Process(os.getpid())
        children = parent.children()
        for child in children:
            child.send_signal(sig_num)
        tlog(f'SIGTERM received, waiting to children to finish','note')
    signal.signal(signal.SIGTERM,main_exit_handler)

    n_gpus_per_node = cli_args.n_gpus_per_node
    n_nodes = cli_args.n_nodes
    if cli_args.external_gpu:
        # make sure you don't specify CUDA_VISIBLE_DEVICES in the command line
        import torch
        print('CUDA device count: ', torch.cuda.device_count())
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
        worker(0, node_rank, n_gpus_per_node, n_nodes)
    else:
        mp.spawn(worker,nprocs=n_gpus_per_node,args=(node_rank,n_gpus_per_node,n_nodes))
