from utils import *
from os.path import join as pjoin
import time
import datetime
import torch
from torchvision import datasets, transforms
from datasets import *
from datasets.colmap_dataset import *
from torch.utils.data import DataLoader
import torch.optim as optim
from models.score_sde.ncsnpp_dual import NCSNpp_dual, NCSNpp_dual_bimanual
import models.score_sde.sde_lib as sde_lib
from .Base_trainer import Base_trainer
from models.score_sde.configs.LDM import LDM_config
import numpy as np
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.latentdiffusion.sb_32_4 import sb_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
import functools
from models.score_sde.layerspp import ResnetBlockBigGANpp
import torch.nn.functional as F
import wandb
from models.vqgan.vqgan import VQModel as VQGAN

def vq_to_img(x, vqgan):
    with torch.no_grad():
        decoded = vqgan.decode(x)
        intermediate_sample = (decoded/2+0.5)
        intermediate_sample = torch.clip(intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()
    return intermediate_sample

def loss_fn(score_sde, batch, cond_im, ff_ref, ff_a, ff_b):
    score_sde.train()
    eps=1e-5

    t = score_sde.t_uniform(batch.shape[0],batch.device)
    perturbed_data, z, std = score_sde.forward_diffusion(batch,t)
    score = score_sde.score(perturbed_data, cond_im, t, ff_ref, ff_a, ff_b)

    losses = torch.square(score * std[:, None, None, None] + z)
    losses = 0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)

    loss = torch.mean(losses)
    return loss

def loss_fn_bimanual(score_sde, w_l_batch, w_l_cond_im, w_r_batch, w_r_cond_im, w_l_ff_ref, w_l_ff_a, w_l_ff_b, w_r_ff_ref, w_r_ff_a, w_r_ff_b, gripper_state=None):
    score_sde.train()
    eps=1e-5

    w_l_t, w_r_t = score_sde.t_uniform(w_l_batch.shape[0], w_l_batch.device)
    w_l_perturbed_data, w_l_z, w_l_std, w_r_perturbed_data, w_r_z, w_r_std = score_sde.forward_diffusion(w_l_batch, w_l_t, w_r_batch, w_r_t)
    w_l_score, w_r_score = score_sde.score(w_l_perturbed_data, w_l_cond_im, w_l_t, w_l_ff_ref, w_l_ff_a, w_l_ff_b, w_r_perturbed_data, w_r_cond_im, w_r_t, w_r_ff_ref, w_r_ff_a, w_r_ff_b, gripper_state=gripper_state)

    w_l_losses = torch.square(w_l_score * w_l_std[:, None, None, None] + w_l_z)
    w_l_losses = 0.5*torch.sum(w_l_losses.reshape(w_l_losses.shape[0], -1), dim=-1)

    w_r_losses = torch.square(w_r_score * w_r_std[:, None, None, None] + w_r_z)
    w_r_losses = 0.5*torch.sum(w_r_losses.reshape(w_r_losses.shape[0], -1), dim=-1)

    losses = (0.5 * w_l_losses) + (0.5 * w_r_losses)

    loss = torch.mean(losses)
    return loss

class Score_sde_trainer(Base_trainer):
    def __init__(self,local_rank,node_rank,n_gpus_per_node,n_nodes,cli_args):
        # distributed helpers
        self.rank = node_rank*n_gpus_per_node + local_rank
        self.world_size = n_nodes*n_gpus_per_node
        self.gpu = local_rank
        self.wandb_logging = cli_args.wandb
        if self.wandb_logging and self.rank == 0:
            wandb.init(
                # set the wandb project where this run will be logged
                project="biaug",
                # track hyperparameters and run metadata
                name=cli_args.instance_data_path.split('/')[-1],
                config=cli_args.__dict__,
            )
        compute_only = self.rank != 0
        super().__init__(compute_only)

        # configure checkpoint behaviour
        self.n_kept_checkpoints = globals.n_kept_checkpoints
        self.checkpoint_interval =globals.checkpoint_interval
        self.max_epoch = globals.max_epoch
        self.checkpoint_retention_interval = globals.checkpoint_retention_interval

        # vqgan
        self.vqgan = None
        self.stable_diffusion_encoder = cli_args.stable_diffusion_encoder
        if not cli_args.stable_diffusion_encoder:
            self.vqgan = VQGAN(**vqgan_32_4_config).to(f'cuda:{self.gpu}')
        else:
            from ldm.models.autoencoder import VQModel as VQ_LDM
            self.vqgan = VQ_LDM(**sb_32_4_config).to(f'cuda:{self.gpu}')
        self.vqgan.eval()

        # score sde configs
        self.config = LDM_config()
        config = self.config
        config.left_right_pose_cond = cli_args.left_right_pose_cond
        config.gripper_state = cli_args.gripper_state

        # main components, dataloaders, model, optimizer
        batch_size = 64//self.world_size
        batch_size = globals.batch_size//self.world_size
        
        self.is_bimanual = False
        if globals.sfm_method == "colmap":
            self.dataset = ColmapDataset(globals.colmap_data_folders, globals.focal_lengths)
        elif globals.sfm_method == "umi":
            self.dataset = UmiDatasetFromFolder(globals.colmap_data_folders, globals.focal_lengths)
        elif globals.sfm_method == "orbslam_bimanual":
            self.dataset = SlamBimanualDataset(globals.colmap_data_folders, globals.focal_lengths, globals.sample_param_lb, globals.sample_param_ub, globals.real_wl_focal_length, globals.real_wr_focal_length)
            self.is_bimanual = True
        else:
            self.dataset = SlamDataset(globals.colmap_data_folders, globals.focal_lengths, globals.sample_param_lb, globals.sample_param_ub)
        
        self.sampler = DistributedSaveableSampler(self.dataset,num_replicas=self.world_size,rank=self.rank,shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,sampler=self.sampler,drop_last=True,num_workers=8,persistent_workers=True) # drop_last used to be True
        print("Data loader length: ", len(self.train_loader))

        if self.is_bimanual:
            score_model = NCSNpp_dual_bimanual(config)
            score_model = score_model.to(f'cuda:{self.gpu}')
            score_model = torch.nn.parallel.DistributedDataParallel(score_model,device_ids=[self.gpu])
            sde_bimanul = sde_lib.VESDE_Bimanual(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                                                            act=torch.nn.SiLU(),
                                                                            dropout=False,
                                                                            fir=True,
                                                                            fir_kernel=[1,3,3,1],
                                                                            init_scale=0,
                                                                            skip_rescale=True,
                                                                            temb_dim=None)
            ray_downsampler = torch.nn.Sequential(
                    ResnetBlock(in_ch=56,out_ch=128,down=True).to(f'cuda:{self.gpu}'),
                    ResnetBlock(in_ch=128,out_ch=128,down=True).to(f'cuda:{self.gpu}'))
            ray_downsampler = torch.nn.parallel.DistributedDataParallel(ray_downsampler,device_ids=[self.gpu])

            self.score_sde = Score_sde_bimanual_model(score_model, sde_bimanul, ray_downsampler)
        else:
            score_model = NCSNpp_dual(config)
            score_model = score_model.to(f'cuda:{self.gpu}')
            score_model = torch.nn.parallel.DistributedDataParallel(score_model,device_ids=[self.gpu])
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

            ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                                                            act=torch.nn.SiLU(),
                                                                            dropout=False,
                                                                            fir=True,
                                                                            fir_kernel=[1,3,3,1],
                                                                            init_scale=0,
                                                                            skip_rescale=True,
                                                                            temb_dim=None)
            ray_downsampler = torch.nn.Sequential(
                    ResnetBlock(in_ch=56,out_ch=128,down=True).to(f'cuda:{self.gpu}'),
                    ResnetBlock(in_ch=128,out_ch=128,down=True).to(f'cuda:{self.gpu}'))
            ray_downsampler = torch.nn.parallel.DistributedDataParallel(ray_downsampler,device_ids=[self.gpu])

            self.score_sde = Score_sde_model(score_model,sde,ray_downsampler)

        # only 1 worker needs to keep ema
        if self.rank == 0:
                self.ema = ExponentialMovingAverage(self.score_sde.parameters(),decay=0.999)

        self.lr = 2e-4
        self.optimizer = optim.Adam(self.score_sde.parameters(), lr=self.lr)
        self.warmup = 5000
        self.grad_clip = 1.

        self.tlog('Setup complete','note')

    def train_epoch(self):
        iteration_start = time.time()
        self.sampler.set_epoch(self.epoch) # important! or else split will be the same every epoch
        dataloader_iter = iter(self.train_loader) # need this to save state
        
        # print(len(list(enumerate(dataloader_iter, start=1))))
        for epoch_it, batch_data in enumerate(dataloader_iter,start=1):
            # print('at train epoch rn')
            self.total_iterations += 1

            if self.is_bimanual:
                ###### unpack data ######
                w_l_im_a = batch_data['w_l_im_a']
                w_l_im_b = batch_data['w_l_im_b']
                w_l_camera_enc_ref = batch_data['w_l_camera_enc_ref']
                w_l_camera_enc_a = batch_data['w_l_camera_enc_a']
                w_l_camera_enc_b = batch_data['w_l_camera_enc_b']

                w_r_im_a = batch_data['w_r_im_a']
                w_r_im_b = batch_data['w_r_im_b']
                w_r_camera_enc_ref = batch_data['w_r_camera_enc_ref']
                w_r_camera_enc_a = batch_data['w_r_camera_enc_a']
                w_r_camera_enc_b = batch_data['w_r_camera_enc_b']

                ###### move data to gpu ######
                w_l_im_a = w_l_im_a.to(f'cuda:{self.gpu}')
                w_l_im_b = w_l_im_b.to(f'cuda:{self.gpu}')
                w_l_camera_enc_ref = w_l_camera_enc_ref.to(f'cuda:{self.gpu}')
                w_l_camera_enc_a = w_l_camera_enc_a.to(f'cuda:{self.gpu}')
                w_l_camera_enc_b = w_l_camera_enc_b.to(f'cuda:{self.gpu}')

                w_r_im_a = w_r_im_a.to(f'cuda:{self.gpu}')
                w_r_im_b = w_r_im_b.to(f'cuda:{self.gpu}')
                w_r_camera_enc_ref = w_r_camera_enc_ref.to(f'cuda:{self.gpu}')
                w_r_camera_enc_a = w_r_camera_enc_a.to(f'cuda:{self.gpu}')
                w_r_camera_enc_b = w_r_camera_enc_b.to(f'cuda:{self.gpu}')

                # encode with vqgan
                with torch.no_grad():
                    w_l_encoded_a = None
                    w_l_encoded_b = None
                    w_r_encoded_a = None
                    w_r_encoded_b = None

                    if not self.stable_diffusion_encoder:
                        w_l_encoded_a = self.vqgan.encode(w_l_im_a)
                        w_l_encoded_b = self.vqgan.encode(w_l_im_b)
                        w_r_encoded_a = self.vqgan.encode(w_r_im_a)
                        w_r_encoded_b = self.vqgan.encode(w_r_im_b)
                    else:
                        w_l_encoded_a = self.vqgan.encode(w_l_im_a)[0]
                        w_l_encoded_b = self.vqgan.encode(w_l_im_b)[0]
                        w_r_encoded_a = self.vqgan.encode(w_r_im_a)[0]
                        w_r_encoded_b = self.vqgan.encode(w_r_im_b)[0]

                # train
                self.optimizer.zero_grad()
                # NOTE: freq_enc: rays might undergo frequency-based encoding (e.g., sinusoidal encoding) to better 
                # capture high-frequency details along the ray, which helps models represent complex geometry.
                w_l_ff_ref = F.pad(freq_enc(w_l_camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
                w_l_ff_a = F.pad(freq_enc(w_l_camera_enc_a),[0,0,0,0,1,1,0,0])
                w_l_ff_b = F.pad(freq_enc(w_l_camera_enc_b),[0,0,0,0,1,1,0,0])

                w_r_ff_ref = F.pad(freq_enc(w_r_camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
                w_r_ff_a = F.pad(freq_enc(w_r_camera_enc_a),[0,0,0,0,1,1,0,0])
                w_r_ff_b = F.pad(freq_enc(w_r_camera_enc_b),[0,0,0,0,1,1,0,0])

                gripper_state = batch_data['gripper_state']
                loss = loss_fn_bimanual(self.score_sde, w_l_encoded_a, w_l_encoded_b, w_r_encoded_a, w_r_encoded_b, w_l_ff_ref, w_l_ff_a, w_l_ff_b, w_r_ff_ref, w_r_ff_a, w_r_ff_b, gripper_state=gripper_state)
            else:
                # original implementation
                # unpack data
                im_a = batch_data['im_a']
                im_b = batch_data['im_b']
                camera_enc_ref = batch_data['camera_enc_ref']
                camera_enc_a = batch_data['camera_enc_a']
                camera_enc_b = batch_data['camera_enc_b']

                # move data to gpu
                im_a = im_a.to(f'cuda:{self.gpu}')
                im_b = im_b.to(f'cuda:{self.gpu}')
                camera_enc_ref = camera_enc_ref.to(f'cuda:{self.gpu}')
                camera_enc_a = camera_enc_a.to(f'cuda:{self.gpu}')
                camera_enc_b = camera_enc_b.to(f'cuda:{self.gpu}')

                # encode with vqgan
                with torch.no_grad():
                    encoded_a = None
                    encoded_b = None

                    if not self.stable_diffusion_encoder:
                        encoded_a = self.vqgan.encode(im_a)
                        encoded_b = self.vqgan.encode(im_b)
                    else:
                        encoded_a = self.vqgan.encode(im_a)[0]
                        encoded_b = self.vqgan.encode(im_b)[0]

                # train
                self.optimizer.zero_grad()
                # NOTE: freq_enc: rays might undergo frequency-based encoding (e.g., sinusoidal encoding) to better 
                # capture high-frequency details along the ray, which helps models represent complex geometry.
                ff_ref = F.pad(freq_enc(camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
                ff_a = F.pad(freq_enc(camera_enc_a),[0,0,0,0,1,1,0,0])
                ff_b = F.pad(freq_enc(camera_enc_b),[0,0,0,0,1,1,0,0])
                loss = loss_fn(self.score_sde, encoded_a, encoded_b, ff_ref, ff_a, ff_b)
            loss.backward()

            # warmup and gradient clip
            if self.warmup > 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr * np.minimum(self.total_iterations / self.warmup, 1.0)
            if self.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(self.score_sde.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            # update ema
            if self.rank == 0:
                self.ema.update(self.score_sde.parameters())

            # print iteration details
            if not self.compute_only:
                # calc eta
                iteration_duration = time.time() - iteration_start
                its_per_sec = 1/iteration_duration
                remaining_its = self.max_epoch*len(self.train_loader) - self.total_iterations
                eta_sec = remaining_its * iteration_duration
                eta_min = eta_sec//60
                eta = str(datetime.timedelta(minutes=eta_min))
                if self.total_iterations % self.print_iteration == 0:
                    self.tlog(f'{self.total_iterations} | loss: {loss.item()} | it/s: {its_per_sec} | ETA: {eta}','iter')

                self.tb_writer.add_scalar('training/loss', loss.item(), self.total_iterations)
                if self.wandb_logging and self.rank == 0:
                    wandb.log({'loss': loss.item(), 'epoch': self.epoch, 'iteration': self.total_iterations})

            # checkpoint if haven't checkpointed in a while
            self.maybe_save_checkpoint(epoch_it,dataloader_iter)

            # check if termination requested
            self.check_termination_request(epoch_it,dataloader_iter)

            # start time here to include data fetching
            iteration_start = time.time()


            if not self.compute_only:
                self.tb_writer.add_scalar('training/loss epoch', loss.item(), epoch_it)
                if self.wandb_logging and self.rank == 0:
                    wandb.log({'loss epoch': loss.item(), 'epoch_it': epoch_it, 'epoch': self.epoch, 'iteration': self.total_iterations})
    
    def validate(self):
        pass

    def state_dict(self,dataloader_iter=None):
        state_dict = {
            'epoch': self.epoch,
            'total_iterations': self.total_iterations,
            'optimizer': self.optimizer.state_dict(),
            'score_sde_model':self.score_sde.state_dict(),
            'training_data_sampler':self.sampler.state_dict(dataloader_iter),
            'ema': self.ema.state_dict()
        }
        return state_dict

    def load_state_dict(self,state_dict):
        self.epoch = state_dict['epoch']
        self.total_iterations = state_dict['total_iterations']
        self.score_sde.load_state_dict(state_dict['score_sde_model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.sampler.load_state_dict(state_dict['training_data_sampler'])
        if self.rank == 0:
            self.ema.load_state_dict(state_dict['ema'])

    def load_checkpoint(self,checkpoint_path):
        super().load_checkpoint(checkpoint_path,map_location=f'cuda:{self.gpu}')
