"""
MIT License

Copyright (c) 2023 Tony Z. Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from .transformer_for_diffusion import TransformerForDiffusion
from typing import List, Optional, Dict, Any, Tuple
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from collections import OrderedDict
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override: Dict[str, Any]):
        super().__init__()
        args_override_method = args_override['method']
        self.camera_names: List[str] = args_override_method['camera_names']
        self.observation_horizon: int = args_override_method['observation_horizon']
        # chunk size
        self.prediction_horizon: int = args_override_method['prediction_horizon']
        self.num_inference_timesteps: int = args_override_method['num_inference_timesteps']
        self.ema_power: float = args_override_method['ema_power']
        self.lr: float = args_override_method['lr']
        self._weight_decay: float = args_override_method.get('weight_decay', 1e-6)
        self._betas = args_override_method.get('betas', (0.9, 0.95))
        self.num_kp: int = 32
        self.feature_dimension: int = 64
        self.ac_dim: int = args_override_method['action_dim']  # 14 + 2
        self.obs_dim: int = self.feature_dimension * \
            len(self.camera_names) + self.ac_dim  # camera features and proprio

        backbones = []
        pools = []
        linears = []

        _W, _H = args_override_method['image_size']

        for _ in self.camera_names:
            conv = ResNet18Conv(
                **{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False})
            backbones.append(conv)
            pools.append(SpatialSoftmax(
                **{'input_shape': conv.output_shape((3, _H, _W)), 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(
                int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        backbones = replace_bn_with_gn(backbones)  # TODO

        model = args_override_method['model'].strip().lower()

        if model == 'cnn':

            noise_pred_net = ConditionalUnet1D(
                input_dim=self.ac_dim,
                global_cond_dim=self.obs_dim*self.observation_horizon,
                **args_override_method.get('cnn_kwargs')
            )

            self._transformer = False

        elif model == 'transformer':

            kwargs: Dict = args_override_method.get('transformer_kwargs')

            noise_pred_net = TransformerForDiffusion(
                input_dim=self.ac_dim,
                output_dim=self.ac_dim,
                cond_dim=self.obs_dim*self.observation_horizon,
                **kwargs
            )

            self._transformer = True

        else:
            raise NotImplementedError

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=args_override_method.get('num_train_timesteps', 100),
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

    def configure_optimizers(self):
        print(f"Weighting decay: {self._weight_decay}")
        if not self._transformer:
            optimizer = torch.optim.AdamW(
                self.nets.parameters(), lr=self.lr, weight_decay=self._weight_decay, betas=self._betas)
        else:
            noise_pred_net: TransformerForDiffusion = self.nets['policy']['noise_pred_net']
            assert isinstance(noise_pred_net, TransformerForDiffusion)

            decay_group, no_decay_group = noise_pred_net.get_optim_groups(
                self._weight_decay)

            no_decay_group['params'].extend(
                self.nets['policy']['backbones'].parameters())
            no_decay_group['params'].extend(
                self.nets['policy']['pools'].parameters())
            no_decay_group['params'].extend(
                self.nets['policy']['linears'].parameters())

            optimizer = torch.optim.AdamW(
                [
                    decay_group,
                    no_decay_group
                ],
                lr=self.lr,
                betas=self._betas,
            )

        return optimizer

    def __call__(self, qpos: torch.Tensor, image: torch.Tensor, actions: Optional[torch.Tensor] = None, is_pad: Optional[torch.Tensor] = None):
        # image: (B,#camera,c,h,w)
        B = qpos.shape[0]
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]  # (B,3,h,w)
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(
                    pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](
                    pool_features)  # (B,64)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)

            # noisy_actions: (B,T,ACTION_DIM)
            # timesteps: (B,)
            # global_cond: (B,OBS_DIM)

            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, obs_cond)

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['loss'] = loss
            loss_dict["total_losses"] = (loss)

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else:  # inference time
            To = self.observation_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    naction,
                    k,
                    obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(
                model_dict["ema"])
            status = [status, status_ema]
        return status