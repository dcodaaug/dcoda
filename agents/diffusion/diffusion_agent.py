from collections import defaultdict
from torch.nn.modules import Module
from yarr.agents.agent import Agent, ActResult, Summary, ScalarSummary
from functools import lru_cache
import torch
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Type, Union
import abc
import logging
import matplotlib.pyplot as plt
import pickle

import h5py

from torchvision import transforms
from .diffusion_policy import DiffusionPolicy

NAME = "DiffusionAgent"

class DiffusionAgent(Agent):
    def __init__(self, config: Dict) -> None:
        self.policy = DiffusionPolicy(config)
        self.device = None
        self.training = None
        self.optimizer = None
        self._summaries: Dict[str, Any] = {}

        self._camera_names: List[str] = config['method'].get("camera_names")
        self._num_queries: int = config['method'].get("chunk_size")
        self._action_dim: int = config['method'].get("action_dim")

        self.temporal_agg: bool = config['method'].get("temporal_agg", False)
        if self.temporal_agg:
            logging.warning(
                f"temporal_agg=True in DiffusionAgent. In principle this is compatible, but Mobile ALOHA doesn't do this.")
        # just large enough to cover the entire episode
        self._max_timesteps: int = config['method'].get("max_timesteps", 400)
        self._shceduler_config = config['method'].get("scheduler", None)
        self.train_demo_path = config['method'].get("train_demo_path")
        self.task_name = config['rlbench'].get("tasks")[0]

    def _make_scheduler(self):
        if self._shceduler_config is None:
            self._scheduler = None
        else:
            kwargs = self._shceduler_config.get("kwargs")
            kwargs = {k: v for k, v in kwargs.items() if v not in ["name"]}
            if self._shceduler_config.name == "cosine":
                self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer, **kwargs)
            else:
                raise NotImplementedError

    def build(self, training: bool, device: torch.device = None):
        self.device = device
        self.training = training
        self.policy = self.policy.to(device)
        if training:
            self.policy.train()
            self.optimizer = self.policy.configure_optimizers()
            self._update_count = 0
            self._make_scheduler()
        else:
            self.policy.eval()
            for p in self.policy.parameters():
                p.requires_grad = False
            self._timestep = 0

            if self.temporal_agg:
                self._all_time_actions = torch.zeros(
                    [self._max_timesteps, self._max_timesteps+self._num_queries, self._action_dim]).to(device)

            self._query_frequency = 1 if self.temporal_agg else self._num_queries

    def reset(self) -> None:
        super(DiffusionAgent, self).reset()
        if not self.training:
            self._timestep = 0
            if self.temporal_agg:
                self._all_time_actions.fill_(0)
        else:
            self.optimizer = self.policy.configure_optimizers()
            self._update_count = 0
            self._make_scheduler()

    @lru_cache()
    def train_stats(self):
        right_joint_positions = []
        left_joint_positions = []

        right_gripper_positions = []
        left_gripper_positions = []

        episodes_dir = (
            f"{self.train_demo_path}/{self.task_name}/all_variations/episodes/"
        )

        for episode in os.listdir(episodes_dir):
            with open(
                os.path.join(episodes_dir, episode, "low_dim_obs.pkl"), "br"
            ) as f:
                d = pickle.load(f)

            for o in d:
                right_joint_positions.append(o.right.joint_positions)
                left_joint_positions.append(o.left.joint_positions)

                right_gripper_positions.append([o.right.gripper_joint_positions[0]])
                left_gripper_positions.append([o.left.gripper_joint_positions[0]])

        right_joint_positions = np.asarray(right_joint_positions, dtype=np.float32)
        left_joint_positions = np.asarray(left_joint_positions, dtype=np.float32)

        right_gripper_positions = np.asarray(right_gripper_positions, dtype=np.float32)
        left_gripper_positions = np.asarray(left_gripper_positions, dtype=np.float32)

        stats = {
            "right_joints_mean": right_joint_positions.mean(axis=0),
            "right_joints_std": right_joint_positions.std(axis=0),
            "left_joints_mean": left_joint_positions.mean(axis=0),
            "left_joints_std": left_joint_positions.std(axis=0),
            "right_gripper_mean": right_gripper_positions.mean(axis=0),
            "right_gripper_std": right_gripper_positions.std(axis=0),
            "left_gripper_mean": left_gripper_positions.mean(axis=0),
            "left_gripper_std": left_gripper_positions.std(axis=0),
        }

        return {k: torch.from_numpy(v).to(self.device) for k, v in stats.items()}

    def normalize_z(self, data, mean, std):
        return torch.nan_to_num((data - mean) / std, nan=0.0) 

    def unnormalize_z(self, data, mean, std):
        return torch.nan_to_num(data * std + mean, nan=0.0)

    def preprocess_qpos(self, observation: dict):
        stats = self.train_stats()

        right_qrev = self.normalize_z(
            observation["right_joint_positions"][:, 0],
            stats["right_joints_mean"],
            stats["right_joints_std"],
        )
        right_qgripper = self.normalize_z(
            observation["right_gripper_joint_positions"][:, 0],
            stats["right_gripper_mean"],
            stats["right_gripper_std"],
        )
        left_qrev = self.normalize_z(
            observation["left_joint_positions"][:, 0],
            stats["left_joints_mean"],
            stats["left_joints_std"],
        )
        left_qgripper = self.normalize_z(
            observation["left_gripper_joint_positions"][:, 0],
            stats["left_gripper_mean"],
            stats["left_gripper_std"],
        )
        qpos = torch.cat(
            [
                right_qrev,
                right_qgripper[:, 0].unsqueeze(-1),
                left_qrev,
                left_qgripper[:, 0].unsqueeze(-1),
            ],
            dim=-1,
        )

        return qpos

    def preprocess_action(self, replay_sample: dict):
        stats = self.train_stats()

        right_qrev = self.normalize_z(
            replay_sample["right_prev_joint_positions"][:, 0],
            stats["right_joints_mean"],
            stats["right_joints_std"],
        )
        right_qgripper = self.normalize_z(
            replay_sample["right_prev_gripper_joint_positions"][:, 0],
            stats["right_gripper_mean"],
            stats["right_gripper_std"],
        )
        left_qrev = self.normalize_z(
            replay_sample["left_prev_joint_positions"][:, 0],
            stats["left_joints_mean"],
            stats["left_joints_std"],
        )
        left_qgripper = self.normalize_z(
            replay_sample["left_prev_gripper_joint_positions"][:, 0],
            stats["left_gripper_mean"],
            stats["left_gripper_std"],
        )
        qpos = torch.cat(
            [
                right_qrev,
                right_qgripper[:, 0].unsqueeze(-1),
                left_qrev,
                left_qgripper[:, 0].unsqueeze(-1),
            ],
            dim=-1,
        )

        right_action_rev = self.normalize_z(
            replay_sample["right_next_joint_positions"],
            stats["right_joints_mean"],
            stats["right_joints_std"],
        )
        right_action_gripper = self.normalize_z(
            replay_sample["right_next_gripper_joint_positions"],
            stats["right_gripper_mean"],
            stats["right_gripper_std"],
        )
        left_action_rev = self.normalize_z(
            replay_sample["left_next_joint_positions"],
            stats["left_joints_mean"],
            stats["left_joints_std"],
        )
        left_action_gripper = self.normalize_z(
            replay_sample["left_next_gripper_joint_positions"],
            stats["left_gripper_mean"],
            stats["left_gripper_std"],
        )
        action_seq = torch.cat(
            [
                right_action_rev,
                right_action_gripper[:, :, 0].unsqueeze(-1),
                left_action_rev,
                left_action_gripper[:, :, 0].unsqueeze(-1),
            ],
            dim=-1,
        )

        return qpos, action_seq

    def preprocess_images(self, replay_sample: dict):
        stacked_rgb = []
        stacked_point_cloud = []

        for camera in self._camera_names:
            rgb = replay_sample["%s_rgb" % camera]
            rgb = rgb if rgb.dim() == 4 else rgb[:, 0]
            stacked_rgb.append(rgb)

            point_cloud = replay_sample["%s_point_cloud" % camera]
            point_cloud = point_cloud if point_cloud.dim() == 4 else point_cloud[:, 0]
            stacked_point_cloud.append(point_cloud)

        stacked_rgb = torch.stack(stacked_rgb, dim=1)
        stacked_point_cloud = torch.stack(stacked_point_cloud, dim=1)

        return stacked_rgb, stacked_point_cloud

    def update(self, step: int, replay_sample: dict) -> dict:
        """
        Args:
            step: ???
            replay_sample (dict): 

            {
                "image_data": (B,#camera,c,h,w),
                "qpos_data": (B,14), # normalized
                "action_data": (B,chunk_size,16), # normalized
                "is_pad": (B,chunk_size),
            }

        Returns:
            dict: _description_
        """
        self._update_count += 1
        self.policy.train()

        # preprocess input
        qpos, action_seq = self.preprocess_action(replay_sample)
        stacked_rgb, stacked_point_cloud = self.preprocess_images(replay_sample)
        is_pad = replay_sample["is_pad"].bool()

        forward_dict = self.policy(qpos, stacked_rgb,  action_seq, is_pad)

        loss = forward_dict['loss']
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._summaries = {
            "loss": forward_dict["loss"],
        }

        if self._scheduler is not None:
            self._scheduler.step()
            self._summaries.update({"lr": self._scheduler.get_last_lr()[0]})

        return forward_dict

    @torch.no_grad()
    def act(self, step: int, observation: dict, deterministic: bool=True) -> ActResult:
        """Note that the action returned is of size self._action_dim
        """
        assert deterministic

        stats = self.train_stats()

        # preprocess input
        qpos = self.preprocess_qpos(observation)
        stacked_rgb, stacked_point_cloud = self.preprocess_images(observation)

        if self._timestep == 0:
            # warm up
            for _ in range(10):
                self.policy(qpos, stacked_rgb)

        if self._timestep % self._query_frequency == 0:
            a_hat = self.policy(qpos, stacked_rgb).squeeze(0)  # (chunk_size,14)
            if not self.temporal_agg:
                self._all_actions = a_hat

        if self.temporal_agg:
            self._all_time_actions[self._timestep,
                                   self._timestep:self._timestep+self._num_queries, :] = a_hat
            actions_for_curr_step = self._all_time_actions[:, self._timestep]
            actions_populated = torch.all(
                actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k *
                                 np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(
                exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step *
                          exp_weights).sum(dim=0)  # (self._action_dim,)
        else:
            raw_action = self._all_actions[self._timestep % self._query_frequency]

        right_a_rev = self.unnormalize_z(
            raw_action[0:7], stats["right_joints_mean"], stats["right_joints_std"]
        )
        right_a_gripper = self.unnormalize_z(
            raw_action[7], stats["right_gripper_mean"], stats["right_gripper_std"]
        )

        left_a_rev = self.unnormalize_z(
            raw_action[8:15], stats["left_joints_mean"], stats["left_joints_std"]
        )
        left_a_gripper = self.unnormalize_z(
            raw_action[15], stats["left_gripper_mean"], stats["left_gripper_std"]
        )

        raw_action = torch.cat(
            [right_a_rev, right_a_gripper, left_a_rev, left_a_gripper], dim=-1
        )

        self._timestep += 1

        return ActResult(raw_action.detach().cpu().numpy())

    def load_weights(self, savedir: str) -> None:
        model_dict = torch.load(os.path.join(savedir, "policy.pt"))
        self.policy.deserialize(model_dict)
        print("Loaded weights from %s" % savedir)

    def save_weights(self, savedir: str) -> None:
        state_dict = self.policy.serialize()
        torch.save(state_dict, os.path.join(savedir, "policy.pt"))

    def act_summaries(self) -> List[Summary]:
        return []

    def update_summaries(self) -> List[Summary]:
        summaries = []
        wandb_dict = {}
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary("%s/%s" % (NAME, n), v))
            wandb_dict['%s/%s' % (NAME, n)] = v

        # for tag, param in self._actor.named_parameters():
        #     summaries.append(
        #
        #     summaries.append(
        #         HistogramSummary('%s/weight/%s' % (NAME, tag), param.data))

        return summaries, wandb_dict