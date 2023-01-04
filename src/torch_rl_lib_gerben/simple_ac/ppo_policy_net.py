from copy import deepcopy

import torch.nn
from torch.utils.tensorboard import SummaryWriter

from src.torch_rl_lib_gerben.simple_ac.policy_net import PolicyNet


class PpoPolicyNet(PolicyNet):
    def __init__(self, n_inputs, hidden_layer_size, clip_eps=0.2, **kwargs):
        super().__init__(n_inputs, hidden_layer_size, **kwargs)
        self.clip_eps = clip_eps

    def compute_loss(self, states, actions, advantages):
        dist = self(states)
        dist_fixed = self.forward(states, fixed_net=True)
        ratio = torch.exp(dist.log_prob(actions) - dist_fixed.log_prob(actions))
        ratio = torch.nan_to_num(ratio, nan=1.0)
        loss = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        ) / self.batch_size
        if self.entropy_factor != 0.0:
            loss -= self.entropy_factor * dist.entropy()
        return loss
