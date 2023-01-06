import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from torch_rl_lib_gerben.simple_ac.value_net import ValueNet


class McValueNet(ValueNet):
    def __init__(self, n_inputs, hidden_layer_size, **kwargs):
        super().__init__(n_inputs, hidden_layer_size, **kwargs)


    """
        Compute Monte Carlo returns and advantages
    """
    def compute_advantage_and_target_returns(self, states, rewards, dones):
        if not (states.size(1) == dones.size(1) and rewards.size(1) == states.size(1) - 1):
            raise ValueError(
                "Cannot compute advantage and target returns: "
                "expected reward trajectories to be 1 shorter than states and dones"
                "got states: %d, dones: %d, rewards: %d" % (states.size(1), dones.size(1), rewards.size(1))
            )

        with torch.no_grad():
            values = self.model(states)
            gained_values = torch.zeros_like(values)
            gained_values[:, -1] = self.model_fixed(states[:, -1])
            gained_values[:, -1][dones[:, -1]] = 0.0

            for i in range(states.size(1) - 2, -1, -1):
                gained_values[:, i] = rewards[:, i] + self.gamma * gained_values[:, i+1]
                gained_values[:, i][dones[:, i]] = 0.0

            gained_values = gained_values[:, :-1].contiguous()

            # Advantage calculation
            advantage = gained_values - values[:, :-1]

            if self.summary_writer is not None:
                self.summary_writer.add_scalar("ValueNet/mean_values",
                                               values.mean(),
                                               self.current_train_step
                                               )

        return advantage, gained_values
