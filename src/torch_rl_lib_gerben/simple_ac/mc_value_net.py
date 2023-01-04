import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from src.torch_rl_lib_gerben.simple_ac.value_net import ValueNet


class McValueNet(ValueNet):
    def __init__(self, n_inputs, hidden_layer_size, gamma=0.99, fix_for_n_training_steps=5, lr=0.0003,
                 batch_size=64, summary_writer: SummaryWriter = None):
        super().__init__(n_inputs, hidden_layer_size, gamma=gamma, fix_for_n_training_steps=fix_for_n_training_steps,
                         lr=lr,
                         batch_size=batch_size, summary_writer=summary_writer)


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

    """
        Trains the value model on the "gained_values" target values for the given states.
        "gained_values" can be retrieved from the compute_advantage_and_target_returns method.
        Training is done in batches of batch_size. The final batch may be smaller than batch_size 
    """

    def training_step(self, states, gained_values):
        n_trajectories = states.size(0)
        trajectory_len = states.size(1) - 1
        n_samples = n_trajectories * trajectory_len
        if not (n_trajectories == gained_values.size(0) and trajectory_len == gained_values.size(1)):
            raise ValueError("Expected n_trajectories: %d and trajectory_len: %d, but got gained_values of size %s" %
                             (n_trajectories, trajectory_len, str(gained_values.size())))

        gained_values = gained_values.view((n_trajectories * trajectory_len, 1))
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("ValueNet/mean_gained_values",
                                           gained_values.mean(),
                                           self.current_train_step
                                           )
        small_states = states[:, :-1].contiguous()
        states = small_states.view(*((n_trajectories * trajectory_len,) + states.size()[2:]))

        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, self.batch_size):
            self.optim.zero_grad()
            start, stop = i * self.batch_size, i * self.batch_size + self.batch_size
            batch_indices = indices[start:stop]
            pred = self.model(states[batch_indices])
            mse = torch.square(pred - gained_values[batch_indices]).mean()
            mse.backward()
            self.optim.step()

        self.current_train_step += 1
        if self.current_train_step % self.fix_for_n_training_steps == 0:
            self.model_fixed.load_state_dict(self.model.state_dict())
