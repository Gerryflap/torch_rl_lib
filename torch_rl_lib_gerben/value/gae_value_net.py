import torch
from torch_rl_lib_gerben.value.value_net import ValueNet


class GaeValueNet(ValueNet):
    """
    Value Network that implements Generalized Advantage Estimation (GAE).
    Values at the end of the trajectories are bootstrapped using the fixed value network.

    Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation."
        arXiv preprint arXiv:1506.02438 (2015).
    """
    def __init__(self, hidden_layer_size, lambd=0.95, **kwargs):
        """
        Initializes the GAE value network
        :param hidden_layer_size: Size of the hidden layers
        :param lambd: The GAE lambda parameter, should be in range [0, 1].
            Trade-off between bias (lower) and variance (higher).
            Using lambda=0 will result in an advantage estimate r_t + gamma * V(s_(t+1)) - V(s) and using
            lambda=1 will result in (r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ... ) - V(s)
        :param kwargs: The other keyword arguments that can be provided to the default ValueNet
        """
        super().__init__(hidden_layer_size, **kwargs)
        self.lambd = lambd

    def compute_advantage_and_target_returns(self, states, rewards, dones):
        if not (states.size(1) == dones.size(1) and rewards.size(1) == states.size(1) - 1):
            raise ValueError(
                "Cannot compute advantage and target returns: "
                "expected reward trajectories to be 1 shorter than states and dones"
                "got states: %d, dones: %d, rewards: %d" % (states.size(1), dones.size(1), rewards.size(1))
            )

        with torch.no_grad():
            values = self.model(states)
            fixed_values = self.model_fixed(states)
            fixed_values[dones] = 0.0
            advantage_estimates = torch.zeros_like(values[:, :-1])
            advantage_estimates[:, -1] = rewards[:, -1] + self.gamma * fixed_values[:, -1] - values[:, -2]

            for i in range(advantage_estimates.size(1) - 2, -1, -1):
                delta = rewards[:, i] + self.gamma * fixed_values[:, i + 1] - values[:, i]
                advantage_estimates[:, i] = delta + self.gamma * self.lambd * advantage_estimates[:, i + 1]

            # Compute the actual gained value by computing V(s) + A(s) (not sure if this is how GAE did it)
            gained_values = values[:, :-1] + advantage_estimates

            if self.summary_writer is not None:
                self.summary_writer.add_scalar("ValueNet/mean_values",
                                               values.mean(),
                                               self.current_train_step
                                               )

        return advantage_estimates, gained_values
