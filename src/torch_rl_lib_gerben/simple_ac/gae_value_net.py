import torch
from src.torch_rl_lib_gerben.simple_ac.value_net import ValueNet


class GaeValueNet(ValueNet):
    def __init__(self, n_inputs, hidden_layer_size, lambd=0.95, **kwargs):
        super().__init__(n_inputs, hidden_layer_size, **kwargs)
        self.lambd = lambd

    """
        Compute GAE(lambda) (Generalized Advantage Estimation) advantages and returns
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
