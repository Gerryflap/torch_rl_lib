import torch

from torch_rl_lib_gerben.value.value_net import ValueNet


class McValueNet(ValueNet):
    """
        Monte Carlo Value Network.
        Computes gained value entirely using the sum of discounter rewards from the trajectory,
            except for bootstrapping at the end (which is done with the fixed V network.

        This introduces a lot of variance between runs, but suffers way less from the inherent bias caused by using
            the estimator V.
    """
    def __init__(self, hidden_layer_size, **kwargs):
        """
        Initializes the MC value network
        :param hidden_layer_size: Size of the hidden layers
        :param kwargs: The other keyword arguments that can be provided to the default ValueNet
        """
        super().__init__(hidden_layer_size, **kwargs)

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
