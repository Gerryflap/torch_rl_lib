import torch.nn

from torch_rl_lib_gerben.policy.policy_net import PolicyNet


class PpoPolicyNet(PolicyNet):
    """
        PolicyNet that implements the PPO (Proximal Policy Optimization) loss function.
        Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

        To summarize: PPO resolves an instability in A2C (the default PolicyNet) by keeping a second fixed
            policy network whose weights are only updated to the new values every N training steps.
            Updates done to pi(s) cannot deviate too much from pi_fixed(s).
            This ensures that the policy network does not change too fast, and massively improves stability.
    """

    def __init__(self, n_inputs, hidden_layer_size, clip_eps=0.2, **kwargs):
        """
        Initializes the PpoPolicyNetwork
        :param n_inputs: Size of the input vector
        :param hidden_layer_size: number of neurons in every hidden layer
        :param clip_eps: ε used in PPO to clip policy deviation from the fixed policy network.
            Higher values may lead to faster training and
            lower values will sacrifice training speed in favor of stability.
            It is recommended to keep this around 0.2 unless you have issues.
        :param kwargs: The other default PolicyNet arguments
        """
        super().__init__(n_inputs, hidden_layer_size, **kwargs)
        self.clip_eps = clip_eps

    def compute_loss(self, states, actions, advantages):
        # PPO loss function
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
