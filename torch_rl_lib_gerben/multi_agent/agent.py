from copy import deepcopy
from threading import Lock

import torch

from torch_rl_lib_gerben.policy.policy_net import PolicyNet


class Agent:
    """
        Class to represent an agent in an environment.
        Since it has no hidden state, it can be used for multiple agents at the same time.
        Model should be updated either manually, or automatically by registering it to the Trainer.
        This class does not collect the data on its own, you need to create Trajectory objects for that.
    """

    def __init__(self, cuda=False):
        self.model = None
        self.cuda = cuda
        self.clip_beta = False
        self.lock = Lock()

    def update_model(self, pi: PolicyNet):
        """
        Updates the model, using a copy of the given policy network.
        Automagically makes a copy and moves the network to the preferred device
        :param pi: The updated policy
        """
        with self.lock:
            if pi.is_cuda():
                if self.cuda:
                    self.model = deepcopy(pi.model)
                else:
                    self.model = deepcopy(pi.model).cpu()
            else:
                if self.cuda:
                    self.model = deepcopy(pi.model).cuda()
                else:
                    self.model = deepcopy(pi.model)

            self.model = pi.model
            self.clip_beta = pi.clip_beta

    def sample_action(self, s: torch.Tensor):
        """
        Sample action (given the state as a cpu Tensor)
        :param s: State tensor
        :return: A Tensor containing the action taken by the agent
        """
        with torch.no_grad():
            if self.cuda:
                s = s.cuda()

            with self.lock:
                pred = self.model(s.view((-1,) + s.size()))
            n_outputs = pred.size()[-1] // 2
            pred = torch.nn.functional.softplus(pred)

            if self.clip_beta:
                pred += 1.0

            distribution = torch.distributions.Beta(pred[:, :n_outputs], pred[:, n_outputs:])
            return distribution.sample()[0]
