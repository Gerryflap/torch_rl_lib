import time

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_rl_lib_gerben.policy.policy_net import PolicyNet
from torch_rl_lib_gerben.util.timer import Timer
from torch_rl_lib_gerben.value.value_net import ValueNet


class ActorCriticTrainer:
    """
        Simple trainer class that collects N trajectories of length T and then trains the policy and value networks
    """

    def __init__(self, pi: PolicyNet, v: ValueNet, env_constructor, state_converter, action_converter,
                 n_trajectories=8, trajectory_length=100, reward_multiplier=1.0, summary_writer: SummaryWriter = None,
                 cuda=False):
        """
            Initializes the ActorCriticTrainer
            :param pi: The policy network
            :param v: The value network
            :param env_constructor: Function to initialize a new (gym-like) environment.
                Should have an optional "render" boolean parameter that is True when we want to render in human mode
            :param state_converter: Function that converts state to a PyTorch tensor (without batch dim)
            :param action_converter: Function that converts an action from the model to an action in the env.
                Note that actions by default will be in the range [0.0, 1.0], so scale accordingly!
            :param n_trajectories: Number of "parallel" trajectories used during training. i.e. when you set this to 32,
                then separate instances of the environment will be used to collect data for every training step
            :param trajectory_length: The length of every trajectory before performing a training step.
                After training, the env will continue again for the next batch of samples.
                If the environment reaches a "done" state, the trainer will reset them and continue until enough data
                is collected.
            :param reward_multiplier: Is multiplied with the reward during training (not testing) to scale
                the rewards into a more reasonable window
            :param summary_writer: TensorBoard summary writer to use
            :param cuda: When true, perform training and collection on the GPU
        """
        self.n_trajectories = n_trajectories
        self.trajectory_length = trajectory_length
        self.state_converter = state_converter
        if cuda:
            self.state_converter = lambda s: state_converter(s).cuda()
        self.action_converter = action_converter
        self.n_actions = pi.n_outputs
        self.env_constructor = env_constructor
        self.reward_multiplier = reward_multiplier
        self.summary_writer = summary_writer
        self.current_train_step = 0
        self.cuda = cuda
        self.device = 'cuda' if cuda else 'cpu'
        self.timer = Timer()

        self.pi = pi
        self.v = v

        # Get state shape
        self.temp_env = env_constructor()
        self.state_shape = state_converter(self.temp_env.reset()[0]).size()
        self.temp_env.close()

        # Init envs and info about previous state and done values
        self.envs = [env_constructor() for _ in range(n_trajectories)]
        self.prev_last_state_and_done = [(env.reset()[0], False) for env in self.envs]

        # Init arrays
        self.states = torch.zeros((n_trajectories, trajectory_length + 1) + self.state_shape, device=self.device)
        self.rewards = torch.zeros(n_trajectories, trajectory_length, 1, device=self.device)
        self.actions = torch.zeros(n_trajectories, trajectory_length, self.n_actions, device=self.device)
        self.dones = torch.zeros(n_trajectories, trajectory_length + 1, 1, dtype=torch.bool, device=self.device)
        self.reset_arrays()

        # Init score system
        self.scores = [0.0 for _ in range(n_trajectories)]

    # Empties arrays and fills first index of state and done arrays with previous state and done information
    def reset_arrays(self):
        self.actions[:] = 0.0
        self.rewards[:] = 0.0
        self.states[:] = 0.0
        self.dones[:] = False

        self.states[:, 0] = torch.stack([self.state_converter(s) for s, _ in self.prev_last_state_and_done], dim=0)
        self.dones[:, 0] = torch.BoolTensor([d for _, d in self.prev_last_state_and_done]).view(self.n_trajectories, 1)

    def collect(self):
        for trajectory_index in range(self.n_trajectories):
            s = self.state_converter(self.prev_last_state_and_done[trajectory_index][0])
            done = self.prev_last_state_and_done[trajectory_index][1]
            s_new_orig = None

            for step in range(self.trajectory_length):
                if not done:
                    self.actions[trajectory_index, step] = self.pi.get_action(s)
                    s_new_orig, r, term, trunc, _ = self.envs[trajectory_index].step(self.action_converter(
                        self.actions[trajectory_index, step]))

                    s_new = self.state_converter(s_new_orig)
                    done = term or trunc
                    self.rewards[trajectory_index, step] = r
                    self.dones[trajectory_index, step + 1] = done
                    self.states[trajectory_index, step + 1] = s_new

                    self.scores[trajectory_index] += r
                else:
                    s_new_orig, _ = self.envs[trajectory_index].reset()
                    s_new = self.state_converter(s_new_orig)
                    done = False
                    self.states[trajectory_index, step + 1] = s_new

                    if self.summary_writer is not None:
                        self.summary_writer.add_scalar("Trainer/train_scores",
                                                       self.scores[trajectory_index],
                                                       self.current_train_step
                                                       )
                    self.scores[trajectory_index] = 0.0

                s = s_new

            self.prev_last_state_and_done[trajectory_index] = (s_new_orig, done)

    def collect_and_train(self):
        if self.summary_writer:
            self.timer.reset()
        self.collect()

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/collect",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        adv, collected_vs = self.v.compute_advantage_and_target_returns(
            self.states,
            self.rewards * self.reward_multiplier,
            self.dones
        )
        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/compute_advantages",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        self.pi.training_step(self.states, self.actions, self.dones, adv)

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/train_pi",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        self.v.training_step(self.states, collected_vs)

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/train_v",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )
        self.current_train_step += 1

    # Test the agent on a full trajectory and return the sum of rewards (and optionally render at an optional fps)
    def test_on_env(self, render=False, cap_fps=None):
        done = False
        env = self.env_constructor(render=render)
        s, _ = env.reset()
        score = 0

        while not done:
            a = self.pi.get_action(self.state_converter(s))
            s_new_orig, r, term, trunc, _ = env.step(self.action_converter(a))
            score += r
            done = term or trunc
            s = s_new_orig

            if render:
                if cap_fps is not None:
                    time.sleep(1.0 / cap_fps)
        env.close()
        return score
