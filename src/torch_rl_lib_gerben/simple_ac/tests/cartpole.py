# This will only run when gymnasium is installed. This is not a requirement for the algorithm implementations
import gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter

from src.torch_rl_lib_gerben.simple_ac.actor_critic_trainer import ActorCriticTrainer
from src.torch_rl_lib_gerben.simple_ac.gae_value_net import GaeValueNet
from src.torch_rl_lib_gerben.simple_ac.mc_value_net import McValueNet
from src.torch_rl_lib_gerben.simple_ac.policy_net import PolicyNet
from src.torch_rl_lib_gerben.simple_ac.ppo_policy_net import PpoPolicyNet
from src.torch_rl_lib_gerben.simple_ac.value_net import ValueNet


def env_init(render=False):
    return gymnasium.make("CartPole-v1", render_mode="human" if render else None).env


def convert_state(s):
    s = torch.from_numpy(s)
    s[0:1] /= 4.8
    s[2:3] /= 0.418
    return s


def convert_action(a):
    env_a = 1 if a.item() > 0.5 else 0
    return env_a


cuda = False
writer = SummaryWriter(comment="cartpole")

value_net = GaeValueNet(4, 64, gamma=0.999, fix_for_n_training_steps=1, summary_writer=writer, lr=3e-4)
policy_net = PpoPolicyNet(4, 64, entropy_factor=0.0, fix_for_n_training_steps=10, summary_writer=writer, lr=3e-4)

if cuda:
    value_net = value_net.cuda()
    policy_net = policy_net.cuda()

trainer = ActorCriticTrainer(policy_net, value_net, env_init, convert_state, convert_action,
                             n_trajectories=8, trajectory_length=100, summary_writer=writer, reward_multiplier=0.005,
                             cuda=cuda)

for train_step in range(50000):
    trainer.collect_and_train()

    # if train_step % 500 == 0:
    #     print("Score: ", trainer.test_on_env(True, 120.0))
