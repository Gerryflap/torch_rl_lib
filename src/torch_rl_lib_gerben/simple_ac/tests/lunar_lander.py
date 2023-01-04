# This will only run when gymnasium is installed. This is not a requirement for the algorithm implementations
import gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter

from src.torch_rl_lib_gerben.simple_ac.actor_critic_trainer import ActorCriticTrainer
from src.torch_rl_lib_gerben.simple_ac.mc_value_net import McValueNet
from src.torch_rl_lib_gerben.simple_ac.policy_net import PolicyNet
from src.torch_rl_lib_gerben.simple_ac.ppo_policy_net import PpoPolicyNet
from src.torch_rl_lib_gerben.simple_ac.value_net import ValueNet


def env_init(render=False):
    return gymnasium.make("LunarLander-v2", continuous=True, render_mode="human" if render else None)


def convert_state(s):
    s = torch.from_numpy(s)
    return s


def convert_action(a):
    return (a * 2.0 - 1.0).numpy()


writer = SummaryWriter()

value_net = McValueNet(8, 64, gamma=0.999, fix_for_n_training_steps=10, summary_writer=writer, lr=3e-4)
policy_net = PpoPolicyNet(8, 64, n_outputs=2, entropy_factor=0.0, fix_for_n_training_steps=10,
                          summary_writer=writer, lr=3e-4)

trainer = ActorCriticTrainer(policy_net, value_net, env_init, convert_state, convert_action,
                             n_trajectories=8, trajectory_length=100, summary_writer=writer, reward_multiplier=0.05)

for train_step in range(50000):
    trainer.collect_and_train()

    if train_step % 500 == 0:
        print("Score: ", trainer.test_on_env(True, 120.0))
