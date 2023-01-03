# This will only run when gymnasium is installed. This is not a requirement for the algorithm implementations
import gymnasium
import torch

from src.torch_rl_lib_gerben.simple_ac.actor_critic_trainer import ActorCriticTrainer
from src.torch_rl_lib_gerben.simple_ac.policy_net import PolicyNet
from src.torch_rl_lib_gerben.simple_ac.value_net import ValueNet


def env_init(render=False):
    return gymnasium.make("Pendulum-v1", render_mode="human" if render else None)


def convert_state(s):
    s = torch.from_numpy(s)
    s[2] /= 8.0
    return s


def convert_action(a):
    return a.numpy() * 4.0 - 2.0


value_net = ValueNet(3, 64)
policy_net = PolicyNet(3, 64, entropy_factor=1e-3)

trainer = ActorCriticTrainer(policy_net, value_net, env_init, convert_state, convert_action, 1,
                             reward_multiplier=1/16.0)

for train_step in range(10000):
    trainer.collect_and_train()

    if train_step % 100 == 0:
        print("Score: ", trainer.test_on_env(True))
