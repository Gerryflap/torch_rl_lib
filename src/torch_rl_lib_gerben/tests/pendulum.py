# This will only run when gymnasium is installed. This is not a requirement for the algorithm implementations
import gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter

from torch_rl_lib_gerben.actor_critic_trainer import ActorCriticTrainer
from torch_rl_lib_gerben.value.mc_value_net import McValueNet
from torch_rl_lib_gerben.policy.policy_net import PolicyNet


def env_init(render=False):
    return gymnasium.make("Pendulum-v1", render_mode="human" if render else None)


def convert_state(s):
    s = torch.from_numpy(s)
    s[2] /= 8.0
    return s


# sin_emb = SinusoidalEmbedding()
#
#
# def convert_state_sin(s):
#     s = torch.from_numpy(s)
#     s[2] /= 8.0
#     s = sin_emb(s.view(1, -1)).view(-1)
#     return s


def convert_action(a):
    return a.numpy() * 4.0 - 2.0


writer = SummaryWriter(comment="pendulum")

value_net = McValueNet(3, 64, gamma=0.99, fix_for_n_training_steps=10, summary_writer=writer, lr=1e-3)
policy_net = PolicyNet(3, 64, entropy_factor=1e-4, summary_writer=writer, lr=1e-3)

trainer = ActorCriticTrainer(policy_net, value_net, env_init, convert_state, convert_action,
                             n_trajectories=8, trajectory_length=135, summary_writer=writer, reward_multiplier=1/16.0)

for train_step in range(50000):
    trainer.collect_and_train()

    # if train_step % 100 == 0:
    #     print("Score: ", trainer.test_on_env(False))
