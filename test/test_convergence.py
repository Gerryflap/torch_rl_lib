import torch
from torch.utils.tensorboard import SummaryWriter

from test.simple_env import SimpleEnv
from torch_rl_lib_gerben.actor_critic_trainer import ActorCriticTrainer
from torch_rl_lib_gerben.policy.ppo_policy_net import PpoPolicyNet
from torch_rl_lib_gerben.value.gae_value_net import GaeValueNet


def test_ppo_gae_simple_env():
    """
        Test method for PPO with GAE.
        Verifies that PPO/GAE still converges under these tested hyperparameters on the simple testing env
    """

    def env_init(render=False):
        return SimpleEnv()

    def convert_state(env_s):
        s = torch.ones((2,))
        s[0] = env_s / 20.0
        s[1] = 1.0 if env_s > 0 else -1.0
        return s

    def convert_action(a):
        env_a = a.item() * 2.0 - 1.0
        return env_a

    summary_writer = None
    # summary_writer = SummaryWriter(comment="simple_env")

    value_net = GaeValueNet(2, 64, summary_writer=summary_writer)
    policy_net = PpoPolicyNet(2, 64, summary_writer=summary_writer)

    trainer = ActorCriticTrainer(policy_net, value_net, env_init, convert_state, convert_action,
                                 reward_multiplier=0.1, summary_writer=summary_writer, trajectory_length=30,
                                 n_trajectories=2)

    for train_step in range(1500):
        trainer.collect_and_train()

    # Make sure that the average score of 10 runs is higher than -1.0
    assert sum([trainer.test_on_env() for _ in range(10)]) / 10.0 > -1.0
