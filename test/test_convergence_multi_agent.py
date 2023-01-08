import time
from queue import Full
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter

from test.simple_env import SimpleEnv
from torch_rl_lib_gerben.multi_agent.agent import Agent
from torch_rl_lib_gerben.multi_agent.multi_agent_trainer import MultiAgentTrainer
from torch_rl_lib_gerben.multi_agent.trajectory import Trajectory
from torch_rl_lib_gerben.policy.ppo_policy_net import PpoPolicyNet
from torch_rl_lib_gerben.value.gae_value_net import GaeValueNet


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


running = True


def data_collection_thread(agent: Agent, trainer: MultiAgentTrainer):
    env = env_init()
    s_env, _ = env.reset()
    s = convert_state(s_env)

    trajectory = Trajectory(30, (s, False, False), 1)
    done = False

    while running:
        if not done:
            a = agent.sample_action(s)
            s_env, r, term, trunc, _ = env.step(convert_action(a))
            s = convert_state(s_env)
            trajectory.submit_data(a, s, r * 0.1, term, trunc)
            done = term or trunc
        else:
            s_env, _ = env.reset()
            s = convert_state(s_env)
            trajectory.submit_reset_data(s)
            done = False

        if trajectory.is_full():
            success = False
            while (not success) and running:
                try:
                    trainer.add_trajectory(trajectory, block=False)
                    success = True
                except Full:
                    time.sleep(0.05)
            prev_state = trajectory.get_final_state()
            trajectory = Trajectory(30, prev_state, 1)


def verify_on_env(agent: Agent):
    env = env_init()
    s_env, _ = env.reset()
    s = convert_state(s_env)

    done = False
    score = 0.0

    while not done:
        a = agent.sample_action(s)
        s_env, r, term, trunc, _ = env.step(convert_action(a))
        s = convert_state(s_env)
        score += r
        done = term or trunc
    return score


def test_multi_agent_ppo_gae_simple_env():
    """
        Test method for PPO with GAE using the Multi Agent stack.
        NOTE: This is still a single agent environment, the multi-agent stack is only used to test it.
        Verifies that PPO/GAE still converges under these tested hyperparameters on the simple testing env
    """
    print("Starting test")
    global running
    summary_writer = None
    summary_writer = SummaryWriter(comment="multi_simple_env")

    custom_policy_model = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.Tanh(),

        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
    )

    value_net = GaeValueNet(64, summary_writer=summary_writer, n_inputs=2)
    policy_net = PpoPolicyNet(64, summary_writer=summary_writer, custom_model=custom_policy_model)

    trainer = MultiAgentTrainer(policy_net, value_net, summary_writer=summary_writer, queue_size=10)
    threads = []

    for _ in range(2):
        agent = Agent()
        trainer.register_agent(agent)
        thread = Thread(target=lambda: data_collection_thread(agent, trainer))
        threads.append(thread)
        thread.start()

    for train_step in range(150):
        trainer.collect_and_train(2)

    running = False

    print("Stopping threads")
    for thread in threads:
        thread.join()

    print("Evaluating")
    agent = Agent()
    agent.update_model(trainer.pi)
    # Make sure that the average score of 10 runs is higher than -1.0
    assert sum([verify_on_env(agent) for _ in range(10)]) / 10.0 > -1.0
