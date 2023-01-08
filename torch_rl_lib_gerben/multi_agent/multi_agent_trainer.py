import time
from queue import Queue, Empty

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_rl_lib_gerben.multi_agent.agent import Agent
from torch_rl_lib_gerben.multi_agent.trajectory import Trajectory
from torch_rl_lib_gerben.policy.policy_net import PolicyNet
from torch_rl_lib_gerben.util.timer import Timer
from torch_rl_lib_gerben.value.value_net import ValueNet


class MultiAgentTrainer:
    """
        Trainer class that can be used in cases that require more flexibility, such as multi-agent environments or
            scenario's where the environment cannot be tightly controlled.

        Rather than all agents submitting data in lock-step, trajectories can just be submitted to the queue when
            they're ready.
        The trainer will only start a train step when enough Trajectory objects are in the queue.
    """

    def __init__(self, pi: PolicyNet, v: ValueNet, queue=None, summary_writer: SummaryWriter = None, cuda=False,
                 queue_size=200, update_agents_every_n=1):
        """
        Initializes the trained
        :param pi: The policy network pi
        :param v: The value network V
        :param queue: Optional custom Queue. In certain cases, like multiprocessing applications,
            a custom Queue can be given. The queue should contain Trajectory objects provided by agents.
            Note: It is recommended to enter a max queue size unless you have infinite RAM.
        :param summary_writer: TensorBoard summary writer to use (tensorboard is disabled when none is given)
        :param cuda: When true, perform training on the GPU
        :param queue_size: Max size of the queue, only used when no custom queue is used
        :param update_agents_every_n: Update agents every N steps.
        """
        self.update_agents_every_n = update_agents_every_n
        self.agents = []
        self.pi = pi
        self.v = v
        self.queue = queue
        self.interrupted = False
        if self.queue is None:
            self.queue = Queue(maxsize=queue_size)
        self.summary_writer = summary_writer
        self.cuda = cuda
        self.timer = Timer()
        self.current_train_step = 0

    def add_trajectory(self, trajectory: Trajectory, block=True):
        """
        Adds the trajectory to the queue
        :param trajectory: The trajectory
        :param block: (default: True) Block the thread until there is space in the queue (when full).
            Choosing False may throw queue.Full if the trainer cannot keep up.
        """
        self.queue.put(trajectory, block=block)

    def collect(self, n_trajectories):
        """
        Collect trajectories
        """
        states = []
        actions = []
        rewards = []
        dones = []

        for _ in range(n_trajectories):
            trajectory = None
            while trajectory is None and not self.interrupted:
                try:
                    trajectory = self.queue.get_nowait()
                except Empty:
                    time.sleep(0.05)
            states.append(trajectory.states)
            actions.append(trajectory.actions)
            rewards.append(trajectory.rewards)
            # TODO: Temporary hack, until compatibility is added
            dones.append(torch.logical_or(trajectory.terms, trajectory.truncs))

        if self.interrupted:
            return None, None, None, None

        arrays = torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones)
        if self.cuda:
            arrays = tuple([array.cuda() for array in arrays])
        return arrays

    def collect_and_train(self, n_trajectories):
        """
            Will attempt to do a training step. Will also reset the "interrupted" status.
        """
        self.interrupted = False

        if self.summary_writer:
            self.timer.reset()
        states, actions, rewards, dones = self.collect(n_trajectories)

        if self.interrupted:
            return

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/collect",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        adv, collected_vs = self.v.compute_advantage_and_target_returns(
            states,
            rewards,
            dones
        )
        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/compute_advantages",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        self.pi.training_step(states, actions, dones, adv)

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/train_pi",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )

        self.v.training_step(states, collected_vs)

        if self.summary_writer:
            self.summary_writer.add_scalar("Timing/train_v",
                                           self.timer.get_duration_and_reset(),
                                           self.current_train_step
                                           )
        self.current_train_step += 1

        self.notify_agents()

    def interrupt(self):
        """
        Will interrupt the training_step method
        :return:
        """
        self.interrupted = True

    def notify_agents(self):
        """
        Updates the models of the agents
        """
        for agent in self.agents:
            agent.update_model(self.pi)

    def register_agent(self, agent: Agent):
        """
        Registers agent for automatic policy model updates and updates the model instantly.
            Note: don't forget to unregister :3
        :param agent: The agent object
        """
        self.agents.append(agent)
        agent.update_model(self.pi)

    def unregister_agent(self, agent: Agent):
        """
        Unregisters agent from automatic policy model updates
        :param agent: The agent to remove from the list
        """
        self.agents.remove(agent)
