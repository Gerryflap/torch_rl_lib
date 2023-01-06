import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter


class ValueNet(torch.nn.Module):
    def __init__(self, n_inputs, hidden_layer_size, gamma=0.99, fix_for_n_training_steps=5, lr=0.0003,
                 batch_size=64, summary_writer: SummaryWriter = None):
        super().__init__()

        self.gamma = gamma
        self.fix_for_n_training_steps = fix_for_n_training_steps
        self.batch_size = batch_size
        self.summary_writer = summary_writer

        self.current_train_step = 0

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, 1),
        )

        self.model_fixed = deepcopy(self.model)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, states):
        return self.model(states)

    """
        Given batches of trajectories of states, rewards, and done values, 
        compute A(s) and target V(s) for all s in the given trajectories apart from the last state in the trajectory.
        For computing r_t + gamma * V(s_(t+1)), a fixed copy of the value network is used that is updated 
            after "fix_for_n_training_steps" steps.
        
        :param states: Batch of state trajectories. 
            Example shape for batch_size of 32, a trajectory len of 101, 
            and a single observation being a vector with 8 elements would be (32, 101, 8).
            The trajectory length for states and dones is one longer than the rewards because 
        :param rewards: Batch of reward trajectories. With the above example this would be (32, 100, 1)
        :param dones: Batch of boolean done values. A "True" value denotes the end state of a trajectory.
            The state thereafter is assumed to be the start of a new one.
            Any V(s) when s is done will be set to 0, since no more reward can be gained. Example shape: (32, 101, 1)
            
        :return: Returns a tuple: A(s_t) and r_t + gamma * V(s_(t+1)) for all states given except the last state of every trajectory.
            A(s_t) denotes the advantage gained by performing a_t. 
            This is the difference between the gained values r_t + gamma * V(s_(t+1)) and the expected values V(s_t). 
    """

    def compute_advantage_and_target_returns(self, states, rewards, dones):
        if not (states.size(1) == dones.size(1) and rewards.size(1) == states.size(1) - 1):
            raise ValueError(
                "Cannot compute advantage and target returns: "
                "expected reward trajectories to be 1 shorter than states and dones"
                "got states: %d, dones: %d, rewards: %d" % (states.size(1), dones.size(1), rewards.size(1))
            )

        with torch.no_grad():
            values = self.model(states)
            fixed_values = self.model_fixed(states)
            values[dones] = 0
            fixed_values[dones] = 0

            # Gained values = r_t + gamma * V(s_(t+1))
            # Add r_t
            gained_values = rewards[:]

            # Only for states that aren't done, add V(s_(t+1)):
            gained_values[~dones[:, :-1]] += self.gamma * fixed_values[:, 1:][~dones[:, :-1]]

            # Advantage calculation
            advantage = gained_values - values[:, :-1]

            if self.summary_writer is not None:
                self.summary_writer.add_scalar("ValueNet/mean_values",
                                               values.mean(),
                                               self.current_train_step
                                               )

        return advantage, gained_values

    """
        Trains the value model on the "gained_values" target values for the given states.
        "gained_values" can be retrieved from the compute_advantage_and_target_returns method.
        Training is done in batches of batch_size. The final batch may be smaller than batch_size 
    """

    def training_step(self, states, gained_values):
        n_trajectories = states.size(0)
        trajectory_len = states.size(1) - 1
        n_samples = n_trajectories * trajectory_len
        if not (n_trajectories == gained_values.size(0) and trajectory_len == gained_values.size(1)):
            raise ValueError("Expected n_trajectories: %d and trajectory_len: %d, but got gained_values of size %s" %
                             (n_trajectories, trajectory_len, str(gained_values.size())))

        gained_values = gained_values.view((n_trajectories * trajectory_len, 1))
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("ValueNet/mean_gained_values",
                                           gained_values.mean(),
                                           self.current_train_step
                                           )
        small_states = states[:, :-1].contiguous()
        states = small_states.view(*((n_trajectories * trajectory_len,) + states.size()[2:]))

        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, self.batch_size):
            self.optim.zero_grad()
            start, stop = i * self.batch_size, i * self.batch_size + self.batch_size
            batch_indices = indices[start:stop]
            pred = self.model(states[batch_indices])
            mse = torch.square(pred - gained_values[batch_indices]).mean()
            mse.backward()
            self.optim.step()

        self.current_train_step += 1
        if self.current_train_step % self.fix_for_n_training_steps == 0:
            self.model_fixed.load_state_dict(self.model.state_dict())
