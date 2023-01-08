import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter


class ValueNet(torch.nn.Module):
    """
        The default Value Network module.
        This class will provide the following features:
        - Represent the value network V(s) as a simple feedforward neural networks with 2 hidden layers.
        - Keep a fixed copy of V(s) that is only updated every N training steps (for stability)
        - Compute advantages A(s_t) for batches of trajectories using TD(0) estimates
        - Training V(s) on these TD(0) estimates

        Subclasses of the ValueNet provide different value or advantage estimators that may perform better
    """

    def __init__(self, hidden_layer_size, gamma=0.99, fix_for_n_training_steps=5, lr=0.0003,
                 batch_size=64, summary_writer: SummaryWriter = None, custom_model: torch.nn.Module = None,
                 n_inputs=None):
        """
        Initializes the value network
        :param hidden_layer_size: Size of the hidden layer(s). When providing a custom model,
            your output should be this size in order to allow the final layer to connect to it.
        :param gamma: Discount factor [0, 1].
            Rewards T steps in the future are discounted by multiplying them with gamma^T.
            It is recommended to keep this between 0.9 - 0.999 usually.
        :param fix_for_n_training_steps: Fix the fixed network for N training steps.
            Lower values may cause instability, higher values may hinder fast convergence.
            Keep this value low unless the V(s) estimates diverge during training
        :param lr: Learning rate given to the Adam optimizer
        :param batch_size: Size of the mini-batches used during training.
            The trajectory data collected from the environment is shuffled and split up in batches.
            The final batch of each training step will be smaller than batch_size if the total size of the
            provided data is not divisible by batch_size (this is not usually an issue).
        :param summary_writer: TensorBoard Summary writer. Will be used to log useful training statistics under
            "ValueNet/" when provided. Providing "None" (which is default) will disable TensorBoard logging.
        :param custom_model: Custom torch module that takes both input of size (N, ...) and (N, T, ...)
            where the "..." denotes the shape of the state.
            (NOTE: use ModelSequenceCompatibilityWrapper when (N, T, ...) is not supported by your architecture)
            It should output tensors of size (N, hidden_layer_size) or (N, T, hidden_layer_size) depending on the input.
            This can be used when the default network architecture is not sufficient (i.e. image inputs)
        :param n_inputs: Size of the input state vector. Only needed when you don't provide a custom model.
        """
        super().__init__()

        self.gamma = gamma
        self.fix_for_n_training_steps = fix_for_n_training_steps
        self.batch_size = batch_size
        self.summary_writer = summary_writer

        self.current_train_step = 0

        if custom_model is None:
            if n_inputs is None:
                raise ValueError("Cannot initialize ValueNet: Either custom_model or n_inputs needs to be defined!")

            custom_model = torch.nn.Sequential(
                torch.nn.Linear(n_inputs, hidden_layer_size),
                torch.nn.Tanh(),

                torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                torch.nn.Tanh(),
            )

        self.model = torch.nn.Sequential(
            custom_model,
            torch.nn.Linear(hidden_layer_size, 1),
        )

        self.model_fixed = deepcopy(self.model)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, states):
        return self.model(states)

    def compute_advantage_and_target_returns(self, states, rewards, dones):
        """
            Given batches of trajectories of states, rewards, and done values,
                compute A(s) and target V(s) for all s in the given trajectories apart from the last state in
                the trajectory.

            For computing r_t + gamma * V(s_(t+1)), a fixed copy of the value network is used that is updated
                after "fix_for_n_training_steps" steps.

            :param states: Batch of state trajectories.
                Example shape for batch_size of 32, a trajectory len of 101,
                and a single observation being a vector with 8 elements would be (32, 101, 8).
                The trajectory length for states and dones is one longer than the rewards because
            :param rewards: Batch of reward trajectories. With the above example this would be (32, 100, 1)
            :param dones: Batch of boolean done values. A "True" value denotes the end state of a trajectory.
                The state thereafter is assumed to be the start of a new one.
                For any s_t that is a terminal state, V(s_t) = 0, since no more reward can be gained.
                Example shape: (32, 101, 1)

            :return: Returns a tuple: A(s_t) and r_t + gamma * V(s_(t+1)) for all states given except the last state
                of every trajectory.
                A(s_t) denotes the advantage gained by performing a_t, which is the difference between the gained
                values r_t + gamma * V(s_(t+1)) and the expected values V(s_t).
        """
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

    def training_step(self, states, gained_values):
        """
            Trains the value model on the "gained_values" target values for the given states.
            "gained_values" can be retrieved from the compute_advantage_and_target_returns method.
            Training is done in batches of batch_size. The final batch may be smaller than batch_size
        """
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
