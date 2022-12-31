import torch


class ValueNet(torch.nn.Module):
    def __init__(self, n_inputs, hidden_layer_size, gamma=0.99):
        super().__init__()

        self.gamma = gamma

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, 1),
            torch.nn.Tanh(),
        )

    def forward(self, states):
        return self.model(states)

    """
        Given batches of trajectories of states, rewards, and done values, 
        compute A(s) and target V(s) for all s in the given trajectories apart from the last state in the trajectory.
        
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
                "got states: %d, dones: %d, rewards: %d"%(states.size(1), dones.size(1), rewards.size(1))
            )

        with torch.no_grad():
            values = self.model(states)
            values[dones] = 0

            # Gained values = r_t + gamma * V(s_(t+1))
            # Add r_t
            gained_values = rewards[:]

            # Only for states that aren't done, add V(s_(t+1)):
            gained_values[~dones[:-1]] += self.gamma * values[:, 1:][~dones[:-1]]

            # Advantage calculation
            advantage = gained_values - values[:-1]

        return advantage, gained_values
