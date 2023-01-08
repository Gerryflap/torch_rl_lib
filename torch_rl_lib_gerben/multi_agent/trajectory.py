import torch


class Trajectory:
    """
        Trajectory class. Used for saving a trajectory of a fixed length for one agent.
        Note that the intended use is to reset the env and keep providing data to the same trajectory until it is full
            after termination or truncation, rather than to cut it off after one episode.
        When the Trajectory is filled, you are expected to create a new one yourself
            (and submit the full one to the trainer)
    """
    def __init__(self, trajectory_length, initial_state: tuple, n_actions):
        """
        Initializes a new trajectory object
        :param trajectory_length: Length of the trajectory
        :param initial_state: tuple containing (state: Tensor, terminated: bool, truncated: bool).
        :param n_actions: number of actions in this environment
        """
        self.initial_state = initial_state
        self.trajectory_length = trajectory_length

        if not isinstance(initial_state[0], torch.Tensor):
            raise ValueError("Initial state is not a tensor instance!")

        self.state_shape = initial_state[0].size()
        self.n_actions = n_actions

        # Init arrays
        self.states = torch.zeros((trajectory_length + 1,) + self.state_shape)
        self.rewards = torch.zeros(trajectory_length, 1)
        self.actions = torch.zeros(trajectory_length, self.n_actions)
        self.terms = torch.zeros(trajectory_length + 1, 1, dtype=torch.bool)
        self.truncs = torch.zeros(trajectory_length + 1, 1, dtype=torch.bool)

        self.states[0] = self.initial_state[0]
        self.terms[0] = self.initial_state[1]
        self.truncs[0] = self.initial_state[2]

        self.index = 0

    def submit_data(self, action: torch.Tensor, new_s: torch.Tensor, r, terminal, truncated):
        """
        Submits collected data to the trajectory after a step done by the agent
        :param action: The action done by the agent (as Tensor! Not the environment action)
        :param new_s: New state (converted to Tensor) given by the environment after doing the action
        :param r: The reward given by the environment
        :param terminal: Whether the next state is a terminal state (i.e. the agent died, or reached the finish)
        :param truncated: Whether the environment is being reset from the outside.
            This can be done because of time constraints etc.
        :return: Nothing
        """
        if self.is_full():
            raise IndexError("Trajectory is full, not more data can be added! Create a new one instead.")

        self.actions[self.index] = action
        self.states[self.index + 1] = new_s
        self.rewards[self.index] = r
        self.terms[self.index + 1] = terminal
        self.truncs[self.index + 1] = truncated

        self.index += 1

    def submit_reset_data(self, new_s: torch.Tensor):
        """
        Submit data after reset
        :param new_s: new state
        """
        if self.is_full():
            raise IndexError("Trajectory is full, not more data can be added! Create a new one instead.")

        self.states[self.index + 1] = new_s
        self.index += 1

    def get_final_state(self) -> tuple:
        """
        :return: The final state tuple (s, term, trunc), to be given to the next trajectory as initial state.
        """
        return self.states[-1], self.terms[-1], self.truncs[-1]

    def is_full(self) -> bool:
        return self.index >= self.trajectory_length

