from copy import deepcopy

import torch.nn


class PolicyNet(torch.nn.Module):
    def __init__(self, n_inputs, hidden_layer_size,
                 fix_for_n_training_steps=5, lr=0.0003, batch_size=64, n_outputs=1, clip_beta=True, entropy_factor=0.0):
        super().__init__()

        self.fix_for_n_training_steps = fix_for_n_training_steps
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        # Clips the Beta distribution to alpha and beta >= 1.
        # Smaller values allow the model to create U-shaped distributions, which is often not very great.
        self.clip_beta = clip_beta
        self.entropy_factor = entropy_factor

        self.current_train_step = 0

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.Tanh(),

            torch.nn.Linear(hidden_layer_size, n_outputs * 2),
        )

        self.model_fixed = deepcopy(self.model)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, states):
        pred = self.model(states)
        pred = torch.nn.functional.softplus(pred)

        if self.clip_beta:
            pred += 1.0

        if len(pred.size()) == 2:
            distribution = torch.distributions.Beta(pred[:, :self.n_outputs], pred[:, self.n_outputs:])
        elif len(pred.size()) == 3:
            distribution = torch.distributions.Beta(pred[:, :, :self.n_outputs], pred[:, :, self.n_outputs:])
        else:
            raise ValueError("Cannot handle prediction of shape ", pred.size())
        return distribution

    def get_action(self, state):
        distribution = self(state.view((-1,) + state.size()))
        return distribution.sample()[0]

    def training_step(self, states, actions, dones, advantages):
        n_trajectories = states.size(0)
        trajectory_len = states.size(1) - 1
        n_samples = n_trajectories * trajectory_len
        if not (n_trajectories == advantages.size(0) and trajectory_len == advantages.size(1)):
            raise ValueError("Expected n_trajectories: %d and trajectory_len: %d, but got advantages of size %s" %
                             (n_trajectories, trajectory_len, str(advantages.size())))

        advantages = advantages.view((n_trajectories * trajectory_len, 1))
        actions = actions.view((n_trajectories * trajectory_len, self.n_outputs))
        small_states = states[:, :-1].contiguous()
        states = small_states.view(*((n_trajectories * trajectory_len,) + states.size()[2:]))

        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, self.batch_size):
            self.optim.zero_grad()
            start, stop = i, i + self.batch_size
            batch_indices = indices[start:stop]
            dist = self(states[batch_indices])
            loss_batch = -(dist.log_prob(actions[batch_indices]) * advantages[batch_indices])
            if self.entropy_factor != 0.0:
                loss_batch -= self.entropy_factor * dist.entropy()
            loss_batch[dones.view(-1)[batch_indices]] = 0.0
            loss = loss_batch.sum()
            loss.backward()
            self.optim.step()

        self.current_train_step += 1
        if self.current_train_step % self.fix_for_n_training_steps == 0:
            self.model_fixed.load_state_dict(self.model.state_dict())
