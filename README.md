# Reinforcement Learning Algorithms in PyTorch
Collection of RL algorithms for personal use.

## Components:
This is a work in progress module. Current supported modules are:

### Simple AC
Simple Actor Critic implementation that has no RNN support and only uses a (partial) Beta distribution as action space.
Value estimation is done using TD-0 (i.e. target $V(s_t)$ is $r + \gamma V(s_(t+1))$)
