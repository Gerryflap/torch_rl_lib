import torch.nn


class ModelSequenceCompatibilityWrapper(torch.nn.Module):
    """
        Wrapper to support models that do not accept inputs with a sequence dimension.
        Given an input (N, T, ...), where N is batch_size and T is the sequence length,
            this wrapper will present the input as (N * T, ...) to the wrapped module,
            then transform the output back to (N, T, ...).
    """

    def __init__(self, preferred_dims, module: torch.nn.Module):
        """
        Initializes the wrapper
        :param preferred_dims: Number of dims the wrapped module wants.
            Any input with 1 more dim is assumed to contain a sequence dim
        :param module: The module to wrap
        """
        super().__init__()
        self.mod = module
        self.preferred_dims = preferred_dims

    def forward(self, inp):
        if inp.dim() == self.preferred_dims:
            return self.mod(inp)
        elif inp.dim() == self.preferred_dims + 1:
            batch_size = inp.size(0)
            sequence_length = inp.size(1)
            outp = self.mod(inp.view((batch_size * sequence_length,) + inp.size()[2:]))
            return outp.view((batch_size, sequence_length) + outp.size()[1:])
        else:
            raise ValueError("Expected input with %d or %d dims, but got size: %s" % (
                self.preferred_dims, self.preferred_dims + 1, inp.size()))
