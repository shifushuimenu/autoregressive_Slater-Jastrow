from sr_lazy import SR_Preconditioner_base
import torch 
import numpy as np

class SR_Preconditioner(SR_Preconditioner_base):
    """This subclass provides the interface to gradients computed in torch."""
    def __init__(self, num_params, num_samples):
        super(SR_Preconditioner_base, self).__init__(num_params, num_samples)

    def _flatten(self, grad):
        """The gradient (and also O_k)  is a list of torch tensor matrices and vectors (same as module.parameters()).
        Flatten it into a single continuous vector. Input `grad`is a numpy array. """
        grad_flat = np.hstack([p.flatten() for p in grad])
        return grad_flat 

    def _unflatten(self, grad, grad_flat):
        """Use information in `grad` to reassemble flattened gradient into blocks of parameters."""
        offset = 0
        grad_new = []
        for ps in [list(p.size()) for p in grad]:
            param_block = grad_flat[offset:offset+np.prod(ps)].reshape(ps)
            grad_new.append(torch.tensor(param_block))
            offset += np.prod(ps)
        return grad_new   