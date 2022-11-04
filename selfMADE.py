"""Masked autoregressive network for indistinguishable particles"""
#
# References:
# -----------
# [1] M. Germain, K. Gregor, I. Murray and H. Larochelle, Made: Masked autoencoder for 
#     distribution estimation, PMLR vol. 37, pp. 881-889
#         http://proceedings.mlr.press/v37/germain15.html
# [2] D. Wu, L. Wang and P. Zhang, Solving statistical mechanics using variational 
#     autoregressive networks, Phys. Rev. Lett. 122, 080602 (2019)
#

import torch
import numpy as np

import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical

from utils import default_dtype_torch
from one_hot import occ_numbers_collapse, occ_numbers_unfold
from bitcoding import int2bin

__all__ = ['selfMADE']

torch.set_default_dtype(default_dtype_torch)

class MaskedLinear(torch.nn.Linear):
    """Ensures autoregressive property of the connectivity matrix"""
    def __init__(self, in_size, out_size, direct_connection=True, bias=True):
        super(MaskedLinear, self).__init__(in_features=in_size, out_features=out_size, bias=bias)
        self.register_buffer('mask', torch.ones(in_size, out_size))
        if not direct_connection:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)
        self.weight.data *= self.mask
        
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MaskedLinear_unfolded(torch.nn.Linear):
    """rewrite MaskedLinear for one-hot encoding of occupation number states"""
    def __init__(self, in_size, out_size, direct_connection=True, bias=True, unfold_size=1):
        super(MaskedLinear_unfolded, self).__init__(in_features=in_size*unfold_size, 
                                                    out_features=out_size*unfold_size, bias=bias)
        self.register_buffer('mask', torch.ones(in_size*unfold_size, out_size*unfold_size))
        mask_ = torch.ones(unfold_size, unfold_size)
        if unfold_size > 1:
            if not direct_connection:
                mask_ = 1 - torch.triu(mask_)
            else:
                mask_ = torch.tril(mask_)            

        mask_ = np.kron(mask_, np.ones((in_size, out_size))) # torch.kron available in newer version of torch
        self.mask = torch.from_numpy(mask_).to(default_dtype_torch) 
        self.weight.data *= self.mask
        
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)        


class Softmax_blocked(torch.nn.Module):
    """
        Apply softmax to each block of outputs representing the probabilities 
        for placing the k-th particle. 
    """
    def __init__(self, D, num_components):
        super(Softmax_blocked, self).__init__()
        self.D = D
        self.num_components = num_components
        self.m = torch.nn.Softmax(dim=1)
        self.register_buffer('Pauli_blocker', torch.zeros(self.num_components, self.D))

    def forward(self, x):
        """
            Actually, when sampling the positions for the k-th particle, the softmax only needs 
            to be applied to the k-th block, since the other blocks play no role. 

            NOTE: THE PAULI BLOCKER LAYER DEPENDS ON THE SAMPLE, IT CANNOT BE USED WITH A 
            BATCH OF SAMPLES !!!!!
        """
        x_out_ = torch.zeros_like(x)  
        for k in range(self.num_components):
            x_out_[..., k*self.D:(k+1)*self.D] = self.m(x[..., k*self.D:(k+1)*self.D]
                                            + self.Pauli_blocker[k,:])                                                                                  
        return x_out_


class selfMADE(torch.nn.Module):
    """
        Neural network composed of autoregressive layers.
        
        The input of the network has been adapted for the purpose of componentwise 
        direct sampling of particle *positions* (as opposed to componentwise direct 
        sampling of *occupation numbers*). Therefore, the components of the sampling 
        process are the (distinguishable particles, which are ordered according to some
        scheme) and the number of components is given by the number of particles (rather 
        than the number of lattice sites). 
 
           The only difference to the MADE neural network (see references below)
           is that here binary input pixels are sort-of 'one-hot' encoded. This means that 
           the input vector of a binary image (e.g. a digit from the MNIST database) 
           consisting of `D` binary pixels is unfolded into a larger binary vector of size
           `D * num_components` which is the concatenation of subarrays of size `D` each. 
           In the first subarray of length `D` the position of the first particle is set 
           with all other positions unset. In the second subarray the position of the second
           particle is set in addition to the position of the first particle, and so on. 
           Obviously, this encoding is redundant. 

        The non-linearities on each output node are parametrized rectified 
        linear units ('PReLU') with a learnable parameter, which - like the weight-matrix 
        of each layer - is also optimized during the training. 

        Parameters:
        -----------
            D: int 
                Dimension of the single-particle Hilbert space, i.e. 
                typically the number of sites. 

            num_components: int 
                Number of particles (canonical ensemble).

            bias_zeroth_component: 1-dim ndarray of length `D`
                Probability distribution for the position of the first particle. 
                For a homogeneous system, this distribution is constant of value 
                1/`D` (default). For a non-translationally invariant systems, it 
                corresponds to the average particle density at each position. 

            net_depth: int >= 2
                The number of layers in the neural network.
                Even if the specified `net_depth` is < 2, there 
                will be at least two autoregressive layers in the neural network.

            indep_PReLU: boolean
               Whether all 'PReLU's in a given layer share the same learnable parameter or have 
               independent parameters. Default is the latter, i.e. `indep_PReLU` = True.
    """
    def __init__(self, **kwargs):
        super(selfMADE, self).__init__()
        self.epsilon = 1e-7 # to avoid overflow in cross-entropy loss
        self.D = kwargs['D']
        self.num_components = kwargs['num_components']
        self.bias_zeroth_component = kwargs.get('bias_zeroth_component', [1.0/self.D] * self.D)
        assert 1 <= self.num_components <= self.D
        self.net_depth = kwargs['net_depth']
        self.indep_PReLU = kwargs.get('indep_PReLU', True)
 
        layers = []
        layers.append(MaskedLinear_unfolded(
            in_size=self.D, out_size=self.D, direct_connection=False, bias=False,
            unfold_size=self.num_components)
                     )
        for _ in range(self.net_depth-2):
            layers.extend([
                torch.nn.PReLU(num_parameters=self.D*self.num_components if self.indep_PReLU else 1, init=0.5),
                MaskedLinear_unfolded(self.D, self.D, direct_connection=True, bias=False, 
                                    unfold_size=self.num_components)
            ])
        layers.extend([
            torch.nn.PReLU(num_parameters=self.D*self.num_components if self.indep_PReLU else 1, init=0.5),
            MaskedLinear_unfolded(self.D, self.D, direct_connection=True, bias=False, 
                                  unfold_size=self.num_components),                                                                                                     
            Softmax_blocked(D=self.D, num_components=self.num_components) # output of each block interpretable as probabilities
        ])
        self.net = torch.nn.Sequential(*layers)
        
        assert type(self.net[-1]) == Softmax_blocked, "Expected `Softmax_blocked` as last network layer."
        # Pauli-blocker for the first component -> take log, because this will be passed through Softmax
        self.net[-1].Pauli_blocker[0, 0:self.D-self.num_components+1] = torch.log(torch.Tensor(self.bias_zeroth_component)[0:self.D-self.num_components+1]) # in combination with a slater sampler the bias should be a uniform distribution (which is the default, anyways)
        self.net[-1].Pauli_blocker[0, self.D-self.num_components+1:] = torch.tensor([float('-inf')], requires_grad=False)
        
    def forward(self, x):
        x_hat = self.net(x)
        return x_hat

    def sample_unfolded(self, seed=None):
        if seed:
            torch.manual_seed(seed)
        with torch.no_grad():
            x_out = occ_numbers_unfold(sample=torch.zeros(self.D).unsqueeze(dim=0), unfold_size=self.num_components,
                duplicate_entries=False)
            pos_one_hot = torch.zeros(self.D).unsqueeze(dim=0)

            # The particle sampled first should be on the leftmost position. 
            # Pauli_blocker is part of Softmax layer. 
            # x_hat_bias is not affected by the Pauli blocker in the Softmax layer 
            # and need to be taken care of explicitly. 
            for i in range(0, self.num_components):
                x_hat = self.forward(x_out) 

                # Make sure that an ordering of the particles is enforced.
                # The position of particle i, k_i, is always to the left of k_{i+1},
                # i.e. k_i < k_{i+1}.
                probs = x_hat[:,i*self.D:(i+1)*self.D]
                pos_one_hot = OneHotCategorical( probs ).sample()
                k_i = torch.nonzero(pos_one_hot[0])[0][0]
                x_out[:,i*self.D:(i+1)*self.D] = pos_one_hot

                if i < self.num_components-1:
                    self.net[-1].Pauli_blocker[i+1,:] = 0.0
                    self.net[-1].Pauli_blocker[i+1,:k_i+1] = torch.tensor([float('-inf')]) 
                    self.net[-1].Pauli_blocker[i+1,self.D-self.num_components+(i+1)+1:] = torch.tensor([float('-inf')])
    
        # reset (to be sure)
        # Don't reset the zero-th component, it has been set in __init__().
        self.net[-1].Pauli_blocker[1:,:] = 0.0

        return x_out

    def log_prob(self, samples):
        """
            Wavefunction amplitude. This method has the same signature as the 
            function `psi_amplitude` of the Slater-Jastrow type ansatz.

            For testing purposes only. 
        """
        samples = torch.as_tensor(samples)
        samples_unfold = occ_numbers_unfold(samples, duplicate_entries=False)
        # Flatten leading dimensions (This is necessary since there may a batch of samples 
        # for several "connecting states", but forward() accepts only one batch dimension.)   
        samples_unfold_flat = samples_unfold.view(-1, samples_unfold.shape[-1])
    
        x_hat = self.forward(samples_unfold_flat)
        mm = x_hat * samples_unfold_flat # Pick only the probabilities at actually sampled positions !
        ones = torch.ones(*mm.shape)        
        log_prob = torch.log(torch.where(mm > 0, mm, ones)).sum(dim=-1)
        # reshape leading dimensions back to original shape (last dim is missing now)
        return log_prob.view(*samples.shape[:-1])
    

    def psi_amplitude(self, samples):
        """
            Wavefunction amplitude. This method has the same signature as the 
            function `psi_amplitude` of the Slater-Jastrow type ansatz.
        """
        samples = torch.as_tensor(samples)
        samples_unfold = occ_numbers_unfold(samples)
        pick_pos = occ_numbers_unfold(samples, duplicate_entries=False)        
        # Flatten leading dimensions (This is necessary since there may a batch of samples 
        # for several "connecting states", but forward() accepts only one batch dimension.)   
        samples_unfold_flat = samples_unfold.view(-1, samples_unfold.shape[-1])
        pick_pos_flat = pick_pos.view(-1, pick_pos.shape[-1])

        x_hat = self.forward(samples_unfold_flat)
        mm = x_hat * pick_pos_flat # Pick only the probabilities at actually sampled positions !
        ones = torch.ones(*mm.shape)        
        amp = torch.sqrt(torch.where(mm > 0, mm, ones).prod(dim=-1))
        # reshape leading dimensions back to original shape (last dim is missing now)
        return amp.view(*samples.shape[:-1])
    
    def psi_amplitude_I(self, samples_I):
        """
            Wavefunction amplitude. Inputs are bitcoded integers. 
        """
        samples_I = torch.as_tensor(samples_I)
        assert len(samples_I.shape) >= 1, "Input should be bitcoded integer (with at least one batch dim.)."
        samples = int2bin(samples_I, self.D)
        return self.psi_amplitude(samples)


    def _cross_entropy(self, sample, x_hat):
        # Cross entropy for binary observations: 
        # Sample values are zero or one, so only one of the two terms in the sum is non-zero.
        ce = ( 
            - sample*torch.log(x_hat + self.epsilon) 
            - (1 - sample)*torch.log(1 - x_hat + self.epsilon)
             )

        # Cross entropy for all samples returned as a vector (not averaged yet).
        return ce.view(ce.shape[0], -1).sum(dim=-1)


    def cross_entropy(self, sample):
        x_hat = self.forward(sample)
        assert(x_hat.requires_grad)
        return self._cross_entropy(sample, x_hat)


def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
