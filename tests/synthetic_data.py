import torch 
import numpy as np

from torch.distributions.categorical import Categorical
from one_hot import occ_numbers_unfold
from bitcoding import int2bin

__all__ = ["Data_dist"]

class Data_dist(object):
    """
        Generate a synthetic data distribution of bitstrings for testing purposes.
        
        If `Nparticles` is None, then the bitstrings are drawn from a grand-canonical
        ensemble. 
        
        If `Nparticles` is not None, then the bitstrings are drawn from a
        'canonical ensemble' where the number of ones in each bitstring is guaranteed 
        to be equal to `Nparticles`.
    """    
    def __init__(self, Nsites, Nparticles=None, seed=42):
        if seed:
            torch.manual_seed(seed)
        self.epsilon = 1e-7 # avoid overflow logs when calculating entropy
        self.Nsites = Nsites 
        self.dim = 2**self.Nsites
        # probability distribution over all basis states 
        self.probs = torch.rand(self.dim) #torch.Tensor([1.0/(i+1)**1.5 for i in range(self.dim)])  #torch.rand(self.dim)         
        if not Nparticles:
            # grand-canonical 
            self.unfold_size = self.Nsites   
            # canonical 
        else:
            assert isinstance(Nparticles, int) and isinstance(Nsites, int) and Nparticles <= Nsites 
            self.Nparticles = Nparticles
            self.unfold_size = self.Nparticles
            cond = torch.Tensor([np.count_nonzero(int2bin(x, ns=self.Nsites))==self.Nparticles for x in range(self.dim)])
            self.probs = self.probs * cond
                
        self.probs /= self.probs.sum(dim=-1)
        self.sampler = Categorical(self.probs) 
        
    def sample(self, rep='binary'):
        """Generate one sample at a time."""
        # IMPROVE: output a batch
        assert rep in ['binary', 'integer', 'both']
        s = self.sampler.sample()
        if rep == 'binary':
            return torch.tensor(int2bin(s, ns=self.Nsites))
        elif rep == 'integer':
            return s
        elif rep == 'both':
            return torch.tensor(int2bin(s, ns=self.Nsites)), s
    
    def sample_unfolded(self):
        s_ = self.sample(rep='binary')
        s_ = s_.unsqueeze(dim=0) #
        return occ_numbers_unfold(sample=s_, unfold_size=self.unfold_size)
    
    def generate_batch_v0(self, batch_size=128, unfold=False):
        """Generate a batch of samples."""
        if unfold:
            batch = torch.zeros(batch_size, self.Nsites*self.unfold_size)
            for i in range(batch_size):
                batch[i,:] = self.sample_unfolded()
        else:
            batch = torch.zeros(batch_size, self.Nsites)
            for i in range(batch_size):
                batch[i,:] = self.sample()
        return batch

    def generate_batch(self, batch_size=128, unfold=False):
        """Generate a batch of samples."""
        batch = torch.zeros(batch_size, self.Nsites)
        for i in range(batch_size):
            batch[i,:] = self.sample()
        if unfold:
            return occ_numbers_unfold(batch, unfold_size=self.unfold_size)
        else:
            return batch         
    
    def entropy(self):
        """Entropy of the synthetic data distribution."""
        entropy = - self.probs * torch.log(self.probs + self.epsilon)
        return entropy.sum(dim=-1)
    
    @classmethod
    def calc_entropy(self, probs):
        """Calculated entropy of a histogram."""
        entropy = - probs * torch.log(probs + 1e-7)
        return entropy.sum(dim=-1)

    def _int2bits(self, x):
        x_ = x
        bits = torch.zeros(self.Nsites)
        for i in range(self.Nsites-1, -1, -1):
            bits[self.Nsites - 1 - i] = x_ // 2**i 
            x_ = x_ - bits[self.Nsites - 1 - i]*2**i
        return bits
        
    @classmethod 
    def bits2int(self, bits):
        """Convert torch tensor of 0s and 1s to its integer representation."""
        bits = torch.Tensor(bits)
        bits = torch.squeeze(bits, dim=0) # implement this as a decorator function ! 
        x = torch.Tensor([0])
        m = len(bits)
        for i in range(m-1, -1, -1):
            x += bits[m -1 - i]*2**i
        return x.to(torch.int64)
