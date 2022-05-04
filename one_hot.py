import torch
import numpy as np

from utils import default_dtype_torch
from bitcoding import bin2pos

def occ_numbers_unfold(sample, unfold_size=None, duplicate_entries=False):
    """ 
        First dimension(s) is (are) the batch dimension(s).

        Arguments:
            sample: 1d ndarray of zeros and ones (occupation numbers)
                of length `D`
            unfold_size: int < `D`
            duplicate_entries: whether to repeat positions of particles 
                sampled in the previous steps of the componentwise sampling.

        Returns:
            v: 1d ndarray of length D*Np where Np is the number of ones 
               in `sample` or it is equal to `unfold_size` if this parameter
               is given. 

        Note: When unfolding occupation number states, there is an ordering of 
              the particle positions `k_i` implied:
              The position of the i-th particle, k_i, is to the left 
              of the i+1-th particle: k_i < k_{i+1}. 

        Example:
        --------
        >>> sample = [[[0,1,0,1],[0,1,1,0],[0,0,1,1]],[[0,1,0,1],[0,1,1,0],[0,0,1,1]]]
        >>> occ_numbers_unfold(sample, duplicate_entries=True)
        tensor([[[0., 1., 0., 0., 0., 1., 0., 1.],
                 [0., 1., 0., 0., 0., 1., 1., 0.],
                 [0., 0., 1., 0., 0., 0., 1., 1.]],
        <BLANKLINE>
                [[0., 1., 0., 0., 0., 1., 0., 1.],
                 [0., 1., 0., 0., 0., 1., 1., 0.],
                 [0., 0., 1., 0., 0., 0., 1., 1.]]], dtype=torch.float64)
        >>> occ_numbers_unfold(sample, duplicate_entries=False) 
        tensor([[[0., 1., 0., 0., 0., 0., 0., 1.],
                 [0., 1., 0., 0., 0., 0., 1., 0.],
                 [0., 0., 1., 0., 0., 0., 0., 1.]],
        <BLANKLINE>
                [[0., 1., 0., 0., 0., 0., 0., 1.],
                 [0., 1., 0., 0., 0., 0., 1., 0.],
                 [0., 0., 1., 0., 0., 0., 0., 1.]]], dtype=torch.float64)

                       
    """
    sample = np.array(sample)
    assert(len(sample.shape)>=2), "Input should have at least one batch dimension."
    D = sample.shape[-1]
        
    if unfold_size:
        assert unfold_size <= D
        Np = unfold_size
    else:
        # All samples should have the same number of particles. 
        Np_ = np.count_nonzero(sample, axis=-1)
        Np = Np_[np.ndindex(Np_.shape).__next__()]
        assert np.allclose(Np_, Np) 

    v = np.zeros(sample.shape[:-1] + (Np*D,))
    positions = bin2pos(sample)

    for k in range(Np):
        if duplicate_entries:
            put_here = positions[..., :k+1]
        else:
            put_here = positions[..., k:k+1]      
        view = v[..., k*D:(k+1)*D]
        np.put_along_axis(view, put_here, values=1, axis=-1)

    return torch.Tensor(v).to(default_dtype_torch)



def occ_numbers_collapse(v, D, duplicate_entries=False): 
    """ 
        First dimension(s) is (are) the batch dimension(s).

        Arguments:
            v: 1d array of length D*Np where D is the dimension of the single-
               particle Hilbert space and Np < D is the number of particles.  
        
        Returns:
            sample: 1d array of zeros and ones (occupation numbers)
                of length D.   
            First dimension is the batch dimension. 

        Example:
        --------
        >>> sample = torch.Tensor([[[1,0,0,0,1,0,1,0],[1,0,0,0,1,1,0,0],[0,1,0,0,0,1,1,0]],[[1,0,0,0,1,0,1,0],[1,0,0,0,1,1,0,0],[0,1,0,0,0,1,1,0]]])
        >>> occ_numbers_collapse(v=sample, D=4, duplicate_entries=True)
        tensor([[[1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 1., 1., 0.]],
        <BLANKLINE>
                [[1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 1., 1., 0.]]])
        >>> sample = torch.tensor([[[1,0,0,0,0,1,0,0],[1,0,0,0,0,1,0,0],[0,1,0,0,0,0,1,0]],[[1,0,0,0,0,0,1,0],[1,0,0,0,0,1,0,0],[0,1,0,0,0,0,1,0]]])
        >>> occ_numbers_collapse(v=sample, D=4, duplicate_entries=False)
        tensor([[[1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 1, 1, 0]],
        <BLANKLINE>
                [[1, 0, 1, 0],
                 [1, 1, 0, 0],
                 [0, 1, 1, 0]]])

    """
    if duplicate_entries:
        last = v[..., -D:] 
        # Check that all samples contain the same number of particles.
        Np_ = np.count_nonzero(last, axis=-1)
        Np = Np_[np.ndindex(Np_.shape).__next__()] # IMPROVE: find less complicated expression 
        assert np.all(Np_ == Np)
        return last 
    else:
        # improve this later 
        Np =  v.shape[-1] // D     
        v_np = np.empty(v.shape[:], dtype=object)        
        v_np = v.numpy()
        out_np = np.add.reduce([v_np[..., k*D:(k+1)*D] for k in range(Np)], axis=0) # "Creating an ndarray from ragged nested sequences is deprecated."
        assert np.all(np.logical_or(out_np == 1, out_np == 0).flatten()) # only one particle per site
        assert np.all(np.sum(out_np, axis=-1) == Np) # all samples have the same number of particles 
        return torch.tensor(out_np)

def _test():
    import doctest 
    doctest.testmod(verbose=True)

if __name__ == '__main__':
    _test()
