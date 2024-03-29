import numpy as np
from bitcoding import bin2pos, int2bin


def sort_onehop_states(hop_from_to, states_I, matrix_elem):
    """sorts connecting states in increasing order of min(r,s) and eliminates duplicate hopping bonds"""
    idx_sorted = np.argsort([min(*el) for el in hop_from_to])
    hop_from_to = np.array(hop_from_to)[idx_sorted]
    states_I = states_I[idx_sorted]
    matrix_elem = matrix_elem[idx_sorted]
    return (hop_from_to, states_I, matrix_elem)


def calc_k_copy(hop_from_to, ref_state):
    """
    `k_copy` indicates the component up to which (inclusive)
    the conditional probabilities are identical to those 
    of the reference state (such that they can be copied). The index into 
    k_copy is the reference state number.

    For example:
        k_copy = (  0, # The first component is conditionally independent, it can always be copied from the reference state. 
                    1, # copy conditional probs up to (inclusive) the second component 
                    1,
                    2, # copy conditional probs up to (inclusive) the third component
                    2, 
                    3 )        
    >>> from physics import Lattice_rectangular, kinetic_term
    >>> Ns=9; rctl = Lattice_rectangular(3,3); I = 2**8 + 2**4 + 2**1
    >>> hop_from_to, states_I, matrix_elem = kinetic_term(I, rctl)
    >>> hop_from_to, _, _ = sort_onehop_states(hop_from_to, states_I, matrix_elem)
    >>> k_copy = (0, 0, 0, 1, 1, 1, 1, 2, 2, 2)
    >>> k_copy == calc_k_copy(hop_from_to, int2bin(I, ns=Ns))
    True
    """
    pos_ref_state = bin2pos(ref_state)
    num_connecting_states = len(hop_from_to)
    k_copy = np.zeros((num_connecting_states,), dtype=int)

    for i in range(num_connecting_states):
        (r,s) = hop_from_to[i]
        min_rs = min(r,s)
        ii = 0
        for pos in pos_ref_state.flatten()[ii:]:
            # Note: it is assumed that `hop_from_to` is already ordered in 
            # increasing order of min(r,s).
            if min_rs <= pos:
                k_copy[i] = ii
                break
            ii = ii + 1

    assert monotonically_increasing(k_copy)

    return tuple(k_copy)


def monotonically_increasing(y):
    b = True 
    for i in range(len(y)-1):
        b = b and (y[i] <= y[i+1])
        if not b: return b 
    return b 


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
