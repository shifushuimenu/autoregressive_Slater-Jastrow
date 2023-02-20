"""
Routines for dealing with Slater determinants.
"""

import numpy as np
from scipy import linalg
from bitcoding import bin2pos
from functools import reduce 

def Slater_determinant_overlap(alpha, Pmatrix):
    """
        Args:
            alpha (bin array): occupation number state
            Pmatrix: P-matrix representation of the Slater determinant (square array of cols of orthogonal single-particle states)
        Returns:
            Overlap <alpha|SD>

        Note: This routine is slow and should only be used for tests.
    """
    Palpha = np.zeros_like(Pmatrix)
    for col, pos in enumerate(np.nonzero(alpha)[0]):
        Palpha[pos, col] = 1
    return np.linalg.det(np.matmul(Palpha.T, Pmatrix))


def Slater2spOBDM(sp_states):
    """
        <Sdet| c_i c_j^{\dagger} |Sdet> = det([P_i^{\prime}]^t P_j^{\prime})

        where P_j^{prime} is the matrix of single-particle states `sp_states`
        with one column added where all elements are zero except the j-th.
        `sp_states` has shape (Ns, Np), P_j^{\prime} has shape (Ns, Np+1).

        The single-particle states need not necessarily be orthogonal. 
    """
    Ns, Np = sp_states.shape
    assert Ns>=Np

    GF = np.zeros((Ns, Ns))   # Green's function (i,j) = <c_i c_j^{\dagger}>
    OBDM = np.zeros((Ns, Ns)) # OBDM (i,j) = <c_i^{\dagger} c_j>

    for j in range(Ns):
        e_j = np.zeros((Ns,1)); e_j[j,0] = 1.0
        P_j_prime = np.hstack((sp_states, e_j))        
        for i in range(Ns):
            e_i = np.zeros((Ns,1)); e_i[i,0] = 1.0
            P_i_prime = np.hstack((sp_states, e_i))

            GF[i,j] = np.linalg.det(np.matmul(P_i_prime.T, P_j_prime))

    OBDM = np.eye(Ns) - GF.T
    return OBDM


# Same functionality as Slater2spOBDM
def Slater2rdm1(C, orthogonal=True):
    # OBDM = C (C^{\dagger} C)^{-1} C^{\dagger}
    # Columns of C are orbitals.
    (ns, num_particles) = C.shape
    if not orthogonal:
        Minv = np.linalg.inv(np.matmul(C.conj().transpose(), C))
        OBDM = reduce(np.matmul, (C, Minv, C.conj().transpose()))
    else:
        assert np.isclose(np.matmul(C.conj().transpose(), C), np.eye(num_particles)).all()
        OBDM = np.matmul(C, C.conj().transpose())
    return OBDM 


def local_OBDM(alpha, sp_states):
    """
        The 'local one-body density matrix' (OBDM) is basis state |alpha> is defined as:

            OBDM_loc(\alpha)_ji = <\alpha| c_j^{\dagger} c_i |\psi> / <\alpha | \psi>
        
        This corresponds to a term in the local kinetic energy and it can be used for that 
        purpose directly. 
        Furthermore, the local OBDM is needed to calculate the ratio of Slater determinants 

            <\beta | psi > / <\alpha | \psi > 

        where basis state \beta is obtained from \alpha by moving one particle from 
        (occupied) site r to (unoccupied) site s. 

        Input:
        ------
        alpha (array of ints) : array of occupation numbers, e.g. [1,0,1,1,0]
        sp_states (Nsites x Nparticles array) : P-matrix representing Slater determinant 

        Output:
        -------
        local_OBDM (Nsites x Nsites matrix): elements of local kinetic energy 

        Example:
        --------
        >>> (_, U) = prepare_test_system_zeroT(Nsites=4, potential='none', HF=True, Nparticles=2, Vnnint=1.0)
        >>> sp_states = U[:,0:2]
        >>> loc_OBDM = local_OBDM([0,1,1,0], sp_states)
    """
    alpha = np.asarray(alpha)
    assert len(alpha.shape) == 1 # no batch dimension 
    Ns, Np = sp_states.shape
    L_idx = bin2pos(alpha)  # select these cols from P-matrix 
    assert len(L_idx) == Np
    M = np.matmul( sp_states, np.linalg.inv(sp_states[L_idx]) ) 
    GG = np.zeros((Ns, Ns))
    GG[:, L_idx] = M[:,:]
    return GG.T


def ratio_Slater(G, alpha, beta, r, s):
    """ Efficient calculation of the ratio of the overlaps with a Slater determinant. 

            R =  <\beta | psi > / <\alpha | \psi > 

        where basis state \beta is obtained from \alpha by moving one particle from 
        (occupied) site r to (unoccupied) site s. 

        Note that by construction: 
            R(r, s=r) = 1.0  and   R(r,s) = R(s,r) 
        If both r and s are occupied, then R(r,s) = -1. 
        If both r and s are unoccupied, then R(r,s) = +1, as it should be.

        Input:
        ------
         G (Nsites x Nsites array) : local OBDM for state alpha: 
                G_{ji} =  <\alpha| c_j^{\dagger} c_i |\psi> / <\alpha | \psi>
    """
    # IMPROVE: assert that beta is indeed obtained from alpha 
    # alpha = np.asarray(alpha) # alpha is not used 
    G = np.asarray(G)
    beta = np.asarray(beta)
    R = (1 - G[r,r] - G[s,s] + G[r,s] + G[s,r])
    # Furthermore, there is an additional sign factor due to the fact that in the P-matrix
    # representation of a Fock state as a Slater determinant columns need to be ordered according 
    # to increasing row index where the 1's are. 
    sign = np.prod([(-1) if beta[i] == 1 else 1 for i in range(min(s,r)+1, max(s,r))])
    return R * sign


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
