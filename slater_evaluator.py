
#!/usr/bin/python3.5
# TODO:
#  - move particles only to neighbouring sites
#    and do not swap particles
#  - implement all sparse matrix-matrix multiplications
#  - store positions of the particles and carry an occupation vector
#    which is updated together with the Slater determinant
#
"""
    Evaluate the ratio of Slater determinants which differ by moving a small 
    number of particles using low-rank update. 
"""

import numpy as np
import numpy.random
from scipy import linalg
from profilehooks import profile


class OneBodyDensityMatrix(object):
    """
        Structure containing the current one-body density matrix (OBDM)
        < alpha | c_i^{+} c_j | psi >  (i,j = 0,...,Ns-1) for Fock state alpha.
        We separately store the matrices 'LRinv_timesL' and 'R' because only
        the first changes when updating the Fock state whereas the latter does
        not. 'LRinv_timesL' is updated by left-multiplication with a dense
        matrix and by right-multiplication with a sparse matrix.
    """
    def __init__(self, Ns, Np):
        # the actual one-body density matrix
        self.matrix = np.zeros((Ns,Ns), dtype=np.float64)
        # auxiliary matrices which are stored in memory for an efficient update
        # of the current OBDM.
        self.LRinv_timesL = np.zeros((Np, Ns), dtype=np.float64)
        self.R = np.zeros((Ns, Np), dtype=np.float64)


class SlaterDetEvaluator(object):
    """provides two functionalities:
          - evaluate overlap between a Fock state and a Slater determinant
          - ratio of overlaps if a particle is moved from site r to site s 
    """
    
    def __init__(self, occ_vec, psi):
        """ Args:
            alpha (matrix) - Slater determinant representing a Fock state
            psi (matrix) - Slater determinant which is to be sampled
            
            Initializes OBDM object. 
        """
        self._make_spOBDM_from_occ_vec(occ_vec, psi)
        
    def _make_spOBDM(self, alpha, psi):
        """
            Build the one-body density matrix
                    OBDM_ij = < alpha | c_i^{+} c_j | psi >  (i,j = 0,...,Ns-1)
            from the Slater determinant | alpha > and | psi >.
            Input:
            |psi>: The Slater determinant wave function to be sampled,
                represented by an Ns x Np rectangular matrix, where Ns is the
                number of orbitals and Np is the number of fermions.
            |alpha>: Slater determinant of a Fock state, represented
                    by an Ns x Np rectangular matrix.
            Output:
            One-body density matrix.
        """
        assert (alpha.shape == psi.shape)
        (Ns, Np) = alpha.shape
        self.Ns = Ns
        self.Np = Np
        assert (Ns >= Np)
        # create OBDM object
        self.OBDM = OneBodyDensityMatrix(Ns, Np)
        L = np.array(alpha).transpose()
        R = np.array(psi)
        LRinv_timesL = np.matmul(linalg.inv(np.matmul(L,R)), L)

        self.OBDM.matrix = np.matmul(R, LRinv_timesL).transpose()
        self.OBDM.LRinv_timesL = LRinv_timesL
        self.OBDM.R = R
        
    def _make_spOBDM_from_occ_vec(self, occ_vec, P):
        """
        Args:
           occ_vec (1darray) - vector of occupation numbers, e.g. [0, 1, 0, 1, 1, 0]
           P (matrix) - P-matrix representation of Slater determinant, (D x N)-matrix
           
           The particle positions are ordered. 
        """
        occ_vec = np.array(occ_vec)
        Ns = len(occ_vec)
        pos = np.nonzero(occ_vec)[0]
        Np = len(pos)
        alpha = np.zeros((Ns, Np))
        for j in np.arange(Np):
           alpha[pos[j], j] = 1 
        
        self._make_spOBDM(alpha, P)
        self.P = P   
        
    def _update_spOBDM(self, r, s):
        """
            Update the OBDM after moving a fermion from site r to site s.
        """
        LL = self.OBDM.LRinv_timesL
        RR = self.OBDM.R
        (Ns, Np) = RR.shape # Ns = number of orbitals; Np = number of particles

        # correction matrices
        U = np.zeros((Np,4), dtype=np.float64)
        V = np.zeros((4,Np), dtype=np.float64)
        U[:,0] = LL[:,r]; U[:,1] = LL[:,s]; U[:,2] = U[:,0]; U[:,3] = U[:,1]
        V[0,:] = +RR[r,:]; V[1,:] = +RR[s,:]; V[2,:] = -RR[s,:]; V[3,:] = -RR[r,:]
        Delta = np.zeros((Ns,Ns), dtype=np.float64)
        Delta[r,r] = Delta[s,s] = 1.0; Delta[r,s] = Delta[s,r] = -1.0

        # Sherman-Morrison formula for updating R(LR)^{-1}L
        tmp1 = linalg.inv(np.eye(4) - np.matmul(V,U))
        tmp2 = np.matmul(np.matmul(U, tmp1), V)

        # dense matrix multiplication from the left
        C_dense = np.eye(Np) + tmp2
        tmp3 = np.matmul(C_dense, LL)
        # sparse matrix multiplication from the right
        # (IMPROVE  by avoiding full matrix multiplication)
        C_sparse = np.eye(Ns) - Delta
        self.OBDM.LRinv_timesL = np.matmul(tmp3, C_sparse)
        self.OBDM.matrix = np.matmul(RR, self.OBDM.LRinv_timesL).transpose()
        # OBDM.R does not change        
                
                
    # In VMC. the amplitude ratio is needed only when computing the local energy (or local off-diag. observables).
    def determinant_ratio(self, r, s):
        """
            Ratio of wavefunction amplitudes of the Slater determinant where a particle has
            been moved from site r to site s divided by the current Slater determinant.
            Uses low-rank update. 
            It is assumed that the OBDM corresponding to the current Slater determinant 
            is in memory.
        """
        # TODO: assert for excluding illegal hoppings 
        G = self.OBDM.matrix.transpose()
        # determinant ratio for the move (r -> s)
        ratio = (1.0 - (G[r,r] - G[s,r]))*(1.0 - (G[s,s] - G[r,s])) \
                - (G[r,r] - G[s,r])*(G[s,s] - G[r,s])
        return ratio       
    
    
    