"""low-rank update of the determinant of numerator and denominator matrices (used 
in the algorithm for ordered componentwise direct sampling from a Slater determinant"""
#IMPROVE: The functions removeadd_rs(), remove_r(), etc. can have division 
# by zero. This gives only a runtime warning, but it should be dealt with.
import numpy as np

from block_update_numpy import ( block_update_inverse,
                           block_update_det_correction,
                           block_update_inverse2,
                           block_update_det_correction2 )


def corr_factor_removeadd_rs(Ainv, r, s):
    """ 
    Correction factor to 
        det(G_{K,K} - N_K)
    with N_K = (n_0, n_1, ..., n_{K-1}) where n_r = 1 and n_s = 0
    and a particle is moved such that after the update n_r = 0 and n_s = 1.         
    """
    return (1 + Ainv[r,r]) * (1 - Ainv[s,s]) + Ainv[r,s] * Ainv[s,r]


def removeadd_rs(Gnum_inv, Gdenom_inv, r, s):
    """
    Correction factor to the ratio of numerator and denominator matrices,
    i.e. the conditional probability, after removing a particle at position `r`
    and adding a particle at position `s`.
    """
    corr_factor_Gnum = corr_factor_removeadd_rs(Gnum_inv, r=r, s=s)
    corr_factor_Gdenom = corr_factor_removeadd_rs(Gdenom_inv, r=r, s=s)
    return corr_factor_Gnum / corr_factor_Gdenom


def remove_r(Gnum_inv, Gdenom_inv, r):
    """
    Correction factor to the ratio of numerator and denominator matrices,
    i.e. the conditional probability, after removing a particle at position `r`.    
    """
    corr_factor_Gnum = corr_factor_remove_r(Gnum_inv, r=r)
    corr_factor_Gdenom = corr_factor_remove_r(Gdenom_inv, r=r)
    return corr_factor_Gnum / corr_factor_Gdenom    


def corr_factor_add_s(Ainv, s):
    """add a particle at position s without removing any particle"""
    return (1 - Ainv[s,s])


def corr_factor_remove_r(Ainv, r):
    """remove a particle at position r without adding any particle"""
    return (1 + Ainv[r,r])


def adapt_Gdenom(Gnum, r, s):
    assert s > r 
    G = Gnum[np.ix_(list(range(0, s+1)), list(range(0, s+1)))]
    G[r,r] = G[r,r] + 1
    G[s,s] = G[s,s] - 1
    return G

# NOT NEEDED ANYMORE 
def adapt_Ainv(Ainv, Gglobal, r, s, i_start, i_end):
    """
    Extend inverse of numerator or denominator matrix from position `i_start` (inclusive)
    up to position `i_end` (inclusive) and put a particle at position `i_end`.
    """
    #assert r == 0 # r != 0 in case of long-range hopping in 1D 
    assert Ainv.shape == (i_start, i_start), "Ainv.shape=(%d, %d) and i_start = %d" % (*Ainv.shape, i_start)
    assert i_start <= i_end

    # put a particle at position `i_end`
    l = i_end+1-i_start
    DD = np.zeros((l, l))
    DD[-1, -1] = -1

    Ainv_extended = block_update_inverse2(Ainv=Ainv, B=Gglobal[0:i_start, i_start:i_end+1], 
        C=Gglobal[i_start:i_end+1, 0:i_start], D=Gglobal[i_start:i_end+1, i_start:i_end+1] + DD)
    corr = block_update_det_correction2(
        Ainv=Ainv, B=Gglobal[0:i_start, i_start:i_end+1], 
        C=Gglobal[i_start:i_end+1, 0:i_start], D=Gglobal[i_start:i_end+1, i_start:i_end+1] + DD
        )
    return Ainv_extended, corr
# NOT NEEDED ANYMORE     
    

def corr_factor_Gdenom_from_Ainv(Ainv, Gglobal, r, s, i_start, i_end):
    """
    Extend inverse of numerator or denominator matrix from position `i_start` (inclusive)
    up to position `i_end` (inclusive) and put a particle at position `i_end`.
    """
    #assert r == 0 # r != 0 in case of long-range hopping in 1D 
    assert Ainv.shape == (i_start, i_start), "Ainv.shape=(%d, %d) and i_start = %d" % (*Ainv.shape, i_start)
    assert i_start <= i_end

    # put a particle at position `i_end`
    l = i_end+1-i_start
    DD = np.zeros((l, l))
    DD[-1, -1] = -1    

    corr_factor = block_update_det_correction2(
        Ainv=Ainv, B=Gglobal[0:i_start, i_start:i_end+1], 
        C=Gglobal[i_start:i_end+1, 0:i_start], D=Gglobal[i_start:i_end+1, i_start:i_end+1] + DD
        )

    return corr_factor


def adapt_Ainv_sing(Gdenom_inv, Gglobal, r, s, i_start, i_end, pos1, pos2):
    """
        This routine is used if `adapt_Ainv` throws a "Singular matrix" exception. 

        Extends inverse of denominator matrix from `i_start` (inclusive) up to position `i_end` (inclusive)
        and puts particles both at position `pos1` and `pos2`. `pos1` < `pos2` are 
        required to lie in the closed interval [i_start, i_end].
    """
    assert r == 0
    assert Gdenom_inv.shape == (i_start, i_start)
    assert i_start <= i_end 
    assert i_start <= pos1 < pos2 <= i_end 

    # put particles at positions `pos1` and `pos2`.
    l = i_end+1-i_start 
    DD = np.zeros(l)
    DD[pos1-i_start] = -1; DD[pos2-i_start] = -1
    DD = np.diag(DD)

    Gnum_inv_ = block_update_inverse2(Ainv=Gdenom_inv, B=Gglobal[0:i_start, i_start:i_end+1], 
        C=Gglobal[i_start:i_end+1, 0:i_start], D=Gglobal[i_start:i_end+1, i_start:i_end+1] + DD)
    corr = block_update_det_correction2(
        Ainv=Gdenom_inv, B=Gglobal[0:i_start, i_start:i_end+1], 
        C=Gglobal[i_start:i_end+1, 0:i_start], D=Gglobal[i_start:i_end+1, i_start:i_end+1] + DD
        )
    return Gnum_inv_, corr



def adapt_Gdenom_inv(Gdenom_inv, Gglobal, r, s):
    """
    Extend inverse of denominator matrix up to position `s` (inclusive)
    and put an additional particle at position `s`.
    """
    #assert s > r or s == 0
    assert Gdenom_inv.shape == (s, s), "Gdenom_inv.shape= (%d, %d)" %(Gdenom_inv.shape)
    #assert s == r + 1 or s == 0
    # put an additional particle at position s 
    Gdenom_inv_ = block_update_inverse(Ainv=Gdenom_inv, B=Gglobal[0:s, s][:, None], C=Gglobal[s,0:s][None, :], D=Gglobal[s,s][None, None] - 1)
    corr = block_update_det_correction(Ainv=Gdenom_inv, B=Gglobal[0:s, s][:, None], C=Gglobal[s,0:s][None, :], D=Gglobal[s,s][None, None] - 1)
    # Gdenom_inv_ has a particle at s and (!) r
    return Gdenom_inv_, corr 


def adapt_Gnum_inv(Gnum_inv, Gglobal, r, s, i_last_nonsing, i):
    assert r > s 
    assert i_last_nonsing < i
    Gnum_inv_ = block_update_inverse2(Ainv=Gnum_inv, B=Gglobal[0:i_last_nonsing+1, i_last_nonsing+1:i+1], 
        C=Gglobal[i_last_nonsing+1:i+1, 0:i_last_nonsing+1], D=Gglobal[i_last_nonsing+1:i+1, i_last_nonsing+1:i+1])
    corr = block_update_det_correction2(
        Ainv=Gnum_inv, B=Gglobal[0:i_last_nonsing+1, i_last_nonsing+1:i+1], 
        C=Gglobal[i_last_nonsing+1:i+1, 0:i_last_nonsing+1], D=Gglobal[i_last_nonsing+1:i+1, i_last_nonsing+1:i+1]
        )
    return Gnum_inv_, corr


def lowrank_update_inv_addremove_rs(Gdenom_inv, r, s):
    """
        Remove a particle at `r` and add one at `s`. 
        Calculate the inverse of A^{'} given the resulting low-rank update:
            A^{'} = A  + U(r,s) * V(r,s)^{T}
        and the correction to the determinant:
            det(A^{'}) = det_corr * det(A).

        Return:
            inv(A^{'})
            det_corr
    """
    m = Gdenom_inv.shape[0]
    # capacitance matrix CC = id_2 + V_T*Ainv*U
    CC = np.array([[1 + Gdenom_inv[r,r], Gdenom_inv[r,s]     ], 
                   [   -Gdenom_inv[s,r], 1 - Gdenom_inv[s,s]]])

    AinvU = np.zeros((m,2))
    VtAinv = np.zeros((2,m))
    AinvU[:,0] = Gdenom_inv[:,r]
    AinvU[:,1] = Gdenom_inv[:,s]
    VtAinv[0,:] = Gdenom_inv[r,:]
    VtAinv[1,:] = - Gdenom_inv[s,:]
    
    # Calculate the determinant and inverse analytically because the condition number of CC can be extremely large. 
    det_CC = 1.0 + (Gdenom_inv[r,r] - Gdenom_inv[s,s]) - Gdenom_inv[r,r] * Gdenom_inv[s,s] + Gdenom_inv[r,s] * Gdenom_inv[s,r]

    inv_CC = np.array([ [ 1.0 - Gdenom_inv[s,s], - Gdenom_inv[r,s]    ], 
                        [ + Gdenom_inv[s,r]    , 1 + Gdenom_inv[r,r]  ] ]) / det_CC

    Gdenom_inv_ = Gdenom_inv - np.matmul(np.matmul(AinvU, inv_CC), VtAinv)
    det_corr = np.linalg.det(CC)

    return Gdenom_inv_, det_corr


def corr3_Gnum_from_Gdenom(Gdenom_inv_, Gglobal, r, s, xmin, i):
    assert s > r
    assert i > r 
    assert Gdenom_inv_.shape[0] == Gdenom_inv_.shape[1] 
    m = Gdenom_inv_.shape[0]
    assert m == s+1, "m=%d, s=%d" %(m, s)
    assert i >= xmin 
    
    # Calculate det(G_num_)  from G_denom_inv_

    dd = s - r # = 1 

    Ainv = Gdenom_inv_
    B = Gglobal[0:s+1, s+1:i+1]
    C = B.transpose()
    D = Gglobal[s+1:i+1, s+1:i+1]
    # Attention! B,C, and D are views into the original Green's function. 
    Dcopy = D.copy()
    Dcopy[-1,-1] = Dcopy[-1,-1] - 1 # add a particle here

    S = Dcopy - np.matmul(np.matmul(C, Ainv), B)
    det_Schur = np.linalg.det(S)

    # How close to zero must det(S) be so that inv(S) throws the error "Singular matrix" ?
    # This is not clear to me. 
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        det_Schur = 0.0
        return det_Schur, 0.0 
    # Calculate the inverse of the block matrix. The needed correction factor 
    # to the determinant is given just by a single element
    #       corr_factor_remove_r =   (1 + Ainv_new[r,r])).
    # Therefor it is sufficient to compute the Ablock of the inverse matrix. 

    AinvB = np.matmul(Ainv, B)
    CAinv = AinvB.transpose()
    # OK: condition number of S is in the range 10^0 to 10^2. 
    corr_factor_remove_r = 1 + (Ainv + np.matmul(AinvB, np.matmul(np.linalg.inv(S), CAinv)))[r,r]

    return det_Schur, corr_factor_remove_r


def det_Gnum_from_Gdenom(Gdenom_inv, det_Gdenom, Gglobal, r, s, xmin, i):
    #assert r > s 
    assert i > r 
    assert Gdenom_inv.shape[0] == Gdenom_inv.shape[1] 
    m = Gdenom_inv.shape[0]
    assert m == xmin, "m=%d, xmin=%d" %(m, xmin)
    assert i >= xmin 
    
    # 1. Calculate Gdenom_inv_ from Gdenom_inv using a low-rank update 
    # where a particle is removed at position `r`` and one is added at position `s`. 
    Gdenom_inv_, det_corr = lowrank_update_inv_addremove_rs(Gdenom_inv, r, s)
    det_Gdenom_ = det_Gdenom * det_corr

    Ainv = Gdenom_inv_
    B = Gglobal[0:xmin, xmin:i+1]
    C = B.transpose()
    D = Gglobal[xmin:i+1, xmin:i+1]
    # Attention! B,C, and D are views into the original Green's function. 
    Dcopy = D.copy()
    Dcopy[-1,-1] = Dcopy[-1,-1] - 1 # add a particle here

    S = Dcopy - np.matmul(np.matmul(C, Ainv), B)
    det_Schur = np.linalg.det(S)

    return det_Gdenom_ * det_Schur


# def lowrank_update_Schur_det_removeadd_rs(D, C, Gdenom_inv, B, r, s):
#     """
#         Low-rank update of the determinant of the Schur complement.
#         Remove a particle at position `r`, add one at position `s`. 
#     """ 
#     CGdenom_inv = np.matmul(C, Gdenom_inv) # should be precomputed
#     Gdenom_invB = CGdenom_inv.transpose()  # should be precomputed
#     CGB = np.matmul(CGdenom_inv, B)        # should be precomputed

#     # capacitance matrix
#     CC = np.array([ [1 + Gdenom_inv[r,r], Gdenom_inv[r,s]], 
#                     [- Gdenom_inv[s,r], 1 - Gdenom_inv[s,s]] ])
#     CC_inv = np.linalg.inv(CC)

#     CGB_updated = CGB - np.matmul( np.matmul(np.hstack((CGdenom_inv[:,r], CGdenom_inv[:,s])), CC_inv),
#                                np.vstack((Gdenom_invB[r,:], - Gdenom_invB[s,:])) )

#     # determinant of the Schur complement after low-rank update

#     return np.linalg.det(D - CGB_updated)


# def lowrank_update_Schur_det_remove_r(D, C, Gdenom_inv, B, r):
#     """
#         Low-rank update of the determinant of the Schur complement.
#         Remove a particle at position `r`, don't add any particle. 
#     """ 
#     CGdenom_inv = np.matmul(C, Gdenom_inv) # should be precomputed
#     Gdenom_invB = CGdenom_inv.transpose()  # should be precomputed
#     CGB = np.matmul(CGdenom_inv, B)        # should be precomputed

#     cc_inv = 1.0 / (1.0 + Gdenom_inv[r,r])

#     # IMPROVE: This can be simplified due to symmetry. 
#     CGB_updated = CGB - np.outer(CGdenom_inv[:,r], Gdenom_invB[r,:]) * cc_inv

#     return np.linalg.det(D - CGB_updated)
