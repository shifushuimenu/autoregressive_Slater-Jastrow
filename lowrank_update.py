import numpy as np

def corr_factor_removeadd_rs(Ainv, r, s):
    """ 
        Correction factor to 
             det(G_{K,K} - N_K)
        with N_K = (n_0, n_1, ..., n_{K-1}) where n_r = 1 and n_s = 0
        and a particle is moved such that after the update n_r = 0 and n_s = 1.         
    """
    return 1 + Ainv[r,r] - Ainv[s,s] - Ainv[r,r] * Ainv[s,s] + Ainv[r,s]*Ainv[s,r]

def corr_factor_add_s(Ainv, s):
    """add a particle at position s without removing any particle """
    return (1 - Ainv[s,s])

def corr_factor_remove_r(Ainv, r):
    """remove a particle at position r without adding any particle"""
    return (1 + Ainv[r,r])
 

def lowrank_update_Schur_det_removeadd_rs(D, C, Gdenom_inv, B, r, s):
    """
        Low-rank update of the determinant of the Schur complement.
        Remove a particle at position `r`, add one at position `s`. 
    """ 
    CGdenom_inv = np.matmul(C, Gdenom_inv) # should be precomputed
    Gdenom_invB = CGdenom_inv.transpose()  # should be precomputed
    CGB = np.matmul(CGdenom_inv, B)        # should be precomputed

    # capacitance matrix
    CC = np.array([ [1 + Gdenom_inv[r,r], Gdenom_inv[r,s]], 
                    [- Gdenom_inv[s,r], 1 - Gdenom_inv[s,s]] ])
    CC_inv = np.linalg.inv(CC)

    CGB_updated = CGB - np.matmul( np.matmul(np.hstack((CGdenom_inv[:,r], CGdenom_inv[:,s])), CC_inv),
                               np.vstack((Gdenom_invB[r,:], - Gdenom_invB[s,:])) )

    # determinant of the Schur complement after low-rank update

    return np.linalg.det(D - CGB_updated)


def lowrank_update_Schur_det_remove_r(D, C, Gdenom_inv, B, r):
    """
        Low-rank update of the determinant of the Schur complement.
        Remove a particle at position `r`, don't add any particle. 
    """ 
    CGdenom_inv = np.matmul(C, Gdenom_inv) # should be precomputed
    Gdenom_invB = CGdenom_inv.transpose()  # should be precomputed
    CGB = np.matmul(CGdenom_inv, B)        # should be precomputed

    cc_inv = 1.0 / (1.0 + Gdenom_inv[r,r])

    # IMPROVE: This can be simplified due to symmetry. 
    CGB_updated = CGB - np.outer(CGdenom_inv[:,r], Gdenom_invB[r,:]) * cc_inv

    return np.linalg.det(D - CGB_updated)