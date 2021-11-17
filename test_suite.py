#!/usr/bin/python3.5
"""
    Routines for benchmarking the scheme of direct sampling of 
    free fermion pseudo density matrices. 
"""

import numpy as np
from scipy import linalg


def occ2int_spinless(occ_vector):
    """
        Map a spinless fermion occupation vector to an integer by interpreting 
        the occupation vector as the binary prepresentation of 
        an integer with the most significant bit to the right. 
        
            occ_vector = [1, 0, 1, 0]   ->   integer = 5
    """
    occ_vector = np.array(occ_vector, dtype=np.int8)
    s = 0
    for k in range(len(occ_vector)):
        # least significant bit to the right
        # if (occ_vector[-(k+1)] == 1):
        #     s = s + 2**k
        # least significant bit to the left            
        if (occ_vector[k] == 1):
            s = s + 2**k
    return s  


def occ2int_spinful(occ_vector_up, occ_vector_dn, debug=False):
    """
        Combine the occupation vectors for spin up and spin down 
        and map the resulting combined occupation vector to 
        an integer. The most significant bit is to the right.

        Example:
        ========
            occ_vector_up = [1, 0, 0, 1]
            occ_vector_dn = [0, 1, 1, 0]
            [occ_vector_up, occ_vector_dn] = [1, 0, 0, 1; 0, 1, 1, 0]  -> integer = 105
    """
    assert(len(occ_vector_up) == len(occ_vector_dn))
    occ_vector_up = np.array(occ_vector_up)
    occ_vector_dn = np.array(occ_vector_dn)
    occ_vector = np.hstack((occ_vector_up, occ_vector_dn))
    
    if (debug):
        print(occ_vector)

    return occ2int_spinless(occ_vector)


def int2occ_spinful(integer, Nsites):
    """
        Convert the integer representing an occupation number vector
        for spin up and spin down into a bitstring. 

        Example:
        ========
            occ_vector_up = [1, 0, 0, 1]
            occ_vector_dn = [0, 1, 1, 0]
            integer = 105 -> [occ_vector_up, occ_vector_dn] = [1, 0, 0, 1; 0, 1, 1, 0]     
    """
    Nspecies = 2

    # least significant bit to the right 
    i = integer 
    bitstring = []
    while(i != 0):
        bit = i % 2
        bitstring.insert(0, bit)
        i = i // 2
    # write leading zeros
    for _ in range(Nspecies*Nsites - len(bitstring)):
        bitstring.insert(0, 0)

    assert(len(bitstring) == 2*Nsites)

    return bitstring 


def prepare_test_system_zeroT(Nsites=21, potential='parabolic', PBC=True):
    """
        One-dimensional system of free fermions with Nsites sites
        in an external trapping potential.
        Return the matrix of single-particle eigenstates. 
    """
    i0=int(Nsites/2)
    V = np.zeros(Nsites)
    t_hop = 1.0
    
    if (potential == 'parabolic'):
        V_max = 1.0*t_hop   # max. value of the trapping potential at the edge of the trap
                        # (in units of the hopping)
        V_pot = V_max / i0**2
        for i in range(Nsites):
            V[i] = V_pot*(i-i0)**2
    elif (potential == 'random-binary'):
        absU = 7.2; dtau=0.05
        alphaU = np.arccosh(np.exp(dtau*absU/2.0))
        V = alphaU * np.random.random_integers(0,1,size=Nsites)
    elif (potential == 'none'):
        V[:] = 0.0
    else:
        print("Unknown type of external potential")
        exit()

    H = np.zeros((Nsites,Nsites), dtype=np.float64)
    for i in range(Nsites):
        H[i,i] = V[i]
        if (i+1 < Nsites):
            H[i,i+1] = -t_hop
            H[i+1,i] = -t_hop
    H[0, Nsites-1] = H[Nsites-1, 0] = -t_hop if PBC else 0.0

    eigvals, U = linalg.eigh(H)

    return (Nsites, U)


def prepare_test_system_finiteT(Nsites=21, beta=1.0, mu=0.0, potential='parabolic'):
    """
        One-dimensional system of free fermions with Nsites sites
        in an external trapping potential. 

        Input:
            beta = inverse temperature 
            mu = chemical potential 
            potential: Type of the external potenial which is either 'parabolic',
               'random-binary', or 'none'.
        Output:
            Return the one-body density matrix (OBDM)
                <c_i^{\dagger} c_j> = Tr(e^{-beta H}c_i^{\dagger} c_j)
            and the occupations of natural orbitals (momentum-distribution
            function for a translationally-invariant system) as a vector.
    """
    #assert(Nsites%2==1), "test_suites: Nsites should be an odd number."

    i0=int(Nsites/2)
    V = np.zeros(Nsites)
    t_hop = 1.0

    if (potential == 'parabolic'):
        V_max = 1.0*t_hop   # max. value of the trapping potential at the edge of the trap
                        # (in units of the hopping)
        V_pot = V_max / i0**2
        for i in range(Nsites):
            V[i] = V_pot*(i-i0)**2
    elif (potential == 'random-binary'):
        absU = 7.2; dtau=0.05
        alphaU = np.arccosh(np.exp(dtau*absU/2.0))
        V = alphaU * np.random.random_integers(0,1,size=Nsites)
    elif (potential == 'none'):
        V[:] = 0.0
    else:
        print("Unknown type of external potential")
        exit()

    H = np.zeros((Nsites,Nsites), dtype=np.float64)
    for i in range(Nsites):
        H[i,i] = V[i]
        if (i+1 < Nsites):
            H[i,i+1] = -t_hop
            H[i+1,i] = -t_hop

    sp_energies, U = linalg.eigh(H)

    # fugacity
    z = np.exp(beta*mu)
    # momentum distribution function (for a translationally-invariant system)
    # or natural orbital occupancies (for a trapped system)
    MDF = np.diag(z*np.exp(-beta*sp_energies) / (1 + z*np.exp(-beta*sp_energies)))
    OBDM = np.matmul(np.matmul(U, MDF), U.conj().T)

    return (Nsites, beta, mu, np.sort(np.diag(MDF))[::-1], OBDM)


def square_region(OBDM, L_A, x0=1, y0=1):
    """
        Extract the elements of the OBDM which correspond to a square 
        region of the real-space lattice of size L_A x L_A.
        The lower left corner of the square region has coordinates (1,1).
    """
    Ns = OBDM.shape[0]
    L = int(np.sqrt(float(Ns)))
 
    row_idx = [(x0-1) + i + (y0-1)*L + (j-1)*L for j in range(1,L_A+1) for i in range(1,L_A+1)]
    col_idx = row_idx 

    return OBDM[np.ix_(row_idx, col_idx)]


def Slater2spOBDM(sp_states):
    """
        <Sdet| c_i c_j^{\dagger} |Sdet> = det([P_i^{\prime}]^t P_j^{\prime})

        where P_j^{prime} is the matrix of single-particle states `sp_states`
        with one column added where all elements are zero except the j-th.
        `sp_states` has shape (Ns, Np), P_j^{\prime} has shape (Ns, Np+1).
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

            print("P_prime_i.T * P_prime_j")
            print(np.matmul(P_i_prime.T, P_j_prime))

            GF[i,j] = np.linalg.det(np.matmul(P_i_prime.T, P_j_prime))

    OBDM = np.eye(Ns) - GF.T
    return OBDM


def local_OBDM(alpha, sp_states):
    """
        NOT TESTED YET !!!

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
        alpha (array of ints) : particle positions in basis states alpha, ordered increasingly 
        sp_states (Nsites x Nparticles array) : P-matrix representing Slater determinant 

        Output:
        -------
        local_OBDM (Nsites x Nsites matrix): elements of local kinetic energy 
    """
    assert len(alpha) == sp_states.shape[1]
    Nsites = sp_states.shape[0]
    L_idx = np.asarray(alpha) # select these cols from P-matrix 
    M = np.matmul( sp_states, np.linalg.inv(sp_states[L_idx]) ) 
    GG = np.zeros((Nsites, Nsites))
    GG[:, L_idx] = M[:,:]
    return GG.T


def ratio_Slater(G, r, s):
    """
        GIVES CORRECT RESULT FOR VALID PAIRS (r,s),
        but also gives result for r unocc -> s occ !!! what to do ?

        R =  <\beta | psi > / <\alpha | \psi > 

        where basis state \beta is obtained from \alpha by moving one particle from 
        (occupied) site r to (unoccupied) site s. 

        Input:
        ------
         G (Nsites x Nsites array) : local OBDM for state alpha:  <\alpha| c_j^{\dagger} c_i |\psi> / <\alpha | \psi>
    """
    G = np.asarray(G)
    R = (1 - G[r,r] - G[s,s] + G[r,s] + G[s,r])
    return R