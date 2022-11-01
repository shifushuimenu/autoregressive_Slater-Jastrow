#!/usr/bin/python3.5
"""
Routines for benchmarking the scheme of direct sampling of 
free fermion pseudo density matrices. 
"""

import numpy as np
from scipy import linalg
from bitcoding import bin2pos

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


def generate_random_config(Ns, Np):
    """
    Generate a random occupation number state (with fixed particle number).
    """
    config = np.zeros((Ns,), dtype=int) 
    #config[0] = 1; config[-1] = 1 # !!!!! REMOVE: Make sure no hopping across periodic boundaries can occur. 
    counter = 0
    while counter < Np:
        pos = np.random.randint(low=0, high=Ns, size=1)
        if config[pos] != 1:
            config[pos] = 1
            counter += 1 
    return config


def HartreeFock_tVmodel(phys_system, potential='none', verbose=False, max_iter=1000, outfile=None):
    """
    Returns single-particle eigenstates of the Hartree-Fock solution of a 
    t-V model on a cubic lattice specified by `phys_system`. 
    """
    ns = phys_system.ns
    num_particles = phys_system.np
    nx = phys_system.nx
    ny = phys_system.ny
    lattice = phys_system.lattice
    Vint = phys_system.Vint
    
    t_hop = 1.0
    
    if potential == 'none':
        V_pot = np.zeros((ns,))
    else:
        raise NotImplementedError
    
    H_kin = np.zeros((ns, ns), dtype=np.float64)
    for i in range(ns):
        for nd in range(lattice.coord // 2): # only valid for cubic lattice 
            j = lattice.neigh[i, nd]
            H_kin[i,j] = -t_hop
            H_kin[j,i] = -t_hop

    eigvals, U = linalg.eigh(H_kin) 
    # BEGIN: Hartree-Fock self-consistency loop 
    converged = False 
    counter = 0
    OBDM = np.zeros((ns, ns))
    while not converged: 
        counter += 1 
        H_HF = H_kin.copy() 
        for i in range(ns):
            H_HF[i,i] = V_pot[i] 
            for nd in range(lattice.coord // 2): # only valid for cubic lattice 
                j = lattice.neigh[i, nd]
                H_HF[i,i] += Vint*OBDM[j,j] 
                H_HF[i,j] = -t_hop - Vint*OBDM[j,i]
                H_HF[j,i] = -t_hop - Vint*OBDM[i,j] 

        eigvals, U = linalg.eigh(H_HF)
        if verbose:
            print(eigvals[0:num_particles+1])
            print("E_GS_HF=", sum(eigvals[0:num_particles]))
        OBDM_new = Slater2spOBDM(U[:, 0:num_particles])

        if np.all(np.isclose(OBDM_new, OBDM, rtol=1e-8)) or counter >= max_iter: 
            converged = True
            if verbose:
                print("converged:")
                print("counter=", counter)
                print("OBDM_new=", OBDM_new)
                print("many-body HF g.s. energy= %f" % (np.sum(eigvals[0:num_particles])), file=outfile)
                fmt_string = "single-particle spectrum = \n" + "%f \n"*ns
                print(fmt_string % (tuple(eigvals)), file=outfile) 
        else:
            OBDM = OBDM_new
    # END: Hartree-Fock 

    eigvals, U = linalg.eigh(H_HF)

    return (eigvals, U)
    
    

def prepare_test_system_zeroT(Nsites=21, potential='parabolic', PBC=True, HF=True, Nparticles=0, Vnnint=0.0, verbose=False):
    """
    One-dimensional system of free fermions with Nsites sites
    in an external trapping potential.
    Return the matrix of single-particle eigenstates. 
    """
    i0=int(Nsites/2)
    V_pot = np.zeros(Nsites)
    t_hop = 1.0
    
    if (potential == 'parabolic'):
        V_max = 1.0*t_hop   # max. value of the trapping potential at the edge of the trap
                        # (in units of the hopping)
        V0_pot = V_max / i0**2
        for i in range(Nsites):
            V_pot[i] = V0_pot*(i-i0)**2
    elif (potential == 'random-binary'):
        absU = 7.2; dtau=0.05
        alphaU = np.arccosh(np.exp(dtau*absU/2.0))
        V_pot = alphaU * np.random.random_integers(0,1,size=Nsites)
    elif (potential == 'none'):
        V_pot[:] = 0.0
    else:
        print("Unknown type of external potential")
        exit()

    H = np.zeros((Nsites,Nsites), dtype=np.float64)
    for i in range(Nsites):
        H[i,i] = V_pot[i]
        if (i+1 < Nsites):
            H[i,i+1] = -t_hop
            H[i+1,i] = -t_hop
    H[0, Nsites-1] = H[Nsites-1, 0] = -t_hop if PBC else 0.0

    eigvals, U = linalg.eigh(H)
    OBDM_initial = Slater2spOBDM(U[:, 0:Nparticles])
    # BEGIN: Hartree-Fock self-consistency loop 
    if HF:
        #assert Vnnint!=0 and Nparticles!=0
        
        converged = False 
        counter = 0
        OBDM = np.zeros((Nsites, Nsites))
        while not converged: 
            counter += 1 
            H = np.zeros((Nsites,Nsites), dtype=np.float64)
            for i in range(Nsites):
                H[i,i] = V_pot[i] + Vnnint*OBDM[i,i]
                if (i+1 < Nsites):
                    H[i,i+1] = -t_hop - Vnnint*OBDM[i+1,i]
                    H[i+1,i] = -t_hop - Vnnint*OBDM[i,i+1]
            H[0, Nsites-1] = H[Nsites-1, 0] = (-t_hop - Vnnint*OBDM[0, Nsites-1]) if PBC else 0.0    

            eigvals, U = linalg.eigh(H)
            if verbose:
                print("min(eigvals)=", min(eigvals))
                print("E_GS_HF=", sum(eigvals[0:Nparticles]))
                print(eigvals[0:Nparticles+1])
            OBDM_new = Slater2spOBDM(U[:, 0:Nparticles])

            if np.all(np.isclose(OBDM_new, OBDM, rtol=1e-4)) or counter == 1000: 
                converged = True
                if verbose:
                    print("converged:")
                    print("g.s. energy=", np.sum(eigvals[0:Nparticles]))
                    #print("OBDM_initial=", np.diag(OBDM_initial))
                    #print("OBDM = ", np.diag(OBDM))
                    #print("OBDM_new = ", np.diag(OBDM_new))
            else:
                OBDM = OBDM_new
    # END: HArtree-Fock 

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


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
