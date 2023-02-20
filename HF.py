"""
Hartree-Fock routines for initialization of the Slater determinant.
"""
#TODO: wrong results if np.linalg.eigh() is used instead of scipy.linalg.eigh()
#      The eigenvalues are the same, but eigenvectors differ, which leads to different 
#      self-consistent HF Hamiltonians for open shell systems. 
# - implement generic lattices, use adjacency matrix 

import numpy as np
from scipy import linalg

from slater_determinant import Slater2spOBDM, Slater2rdm1
from VMC_common import PhysicalSystem  


def HartreeFock_tVmodel(phys_system, potential='none', verbose=True, max_iter=1000, mix=0.1, outfile=None):
    """
    Returns single-particle eigenstates of the Hartree-Fock solution of a 
    t-V model on a cubic lattice specified by `phys_system`. 

    Example:
    --------
    >>> phys_system = PhysicalSystem(4, 4, 16, 8, dim=2, Vint=3.0) 
    >>> eigvals, U = HartreeFock_tVmodel(phys_system, outfile=open("HF_test.dat","w"))
    """
    def HF_gs_energy(OBDM):
        """Note: The Hartree-Fock ground state energy is *not* the sum of HF eigenvalues.
           (see Szabo-Ostlund,  chapter 3).
           Two alternative ways of calculating the ground state energy are shown here."""
        # one-body Hamiltonian 
        H_onebody = H_kin.copy()
        for i in range(ns):
            H_onebody += V_pot[i]
        H_int = np.zeros((ns,ns), dtype=np.float64)
        for i in range(ns):
            for nd in range(lattice.coord):
                j = lattice.neigh[i,nd]
                H_int[i,i] += Vint*OBDM[j,j]
                H_int[i,j] = - Vint*OBDM[j,i]
                H_int[j,i] = - Vint*OBDM[i,j]
        # Hartree-Fock ground state energy: E_tot = Tr(rho H_onebody) + (1/2) Tr(rho H_int)
        # H_HF = H_onebody + H_int is *not* exactly the Hamiltonian used in the self-consistency loop. 
        E_tot = np.trace(np.matmul(OBDM, H_onebody)) + 0.5 * np.trace(np.matmul(OBDM, H_int))

        # adjacency matrix 
        Adj = np.zeros((ns,ns), dtype=np.float64)
        for i in range(ns):
            for nd in range(lattice.coord):
                j = lattice.neigh[i,nd]
                Adj[i,j] = 1 
        # Eq. (17) from Phys. Rev. B 102, 205122
        E_tot2 = - t_hop * np.einsum('ij,ji->', Adj, OBDM) - 0.5 * Vint * ( 
                np.einsum('ij,ij,ji->', Adj, OBDM, OBDM) - np.einsum('ij,ii,jj->', Adj, OBDM, OBDM)
                )
        assert np.isclose(E_tot, E_tot2)

        return E_tot
    
    assert isinstance(phys_system, PhysicalSystem)
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
    # some random initialization (we do not care from which random emsemble)
    q,r = np.linalg.qr(np.random.randn(ns,ns))
    OBDM = np.matmul(q[:,0:num_particles], q[:,0:num_particles].transpose())
    fh_conv = open("HF_energyconv.dat", "w")
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
        OBDM_new = Slater2spOBDM(U[:, 0:num_particles])
        if verbose:
            E_gs_HF = HF_gs_energy(OBDM_new)
            print("E_gs_HF=", E_gs_HF)
            print("%d %f"%(counter, E_gs_HF), file=fh_conv)

        if np.all(np.isclose(OBDM_new, OBDM, rtol=1e-8)) or counter >= max_iter: 
            converged = True
            if verbose:
                print("converged")
                print("num iterations=", counter)
                print("Ns = %d"%(ns), file=outfile)
                print("Np = %d"%(num_particles), file=outfile)
                print("number of HF self-consistency iterations: %d"%(counter), file=outfile)
                print("many-body HF g.s. energy= %12.8f" % (HF_gs_energy(OBDM_new)), file=outfile)
                print("many-body HF g.s. energy= %f" % HF_gs_energy(OBDM_new), file=outfile)
                fmt_string = "single-particle spectrum = \n" + "%f \n"*ns
                print(fmt_string % (tuple(eigvals)), file=outfile) 
        else:
            OBDM = (1-mix) * OBDM_new + mix * OBDM
    # END: Hartree-Fock 

    eigvals, U = linalg.eigh(H_HF)

    return (eigvals, U)
    
    

def prepare_test_system_zeroT(Nsites=21, potential='parabolic', PBC=True, HF=True, Nparticles=0, Vnnint=0.0, verbose=False):
    """
    One-dimensional system of free fermions with Nsites sites
    in an external trapping potential.
    Return the matrix of single-particle eigenstates. 

    Example:
    --------
    >>> (_, U) = prepare_test_system_zeroT(Nsites=4, potential='none', HF=True, Nparticles=2, Vnnint=1.0)
    >>> np.isclose(U[0,0], 0.5269026969395422)
    True
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
