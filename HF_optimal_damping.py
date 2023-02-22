"""
Attempt at optimal dampling algorithm. 
(Not working, i.e. not improving convergence for difficult cases.
Also the energy is not strictly decreasing.)

Refs: 
[1] https://aip.scitation.org/doi/pdf/10.1063/1.1470195
[2] Cances and Le Bris: Can we outperform the DIIS approach for electronic structure calculations?
https://doi.org/10.1002/1097-461X(2000)79:2<82::AID-QUA3>3.0.CO;2-I
"""

import numpy as np
from scipy import linalg

from slater_determinant import Slater2spOBDM
from VMC_common import PhysicalSystem  

def get_H_kin(phys_system):
    """One-body part of Fock operator"""
    ns = phys_system.ns
    lattice = phys_system.lattice 
    H_kin = np.zeros((ns, ns), dtype=np.float64)
    for i in range(ns):
        for nd in range(lattice.coord // 2): # only valid for cubic lattice 
            j = lattice.neigh[i, nd]
            H_kin[i,j] = -1.0
            H_kin[j,i] = -1.0  
    return H_kin  

def get_G(phys_system, OBDM):
    """Two-body part of Fock operator."""
    ns = phys_system.ns
    lattice = phys_system.lattice 
    G = np.zeros((ns, ns), dtype=np.float64)
    for i in range(ns):
        for nd in range(lattice.coord): # only valid for cubic lattice     
            j = lattice.neigh[i,nd]
            G[i,i] += OBDM[j,j]
            G[i,j] = - OBDM[j,i]
            G[j,i] = - OBDM[i,j]            
    return phys_system.Vint * G

def get_Fock_operator(phys_system, OBDM):
    return get_H_kin(phys_system) + get_G(phys_system, OBDM)

def line_search(phys_system, OBDM1, OBDM2):
    diff = OBDM2 - OBDM1
    Fock1 = get_Fock_operator(phys_system, OBDM1)
    Fock2 = get_Fock_operator(phys_system, OBDM2)
    s = np.trace(Fock1 @ diff)
    c = np.trace((Fock2 - Fock1) @ diff)

    l_min = 1.0 if c <= -s else 1.0
    return l_min

def HartreeFock_tVmodel(phys_system, potential='none', verbose=True, max_iter=1000, mix=0.5, level_shift=5.0, outfile=None):
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

    F = get_Fock_operator(phys_system, OBDM)
    
    while not converged: 
        counter += 1 
        eigvals, U = linalg.eigh(F)
        # Aufbau principle
        OBDM_new = Slater2spOBDM(U[:, 0:num_particles])
        if verbose:
            E_gs_HF = HF_gs_energy(OBDM_new)
            E_kin = np.trace(H_kin @ OBDM_new)
            E_int = 0.5*np.trace(get_G(phys_system, OBDM_new) @ OBDM_new)
            print("E_gs_HF=", E_gs_HF,"E_kin=", E_kin, "E_int=", E_int)
            print("%d %f %f %f"%(counter, E_gs_HF, E_kin, E_int), file=fh_conv)

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
            mix_opt = line_search(phys_system, OBDM, OBDM_new)

            level_shift = 3.0
            Fock_new = get_Fock_operator(phys_system, OBDM_new) - level_shift * OBDM_new
            #Fock1 = get_Fock_operator(phys_system, OBDM)
            F = (1 - mix_opt) * F + mix_opt * Fock_new
            OBDM = (1 - mix_opt) * OBDM + mix_opt * OBDM_new

            print("mix_opt=", mix_opt)
    # END: Hartree-Fock 

    eigvals, U = linalg.eigh(F)

    return (eigvals, U)
    


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    #_test()
    phys_system = PhysicalSystem(6, 6, 36, 13, dim=2, Vint=10.0) 
    # phys_system = PhysicalSystem(4, 4, 16, 5, dim=2, Vint=4.0) 
    eigvals, U = HartreeFock_tVmodel(phys_system, outfile=open("HF_test.dat","w"))    
