"""
Attempt at optimal dampling algorithm combined with level shifting. 
(Not working, i.e. not improving convergence for difficult cases.
Also the energy is not strictly decreasing.)
Only level shifting is useful. 

Refs: 
[1] https://aip.scitation.org/doi/pdf/10.1063/1.1470195
[2] Cances and Le Bris: Can we outperform the DIIS approach for electronic structure calculations?
https://doi.org/10.1002/1097-461X(2000)79:2<82::AID-QUA3>3.0.CO;2-I
[3] https://www.esaim-m2an.org/articles/m2an/pdf/2000/04/m2an899.pdf
"""

import numpy as np
from scipy import linalg
import itertools

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

def HartreeFock_tVmodel(phys_system, potential='none', verbose=True, max_iter=1000, level_shift=0.1, outfile=None):
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
            mix_opt = line_search(phys_system, OBDM, OBDM_new) # This has no effect since mix_opt=1.0, always. 

            Fock_new = get_Fock_operator(phys_system, OBDM_new) - level_shift * OBDM_new
            #Fock1 = get_Fock_operator(phys_system, OBDM)
            F = (1 - mix_opt) * F + mix_opt * Fock_new
            OBDM = (1 - mix_opt) * OBDM + mix_opt * OBDM_new

            print("mix_opt=", mix_opt)
    # END: Hartree-Fock 

    HF_energies, HF_orbitals = linalg.eigh(F)

    return (HF_energies, HF_orbitals)
    

def calc_density_corr(phys_system, OBDM):

    def translate(s, n, T_d):
        """translate set of sites `s` using mapping `T_d` for n times"""
        for _ in range(n):
            s = T_d[s]
        return s
    ###### setting up user-defined symmetry transformations for 2d lattice ######
    nx=phys_system.nx
    ny=phys_system.ny
    ns=phys_system.ns
    s = np.arange(ns) # sites [0,1,2,....]

    x = s%nx # x positions for sites
    y = s//nx # y positions for sites
    T_x = (x+1)%nx + nx*y # translation along x-direction
    T_y = x +nx*((y+1)%ny) # translation along y-direction
    P_x = x + nx*(ny-y-1) # reflection about x-axis
    P_y = (nx-x-1) + nx*y # reflection about y-axis
    #
    nncorr = np.zeros((nx, ny), dtype=np.float64)

    for tx, ty in itertools.product(range(nx), range(ny)):
        # all pairs of sites (i,j) with distance dist(i,j) = (tx,ty)
        pair_list = [[i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(ns)]
        nncorr[tx, ty] = sum([(OBDM[i,i]*OBDM[j,j] - OBDM[j,i]*OBDM[i,j]) for (i,j) in pair_list]) / ns

    return nncorr

# def graph_distance_corr(phys_system, OBDM):
#     """Correlation function as a funtion of graph distance"""
#     from graph_distance import graph_distance_L6
#     assert phys_system.nx == 6 == phys_system.ny
#     S, num_dist = graph_distance_L6()
#     av = np.sum(config_2D) / L**2
#     corr = np.zeros(L+1)
#     for ix in range(L):
#         config_2D_shiftx = np.roll(config_2D, shift=-ix, axis=0)
#         for iy in range(L):
#             config_2D_shifted = np.roll(config_2D_shiftx, shift=-iy, axis=1)    
#             for r in (0,1,2,3,4,5,6,):    
#                 corr[r] += np.sum([(config_2D_shifted[(0, 0)] - av) * (config_2D_shifted[p] - av) for p in S[r]]) / (num_dist[r] * L**2)


def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    #_test()
    L=8
    num_particles = 32
    Vint = 4.05
    phys_system = PhysicalSystem(L, L, L*L, num_particles, dim=2, Vint=Vint) 
    # phys_system = PhysicalSystem(4, 4, 16, 5, dim=2, Vint=4.0) 
    energies, U = HartreeFock_tVmodel(phys_system, level_shift=3.0, max_iter=10000, outfile=open("HF_test.dat","w"))    

    C= U[:,0:phys_system.np]
    OBDM = np.matmul(C, C.conj().transpose())
    nncorr = calc_density_corr(phys_system, OBDM)

    np.savetxt("nncorr_L%dNp%dVint%f.dat"%(L, num_particles, Vint), nncorr.flatten())
