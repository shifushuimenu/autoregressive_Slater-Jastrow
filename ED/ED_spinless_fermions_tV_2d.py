import itertools
import numpy as np
import h5py
import pickle
import argparse 

from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_general 

desc_str = "Exact diagonalization of t-V model of spinless fermions \n" + \
       "on the square lattice (with pbc) using QuSpin."
parser = argparse.ArgumentParser(description=desc_str)
parser.add_argument('Lx', type=int, help='width of square lattice')
parser.add_argument('Ly', type=int, help='height of square lattice')
parser.add_argument('Np', metavar='N', type=int, help='number of particles')
parser.add_argument('Vint', metavar='V/t', type=float, help='nearest neighbout interaction (V/t > 0 is repulsive)')
parser.add_argument('--kmax', type=int, help="maximum number of eigenvalues in ED (default=1)", default=1)
parser.add_argument('--momentum_sector', type=int, nargs='*', dest='kblocks',
                    help="target specific momentum sectors ( default: all momentum sectors; KBLOCK denotes two integers, e.g. `1 1 -1 -1 1 -1 -1 1` )")
args = parser.parse_args()

# Test case: Lx=Ly=4, Np=7, Vint=3.0 => g.s. energy = -3.989234414725094

Lx = args.Lx; Ly = args.Ly; Np = args.Np; Vint = args.Vint 
kmax = args.kmax 
assert 0 < Np < Lx*Ly

if args.kblocks is not None:
    NK = len(args.kblocks)
    assert NK%2 == 0
    kxs = [args.kblocks[2*i] for i in range(NK//2)]
    kys = [args.kblocks[2*i+1] for i in range(NK//2)]
    kblocks = list(zip(kxs,kys))

J=1.0 # hopping matrix element
N_2d = Lx*Ly # number of sites
paramstr = "Lx%dLy%dNp%dVint%f"%(Lx, Ly, Np, Vint)

###### setting up user-defined symmetry transformations for 2d square lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
#
def translate(s, n, T_d):
    """translate set of sites `s` using mapping `T_d` for n times"""
    for _ in range(n):
        s = T_d[s]
    return s

def find_deg_kblocks(energy_dict):
    """from the dictionary with energies for each kblock find the pairs (kx,ky)
       of the degenerate momentum blocks containing the ground state"""
    minEs = []; ks = []; energies = []
    for item in energy_dict.values(): # each item is a dictionary itself 
        energies.extend(item.get("energies"))
        minEs.append(item.get("minE"))
        ks.append((item.get("kx"), item.get("ky")))
    indx = np.argsort(minEs)    
    minEs_sorted = np.array(minEs)[indx] # convert to numpy array to use fancy indexing 
    ks_sorted = np.array(ks)[indx] 
    deg = 0
    e0 = minEs_sorted[0]
    for e in minEs_sorted:
        if np.abs(e - e0) > 1e-8: break 
        else: deg += 1
    deg_kblocks = ks_sorted[0:deg]

    # first excited state 
    energies.sort()
    assert np.isclose(energies[deg-1], e0, atol=1e-8)
    if len(energies) > deg: e1 = energies[deg]
    else: e1 = np.NaN


    return e0, e1, deg, deg_kblocks

###### setting up hamiltonian ######
hopping_left=[[-J,i,T_x[i]] for i in range(N_2d)] + [[-J,i,T_y[i]] for i in range(N_2d)]
hopping_right=[[+J,i,T_x[i]] for i in range(N_2d)] + [[+J,i,T_y[i]] for i in range(N_2d)]
interaction=[[Vint,i,T_x[i]] for i in range(N_2d)] + [[Vint,i,T_y[i]] for i in range(N_2d)]
#
static=[["+-",hopping_left],["-+",hopping_right],["nn",interaction]]

Egs = 100000.0
energies_dict = dict()
num_states_tot = 0

if args.kblocks is None:
    kblocks = itertools.product(range(-Lx//2,Lx//2), range(-Ly//2,Ly//2))

for (kx, ky) in kblocks:
    basis_kblock=spinless_fermion_basis_general(N_2d,Nf=Np,_Np=Np,kxblock=(T_x,kx),kyblock=(T_y,ky))
    num_states_tot += basis_kblock.Ns

    if basis_kblock.Ns > 1:
        H=hamiltonian(static,[],basis=basis_kblock,dtype=np.complex64) # double precision required to get ground state degeneracies right 
        ## diagonalise H
        E, v = H.eigsh(k=kmax, which='SA') # sparse 
        print("E=", E)
        energies_dict["kx%dky%d"%(kx,ky)] = {"kx":kx, "ky":ky, "energies": E[0:kmax], "minE": E[0]}        

SzSzcorr  = np.zeros((Lx, Ly)) 
nncorr = np.zeros((Lx,Ly))
# To obtain correct correlation functions it is important to sum over all degenerate 
# momentum sectors. Enter the degenerate momentum sectors here ! :
e0, e1, num_deg_gs, deg_kblocks = find_deg_kblocks(energies_dict)

# store degenerate ground state kblocks for reusing in another run
if not args.kblocks:
    with open("deg_kblocks_"+paramstr+".tmp", 'w') as fh:
        fh.write(" ".join([str(k[0])+" "+str(k[1]) for k in deg_kblocks]))

for (kx,ky) in deg_kblocks:
    basis_kblock=spinless_fermion_basis_general(N_2d,Nf=Np,_Np=Np,kxblock=(T_x,kx),kyblock=(T_y,ky))
    H=hamiltonian(static,[],basis=basis_kblock,dtype=np.complex64)
    E_gs, v_gs = H.eigsh(k=kmax, which='SA') # sparse 

    # construct operators for measuring spin-spin and density-density correlation function 
    for tx in range(Lx):
        for ty in range(Ly):
            operator_list = [["nn", [[1, i, translate(translate(i, tx, T_x), ty, T_y)] for i in range(N_2d)]], ]
            C = hamiltonian(static_list=operator_list,dynamic_list=[],basis=basis_kblock) 
            av = C.expt_value(v_gs, time=0)[0].real
            nncorr[tx,ty] += av / N_2d / num_deg_gs
            SzSzcorr[tx, ty] += (4*av.real - 4*Np + N_2d) / N_2d / num_deg_gs # transform from density-density to spin-spin correlation function (spin=+/-1)

# Output 
print("Ground state energy=", e0, "in momentum sectors ", deg_kblocks)
print("First excited state energy=", e1)

np.savetxt("SzSzcorr_ED_"+paramstr+".dat", SzSzcorr)
np.savetxt("nncorr_ED_"+paramstr+".dat", nncorr)
# Connected density-density correlation function (valid only for translationally invariant systems)
np.savetxt("nncorr_conn_ED_"+paramstr+".dat", nncorr[:,:] - (Np/N_2d)**2)
with open("gs_energies_"+paramstr+".pickle", "wb") as fh:
    pickle.dump(energies_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
with open("gs_energy_ED_"+paramstr+".dat", "w") as fh:
    print("g.s. energy = ", e0, file=fh)
    print("in momentum sector ", deg_kblocks, file=fh)
    print("1st excited state energy = ", e1, file=fh)
    print("", file=fh)
    print("Note: To consider 1st excited states both in the same momentum sector as the ground state", file=fh)
    print("as well as other momentum sectors, it is required that args.kmax >= 2.", file=fh)
    print("If args.kmax = 1, the energy reported here is the minimum energy in momentum blocks other than the ground state.", file=fh)
    print("(or `nan` if only a single momentum sector was targeted)", file=fh)

# ground state and first excited state with their respective momentum labels 
# There can be several degenerate excited states. 

with h5py.File(paramstr+".h5", "w") as f:
    d1 = f.create_dataset("SzSzcorr", shape=SzSzcorr.shape)
    d1[...] = SzSzcorr
    d2 = f.create_dataset("nncorr", shape=nncorr.shape)
    d2[...] = nncorr
    d3 = f.create_dataset("deg_kblocks", shape=deg_kblocks.shape)
    d3[...] = deg_kblocks
    d4 = f.create_dataset("e0e1", shape=(2,)) # ground state energy and first excited state energy
    d4[...] = [e0, e1]
