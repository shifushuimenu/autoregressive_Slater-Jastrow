# IMPROVE: Organize imports globally in a better way
#          when making a package with an  __init__.py file. 

from Slater_Jastrow_simple import *
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from slater_sampler_ordered import SlaterDetSampler_ordered
from test_suite import prepare_test_system_zeroT, HartreeFock_tVmodel
from monitoring_old import logger 

from time import time 
import matplotlib.pyplot as plt

Parallel = False 

if Parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    master = 0
    print("MPI: proc %d of %d" % (MPI_rank, MPI_size))
else:
    MPI_rank = 0
    MPI_size = 1
    master = 0

torch.set_default_dtype(default_dtype_torch)
torch.autograd.set_detect_anomaly(True)

# set random number seed
use_cuda = False
seed = 44 + MPI_rank
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

max_iter = 10 #1000 
num_samples = 10 # 100  # samples per batch
num_bin = 5 #50
Nx = 5  # 15
Ny = 5
Nsites = Nx*Ny  # 15  # Nsites = 64 => program killed because it is using too much memory
space_dim = 2
Nparticles = 12

optimize_orbitals = True # whether to include columns of P-matrix in optimization
learning_rate_SD = 0.02

#Vint_array = np.array([1.0, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
Vint = 10.0 #  Vint_array[MPI_rank]

param_suffix = "_Nx{}Ny{}Np{}V{}".format(Nx, Ny, Nparticles, Vint)
logger.info_refstate.outfile = "lowrank_stats"+param_suffix+".dat"

# for debugging:
# If deactivate_Jastrow == True, samples are drawn from the Slater determinant without the Jastrow factor. 
deactivate_Jastrow = False

def train(VMCmodel, learning_rate, learning_rate_SD, num_samples=100, num_bin=50, use_cuda=False):
    '''
    train a model using stochastic gradient descent 

    Args:
        model (obj): a model that meets VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    if use_cuda:
        VMCmodel.ansatz.cuda()

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.        
        sample_list = np.zeros((num_samples, Nsites)) 
        log_probs = np.zeros((num_samples,))
        print("before sampling")
        with torch.no_grad():
            t1 = time()
            for i in range(num_samples):
                sample_unfolded, log_prob_sample = VMCmodel.ansatz.sample_unfolded()
                log_probs[i] = log_prob_sample
                sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
            t2 = time()
            VMCmodel.t_sampling += (t2-t1)

        energy, grad, energy_grad, precision = vmc_measure(VMCmodel.local_measure, sample_list, log_probs, num_bin=num_bin)

        # update variables using stochastic gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for (name, par), g in zip(VMCmodel.ansatz.named_parameters(), g_list):
            if name == 'slater_sampler.P':
                delta = learning_rate_SD * g
            else:
                delta = learning_rate * g
            par.data -= delta

        # re-orthogonalize the columns of the Slater determinant
        # and update bias of the zero-th component 
        if isinstance(VMCmodel.ansatz, SlaterJastrow_ansatz):
            print("slater_sampler.P.grad=", VMCmodel.ansatz.slater_sampler.P.grad)
            print("named parameters", list(VMCmodel.ansatz.named_parameters()))

            if VMCmodel.ansatz.slater_sampler.optimize_orbitals:
                VMCmodel.ansatz.slater_sampler.reortho_orbitals()

            VMCmodel.ansatz.slater_sampler.reset_sampler()
            VMCmodel.ansatz.bias_zeroth_component[:] = VMCmodel.ansatz.slater_sampler.get_cond_prob(k=0)

        yield energy, precision
        
        

# visualize the loss history
energy_list, precision_list = [], []
av_list, sigma_list = [], []
def _update_curve(energy, precision):
    energy_list.append(energy)
    precision_list.append(precision)
    Nb = len(energy_list)
    av = np.sum(energy_list) / Nb
    sigma = np.sqrt((np.sum([s**2 for s in precision_list]) / Nb) / Nb)
    av_list.append(av)
    sigma_list.append(sigma)
    if len(energy_list)%(max_iter-1) == 0:
        xvals = np.arange(1, len(energy_list) + 1)
        #plt.errorbar(xvals, energy_list, yerr=precision_list, capsize=3, label="Slater-Jastrow")
        #plt.errorbar(xvals, av_list, yerr=sigma_list, capsize=3)
        ## dashed line for exact energy
        #plt.axhline(E_exact, ls='--', label="exact")
        #plt.title("$L$=%d, $N$=%d, $V/t$ = %4.4f" % (Nsites, Nparticles, Vint))
        #plt.legend(loc="upper right")
        #plt.show()

    MM = np.hstack((np.array(energy_list)[:,None], np.array(precision_list)[:,None],
                    np.array(av_list)[:,None], np.array(sigma_list)[:,None]))
    np.savetxt("energies"+param_suffix+".dat", MM)


ckpt_outfile = "state"+param_suffix+".pt"
def _checkpoint(VMCmodel):
    """Save most recent SJA state to disk."""
    state = {
        "energy": energy,
        "precision": precision, 
        "net": VMCmodel.ansatz.state_dict()
    }
    torch.save(state, ckpt_outfile)


phys_system = PhysicalSystem(nx=Nx, ny=Ny, ns=Nsites, num_particles=Nparticles, D=space_dim, Vint=Vint)

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
fh = open("HF_energy"+param_suffix+".dat", "w")
(eigvals, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none", outfile=fh, max_iter=20)
np.savetxt("eigvecs.dat", eigvecs)
#(_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none', HF=True, PBC=False, Nparticles=Nparticles, Vnnint=Vint)

Sdet_sampler = SlaterDetSampler_ordered(
        Nsites=Nsites, 
        Nparticles=Nparticles, 
        single_particle_eigfunc=eigvecs, 
        eigvals=eigvals, 
        naive_update=False, 
        optimize_orbitals=optimize_orbitals
        )

SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, 
        num_components=Nparticles, 
        D=Nsites, 
        net_depth=2, 
        deactivate_Jastrow=deactivate_Jastrow
        )

VMCmodel_ = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA

E_exact = -3.6785841210741 #-3.86925667 # 0.4365456400025272 #-3.248988339062832 # -2.9774135797163597 #-3.3478904193465335


if True: 
    t0 = time()
    t0_tmp = t0
    for i, (energy, precision) in enumerate(train(VMCmodel_, learning_rate=0.2, learning_rate_SD=learning_rate_SD, num_samples=num_samples, num_bin=num_bin, use_cuda = use_cuda)):
        t1_tmp = time()
        print('Step %d, dE/|E| = %.4f, elapsed = %.4f' % (i, -(energy - E_exact)/E_exact, t1_tmp-t0_tmp))
        _update_curve(energy, precision)
        _checkpoint(VMCmodel_)
        t0_tmp = time()
        # stop condition
        if i >= max_iter:
            break
    t1 = time()
    print("## Timings:")
    print("## elapsed =%10.6f for %d samples per iteration with %d iterations" % (t1-t0, num_samples, max_iter))
    print("## t_psiloc=", VMCmodel_.t_psiloc)
    print("## t_logprob_B=", VMCmodel_.ansatz.t_logprob_B)
    print("## t_logprob_F=", VMCmodel_.ansatz.t_logprob_F)
    print("## t_sampling=", VMCmodel_.t_sampling)
    print("## t_locE=", VMCmodel_.t_locE)
    print("## t_backward=", VMCmodel_.t_grads)


szsz_corr = np.zeros(Nsites)
szsz_corr_2D = np.zeros((phys_system.nx, phys_system.ny))
corr_ = np.zeros(Nsites)
corr_2D_ = np.zeros((phys_system.nx, phys_system.ny))


print("Now sample from the converged ansatz")
state_checkpointed = torch.load(ckpt_outfile)
VMCmodel_.ansatz.load_state_dict(state_checkpointed['net'])
num_samples = 10
for _ in range(num_samples):
    sample_unfolded, log_prob_sample = VMCmodel_.ansatz.sample_unfolded()
    config = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
    print("config=", config) 
    config_sz = 2*config - 1
    corr_[:] = 0.0
    for k in range(0, Nsites):
        corr_[k] = (np.roll(config_sz, shift=-k) * config_sz).sum(axis=-1) / Nsites
    szsz_corr[:] += corr_[:]

    # 2D spin-spin correlations 
    config_2D = config.reshape((phys_system.nx, phys_system.ny))
    config_2D_sz = 2*config_2D - 1
    corr_2D_[:,:] = 0.0
    for kx in range(0, phys_system.nx):
        for ky in range(0, phys_system.ny):
           corr_2D_[kx, ky] = (np.roll(np.roll(config_2D_sz, shift=-kx, axis=0), shift=-ky, axis=1) * config_2D_sz).sum() / phys_system.ns
    szsz_corr_2D[:,:] += corr_2D_[:, :]

szsz_corr[:] /= num_samples
szsz_corr_2D[:,:] /= num_samples

np.savetxt("szsz_corr"+param_suffix+".dat", szsz_corr)
np.savetxt("szsz_corr_2D.dat", szsz_corr_2D)

#plt.plot(range(Nsites), szsz_corr[:], '--b')
#plt.show()
