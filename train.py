# IMPROVE: Organize imports globally in a better way
#          when making a package with an  __init__.py file. 

from Slater_Jastrow_simple import *
#from slater_sampler_ordered_memory_layout import SlaterDetSampler_ordered
from slater_sampler_ordered import SlaterDetSampler_ordered
import time
import matplotlib.pyplot as plt
from test_suite import prepare_test_system_zeroT, HartreeFock_tVmodel

torch.set_default_dtype(default_dtype_torch)
torch.autograd.set_detect_anomaly(True)

# set random number seed
use_cuda = False
seed = 42
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def train(VMCmodel, learning_rate, num_samples=100, use_cuda=False):
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
        sample_probs = np.zeros((num_samples,))
        print("before sampling")
        with torch.no_grad():
            for i in range(num_samples):
                sample_unfolded, sample_prob = VMCmodel.ansatz.sample_unfolded()
                sample_probs[i] = sample_prob
                sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()

        print("before vmc_measure")
        energy, grad, energy_grad, precision = vmc_measure(VMCmodel.local_measure, sample_list, sample_probs, num_bin=50)

        # update variables using stochastic gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(VMCmodel.ansatz.parameters(), g_list):
            delta = learning_rate * g
            var.data -= delta

        # re-orthogonalize the columns of the Slater determinant
        # and update bias of the zero-th component 
        if isinstance(VMCmodel.ansatz, SlaterJastrow_ansatz):
            print("slater_sampler.P.grad=", VMCmodel.ansatz.slater_sampler.P.grad)
            print("named parameters", list(VMCmodel.ansatz.named_parameters()))
            VMCmodel.ansatz.slater_sampler.reortho_orbitals()
            VMCmodel.ansatz.slater_sampler.reset_sampler()
            VMCmodel.ansatz.bias_zeroth_component[:] = VMCmodel.ansatz.slater_sampler.get_cond_prob(k=0)

        yield energy, precision
        
        

# visualize the loss history
energy_list, precision_list = [], []
def _update_curve(energy, precision):
    energy_list.append(energy)
    precision_list.append(precision)
    if len(energy_list)%(max_iter-1) == 0:
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3, label="Slater-Jastrow")
        # dashed line for exact energy
        plt.axhline(E_exact, ls='--', label="exact")
        plt.title("$L$=%d, $N$=%d, $V/t$ = %4.4f" % (Nsites, Nparticles, Vint))
        plt.legend(loc="upper right")
        plt.show()

    MM = np.hstack((np.array(energy_list)[:,None], np.array(precision_list)[:,None]))
    np.savetxt("energies_Ns{}Np{}V{}.dat".format(Nsites, Nparticles, Vint), MM)


def _checkpoint(VMCmodel):
    """Save most recent SJA state to disk."""
    state = {
        "energy": energy,
        "precision": precision, 
        "net": VMCmodel.ansatz.state_dict()
    }
    torch.save(state, 'state_Ns{}Np{}V{}.pt'.format(Nsites, Nparticles, Vint))




max_iter = 1000 
Nsites = 9 # Nsites = 64 => program killed because it is using too much memory
Nparticles = 4
Vint = 2.5


phys_system = PhysicalSystem(nx=3, ny=3, ns=Nsites, np=Nparticles, D=2, Vint=Vint)

# Aggregation of MADE neural network as Jastrow factor 
# and Slater determinant sampler. 
(_, eigvecs) = HartreeFock_tVmodel(phys_system, potential="none")
#(_, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none', HF=True, PBC=False, Nparticles=Nparticles, Vnnint=Vint)
Sdet_sampler = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive=False)
SJA = SlaterJastrow_ansatz(slater_sampler=Sdet_sampler, num_components=Nparticles, D=Nsites, net_depth=2)

VMCmodel_ = VMCKernel(energy_loc=phys_system.local_energy, ansatz=SJA)
del SJA

E_exact = 0.4365456400025272 #-3.248988339062832 # -2.9774135797163597 #-3.3478904193465335

t0 = time.time()
for i, (energy, precision) in enumerate(train(VMCmodel_, learning_rate = 0.1, num_samples=100, use_cuda = use_cuda)):
    t1 = time.time()
    print('Step %d, dE/|E| = %.4f, elapsed = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
    _update_curve(energy, precision)
    _checkpoint(VMCmodel_)
    t0 = time.time()

    # stop condition
    if i >= max_iter:
        break

szsz_corr = np.zeros(Nsites)
corr_ = np.zeros(Nsites)


print("Now sample from the converged ansatz")
dd = torch.load('state_Ns{}Np{}V{}.pt'.format(Nsites, Nparticles, Vint))
VMCmodel_.ansatz.load_state_dict(dd['net'])
num_samples = 1000
for _ in range(num_samples):
    sample_unfolded, sample_prob = VMCmodel_.ansatz.sample_unfolded()
    config = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
    print("config=", config)       
    config_sz = 2*config - 1
    corr_[:] = 0.0
    for k in range(0, Nsites):
        corr_[k] = (np.roll(config_sz, shift=-k) * config_sz).sum(axis=-1) / Nsites
    szsz_corr[:] += corr_[:]

szsz_corr[:] /= num_samples

np.savetxt("szsz_corr_Ns{}Np{}V{}.dat".format(Nsites, Nparticles, Vint), szsz_corr)

plt.plot(range(Nsites), szsz_corr[:], '--b')
plt.show()
