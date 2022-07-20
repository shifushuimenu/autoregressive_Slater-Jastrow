import numpy as np
import torch 
from time import time 

from one_hot import occ_numbers_collapse 


def binning_statistics(obs_list, num_bin):
    '''
    calculate avriance and autorcorrelation time for a  list of observable values
    using the binning method
    '''
    num_sample = len(obs_list)
    if num_sample % num_bin != 0:
        raise ValueError(f"num_bin = {num_bin}")
    size_bin = num_sample // num_bin

    mean = np.mean(obs_list, axis=0)
    variance = np.var(obs_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var(
        [np.mean(obs_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin * \
        np.abs(np.mean(variance_binned) / np.mean(variance))
    stderr = np.sqrt(variance_binned / num_bin)
    print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
          (mean, stderr, t_auto))
    return mean, stderr

# =========================================================================
# Maybe incorporate stochastic reconfiguration into the Trainer class ? 

def train_SR(VMCmodel, learning_rate, learning_rate_SD, precond, num_samples=100, num_bin=50, use_cuda=False):
    '''
    train a model using stochastic gradient descent with stochastic reconfiguration 
    as a preconditioner 

    Args:
        model (obj): a model that meets VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    if use_cuda:
        VMCmodel.ansatz.cuda()

    Nsites = VMCmodel.ansatz.D

    # get expectation values for energy, gradient and their product,
    # as well as the precision of energy.        
    sample_list = np.zeros((num_samples, Nsites)) 
    log_probs = np.zeros((num_samples,))
    with torch.no_grad():
        t1 = time()
        for i in range(num_samples):
            sample_unfolded, log_prob_sample = VMCmodel.ansatz.sample_unfolded()
            log_probs[i] = log_prob_sample
            sample_list[i] = occ_numbers_collapse(sample_unfolded, Nsites).numpy()
        t2 = time()
        VMCmodel.t_sampling += (t2-t1)

    # gradient of the energy
    # grad_k = <H * Delta_k> - <H>*<Delta_k>
    av_H, av_Ok, av_HtimesOk, precision = vmc_measure(VMCmodel.local_measure, sample_list, log_probs, precond=precond, num_bin=num_bin)
    g_list = [av_HtimesOk - av_H*av_Ok for (av_HtimesOk, av_Ok) in zip(av_HtimesOk, av_Ok)]

    # stochastic reconfiguration: 
    # g = S^{-1} * g        
    t1 = time()
    g_list = precond.apply_Sinv(g_list, tol=1e-8)
    t2 = time()
    VMCmodel.t_SR += (t2-t1)

    for (name, par), g in zip(VMCmodel.ansatz.named_parameters(), g_list):
        #if name == 'slater_sampler.T':
        #    delta = learning_rate_SD * g
        #else:
        delta = learning_rate * g
        par.data -= delta

    VMCmodel.ansatz.slater_sampler.rotate_orbitals()

    return av_H, precision


#@profile
def vmc_measure(local_measure, sample_list, log_probs, precond, num_bin=50):
    '''
    get energy, gradient and there product averaged over a batch of samples 

    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of configurations.
        log_probs (lit): the log probs for the configurations.
        num_bin (int): number of bins in binning statistics.

    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    energy_loc_list, grad_loc_list = [], []
    for i, (config, log_prob) in enumerate(zip(sample_list, log_probs)):
        # back-propagation is used to get gradients.
        energy_loc, grad_loc = local_measure([config], log_prob) # ansatz.psi requires batch dim)
        energy_loc_list.append(energy_loc)
        grad_loc_list.append(grad_loc)

 
        precond.accumulate(grad_loc)


    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))

    return energy.item(), grad_mean, energy_grad, energy_precision

# =========================================================================

class Trainer(object):
    """doc string"""
    def __init__(self, VMCmodel, learning_rate, optim_name, num_samples=100, num_bin=50, use_cuda=False):

        if use_cuda:
            VMCmodel.ansatz.cuda()
        self.VMCmodel = VMCmodel
        self.Ns = self.VMCmodel.ansatz.D
        self.num_samples = num_samples # number of samples per training epoch
        self.num_bin = num_bin

        self.energy = None  # average energy in the current epoch 
        self.precision = None # variance of energy in the current epoch 

        if optim_name in ["SGD"]:
            self.optimizer = torch.optim.SGD(self.VMCmodel.ansatz.parameters(), lr=learning_rate, momentum=0.9)
        elif optim_name in ["Adam"]:
            self.optimizer = torch.optim.Adam(self.VMCmodel.ansatz.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
        elif optim_name in ["RMSprop"]:
            self.optimizer = torch.optim.RMSprop(self.VMCmodel.ansatz.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08) 
        else:
            raise ValueError(f"Unknown optimizer name {optim_name}")

    def _reinforcement_loss_fn(self, config_list):

        energy_list = []  # list of numbers
        log_psi_list = [] # list of torch tensors 

        for config in config_list:

            # density estimation
            psi_loc = self.VMCmodel.ansatz.psi_amplitude(torch.from_numpy(np.array([config]))) # requires batch dimension 
            log_psi = torch.log(torch.abs(psi_loc))
            assert log_psi.requires_grad
            # calculation of local energy 
            with torch.no_grad():
                eloc = self.VMCmodel.energy_loc(np.array([config]), psi_loc.data, ansatz=self.VMCmodel.ansatz).item()

            energy_list.append(eloc)
            log_psi_list.append(log_psi)

        av_local_energy = np.mean(energy_list)

        # loss from reinforcement learning 
        loss = torch.sum(torch.tensor([log_psi_list[i] * (energy_list[i] - av_local_energy) for i in range(self.num_samples)], requires_grad=True))

        assert loss.requires_grad 

        # store current av. energy and error 
        (ene, std_ene) = binning_statistics(energy_list, num_bin=self.num_bin)
        self.energy = ene 
        self.precision = std_ene 

        return loss 


    def train_standard_optimizer(self):

        # generate a list of samples for this training epoch 
        config_list = np.zeros((self.num_samples, self.Ns)) 
        log_probs = np.zeros((self.num_samples,))
        with torch.no_grad():
            t1 = time()
            for i in range(self.num_samples):
                sample_unfolded, log_prob_sample = self.VMCmodel.ansatz.sample_unfolded()
                log_probs[i] = log_prob_sample
                config_list[i] = occ_numbers_collapse(sample_unfolded, self.Ns).numpy()
            t2 = time()
            self.VMCmodel.t_sampling += (t2-t1)

        self.optimizer.zero_grad()
        loss = self._reinforcement_loss_fn(config_list)
        loss.backward()
        self.optimizer.step()

        return self.energy, self.precision 
    
    