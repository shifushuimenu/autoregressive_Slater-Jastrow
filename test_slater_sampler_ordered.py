import unittest 

import numpy as np
import torch 

from slater_sampler_ordered import SlaterDetSampler_ordered 
from HF import prepare_test_system_zeroT
from slater_determinant import Slater_determinant_overlap, Slater2spOBDM
from one_hot import occ_numbers_collapse
from bitcoding import bin2int 

import matplotlib.pyplot as plt 

class TestSlaterSampler(unittest.TestCase):

    def test_direct_sampling(self, visualize=False):

        Nsites = 10 
        Nparticles = 5
        dim = 2**Nsites 

        (Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=Nsites, potential='none', PBC=False, HF=False)
        P_SD = eigvecs[:, 0:Nparticles]

        SDsampler = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive_update=False)

        num_samples = 100
        probs_sampler = np.zeros(dim)
        probs_SD = np.zeros(dim)
        for i in range(num_samples):
            occ_vec, prob_sample = SDsampler.sample()
            state_index = np.asarray(bin2int(np.asarray(occ_vec, dtype=np.int64)), dtype=np.int64) # to messy typecasts are due to the fact that output type of bin2int is 'object' to allow long integers => clean this up somehow
            probs_sampler[state_index] = np.exp(SDsampler.log_prob(occ_vec).item())
            probs_SD[state_index] = np.abs(Slater_determinant_overlap(occ_vec, P_SD))**2

        if visualize:
            plt.plot(range(dim), probs_SD, '-ob', label="probs_SD")
            plt.plot(range(dim), probs_sampler, '-or', label="probs_sampler")
            plt.legend(loc="best")
            plt.show()

        assert np.all(np.isclose(probs_SD, probs_sampler, atol=1e-8))

if __name__ == "__main__":
   
   unittest.main()
