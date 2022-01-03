from test_suite import ( 
        prepare_test_system_zeroT,
        Slater2spOBDM
    )
from slater_sampler_ordered_tmp import *

(Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=10, potential='none', PBC=False, HF=False)
Nparticles = 5
num_samples = 4

SDsampler1 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive=True)
SDsampler2 = SlaterDetSampler_ordered(Nsites=Nsites, Nparticles=Nparticles, single_particle_eigfunc=eigvecs, naive=False)

occ_vec, prob_sample = SDsampler1.sample()
occ_vec, prob_sample = SDsampler2.sample()
