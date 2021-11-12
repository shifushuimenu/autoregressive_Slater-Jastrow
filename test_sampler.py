# Problem: Direct sampling of the Slater determinant only samples with 
# the correct probabilities if no order on the positions is imposed. 

from test_suite import *
from Slater_Jastrow_simple import *

(Nsites, eigvecs) = prepare_test_system_zeroT(Nsites=10, potential='none', PBC=False)
Nparticles = 5
SDsampler = SlaterDetSampler(eigvecs, Nparticles=Nparticles)

for i in range(10):
    config, prob = SDsampler.sample()
    print("=====================================================")
    prob_chaineval = SDsampler.chainevaluate_amplitude(config)**2

    print("config", config)
    print("prob=%.12f" % prob)
    print("prob_chaineval=%.12f" % prob_chaineval)
    print("prob from overlap=%.12f" % SDsampler.psi_amplitude([config])**2)