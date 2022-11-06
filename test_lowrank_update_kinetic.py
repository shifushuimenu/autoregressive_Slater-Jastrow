"""
Check that the conditional probs calculated via lowrank update from the reference state alpha
agree with the conditional probs output by the slater sampler for each one-hop state
"""
import unittest 
import numpy as np

from VMC_common import PhysicalSystem
from k_copy import sort_onehop_states 
from bitcoding import bin2int, int2bin, bin2pos, generate_random_config
from one_hot import occ_numbers_unfold 
from physics import kinetic_term
from test_suite import HartreeFock_tVmodel
from slater_sampler_ordered import SlaterDetSampler_ordered
from SlaterJastrow_ansatz import _cond_prob2log_prob


class TestLRUpdate(unittest.TestCase):

    def _init_objects(self, Lx=6, Ly=6, Nparticles=7, Vint=12.0):

        self.Nsites = Lx*Ly
        self.Nparticles = Nparticles 
        self.phys_system = PhysicalSystem(nx=Lx, ny=Ly, ns=Lx*Ly, num_particles=Nparticles, 
                dim=2, Vint=Vint)
        (eigvals, eigvecs) = HartreeFock_tVmodel(self.phys_system, potential="none", 
                max_iter=20)
        self.SDsampler = SlaterDetSampler_ordered(
                Nsites=self.Nsites, 
                Nparticles=self.Nparticles, 
                single_particle_eigfunc=eigvecs, 
                eigvals=eigvals, 
                naive_update=False, 
                optimize_orbitals=False,
                outdir="./"
                )

    def test_lowrank_kinetic(self):
        self._init_objects()

        alpha = generate_random_config(self.Nsites, self.Nparticles)
        ref_I = bin2int(alpha)
        rs_pos, xs_I, matrix_elem = sort_onehop_states(*kinetic_term(int(ref_I), self.phys_system.lattice))

        cond_prob_onehop, cond_prob_refstate = self.SDsampler.lowrank_kinetic(ref_I, xs_I, rs_pos)
        xs = int2bin(xs_I, self.Nsites)
        xs_unfolded = occ_numbers_unfold(xs)
        xs_pos = bin2pos(xs)
        log_probs1 = _cond_prob2log_prob(xs_hat=cond_prob_onehop, xs_unfolded=xs_unfolded, xs_pos=xs_pos, 
            num_components=self.Nparticles, D=self.Nsites)
        log_probs2 = self.SDsampler.log_prob(xs)
        assert np.all(np.isclose(log_probs1, log_probs2, atol=1e-16))

    def test_lowrank_Coulomb(self):
        assert 0 == 0

if __name__ == "__main__":
    unittest.main()
