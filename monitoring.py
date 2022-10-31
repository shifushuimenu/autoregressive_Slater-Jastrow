"""Structures for CPU timing, monitoring matrix condition numbers, etc."""
import os 
import sys
import posix

from accumulator import AccumulatorWithVariance 

class _info_refstate(object):
    """
    Stores elapsed time for different steps of the low-rank update 
    of conditional probabilities of one-hop states based on a reference state.
    """
    def __init__(self):
        self._outfile = None
        self.reset()

    @property
    def outfile(self):
        """Output file for statistics of lowrank update for kinetic energy.""" 
        return self._outfile

    @outfile.setter
    def outfile(self, outfile):
        self._outfile = outfile 
        if os.path.exists(outfile):
            print("Removing existing file: %s" % self._outfile, file=sys.stderr)
            posix.remove(outfile)
        
    def reset(self):
        self.accumulator = AccumulatorWithVariance(
            "num_onehop_states", 
            "elapsed_ref",
            "elapsed_connecting_states", 
            "elapsed_singular", 
            "elapsed_adapt", 
            "counter_nonsingular",
            "counter_singular",
            "counter_skip", 
            "size_support"  # size of the support of the conditional probs. (for all connecting states)
        )

        # Maximum condition number of the denominator matrix encountered 
        # during the simulation.
        self.Gdenom_cond_max = 0

    def print_summary(self, outfile="lowrank_stats.dat"):
        of = self._outfile if self._outfile is not None else outfile 
        fh = open(of, "w")
        fh.write("\n")
        fh.write( "Low-rank update of cond. probs. for one-hop states based on their reference state:\n")
        fh.write(str(self.accumulator))
        fh.write( "  max. cond. number (Gdenom) = %16.8f\n" % (self.Gdenom_cond_max))             
        fh.close()

class logger(object):
    # class variable: just one instance  
    info_refstate = _info_refstate()        
