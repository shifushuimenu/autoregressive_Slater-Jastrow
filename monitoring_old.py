"""Structures for CPU timing, monitoring matrix condition numbers, etc."""
import os 
import sys 
import posix

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
        self.num_onehop_states = 0
        self.elapsed_ref = 0.0
        self.elapsed_connecting_states = 0.0
        self.elapsed_singular = 0.0
        self.elapsed_adapt = 0.0

        self.counter_nonsingular = 0
        self.counter_singular = 0
        self.counter_skip = 0

        # Maximum condition number of the denominator matrix encountered 
        # during the simulation.
        self.Gdenom_cond_max = 0

    def accumulate(self):
        pass

    def print_summary(self, outfile="lowrank_stats.dat"):
        of = self._outfile if self._outfile is not None else outfile 
        fh = open(of, "a")
        fh.write("\n")
        fh.write( "Low-rank update of cond. probs. for one-hop states based on their reference state:\n")
        fh.write( "  num. onehop states = %d" % (self.num_onehop_states) + "\n")
        fh.write( "  elapsed_ref [s] =         %16.8f" % (self.elapsed_ref) + "\n")
        fh.write( "  elapsed_connecting_states=%16.8f" % (self.elapsed_connecting_states) + "\n")
        fh.write( "  elapsed_singular=         %16.8f" % (self.elapsed_singular) + "\n")
        fh.write( "  elapsed_adapt=            %d" % (self.elapsed_adapt) + "\n")        
        fh.write( "  counter_singular=         %d" % (self.counter_singular) + "\n")
        fh.write( "  counter_skip=             %d" % (self.counter_skip) + "\n")
        fh.write( "  counter_nonsingular=      %d" % (self.counter_nonsingular) + "\n")
        fh.write( "  max. cond. number (Gdenom) = %16.8f" % (self.Gdenom_cond_max))             
        fh.close()

class logger(object):
    # class variable: just one instance  
    info_refstate = _info_refstate()        
