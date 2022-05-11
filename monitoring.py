"""Structures for CPU timing, monitoring matrix condition numbers, etc."""

class _info_refstate(object):
    """
    Stores elapsed time for different steps of the low-rank update 
    of conditional probabilities of one-hop states based on a reference state.
    """
    def __init__(self):
        self.reset()

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

    def print_summary(self):        
        print( "Low-rank update of cond. probs. for one-hop states based on their reference state:\n"
               "  num. onehop states = %d" % (self.num_onehop_states) + "\n" +
               "  elapsed_ref [s] =         %16.8f" % (self.elapsed_ref) + "\n" +
               "  elapsed_connecting_states=%16.8f" % (self.elapsed_connecting_states) + "\n" +
               "  elapsed_singular=         %16.8f" % (self.elapsed_singular) + "\n" +
               "  elapsed_adapt=            %d" % (self.elapsed_adapt) + "\n" +              
               "  counter_singular=         %d" % (self.counter_singular) + "\n" +
               "  counter_skip=             %d" % (self.counter_skip) + "\n" +
               "  counter_nonsingular=      %d" % (self.counter_nonsingular) + "\n" + 
               "  max. cond. number (Gdenom) = %16.8f" % (self.Gdenom_cond_max) + "\n"
             )


class logger(object):
    info_refstate = _info_refstate()        
