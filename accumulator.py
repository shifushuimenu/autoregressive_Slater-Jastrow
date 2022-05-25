"""Overload dictionary methods in order to record statistics about how often dictionary items were set and with which values (i.e. compute average and standard deviation)."""
from math import sqrt

class AccumulatorWithVariance(dict):
    """accumulate values for dictionary entries and calculate their average and variance on the fly"""
    def __init__(self, *accumulator_keys):    
        self.accumulator_keys = accumulator_keys
        self._counter = {}
        self._av, self._av2 = {}, {}
        for k in self.accumulator_keys:
            self._av.setdefault(k, 0)
            self._av2.setdefault(k, 0)
            self._counter.setdefault(k, 0)
            
    def __setitem__(self, key, value):        
        """update average and variance"""
        coeff = 1.0/(self._counter[key]+1)
        self._av[key] = coeff*(self._counter[key]*self._av[key]+value)
        self._av2[key] = coeff*(self._counter[key]*self._av2[key]+value**2)
        self._counter[key] += 1
        
    def __getitem__(self, key):
        """return average, standard deviation and number of counts of dictionary entry"""
        sigma = sqrt(self._av2[key] - self._av[key]**2)
        return (self._av[key], sigma, self._counter[key])
        
    def __repr__(self):
        """If acc is an AccumulatorWithVariance object, use 
              fh.write(str(acc))
           to output the statistics of all entries to the file handle fh.
        """
        s = ""
        for k in self.accumulator_keys:
            s += "%s = %e +/- %e (counts = %d)" % (k, *self.__getitem__(k)) + "\n"
        return s 


if __name__ == "__main__":
    
    acc = AccumulatorWithVariance("name1", "name2", "name3")
    acc["name1"] = 10
    acc["name1"] = 20
    acc["name1"] = 30
    acc["name2"] = 30
    acc["name2"] = 30
   
    with open("test.dat", "w") as fh:
        fh.write(str(acc))
    
    
