import math

class Welford(object):

    def __init__(self):
        self.mean = 0.0
        self.sqdist = 0.0
        self.tot = 0.0

    def update(self, newval):
        self.tot += 1.0
        delta = newval-self.mean
        self.mean += delta/self.tot
        delta2 = newval-self.mean
        self.sqdist += delta * delta2

    def final_mean(self):
        return self.mean
    def final_stdev(self):
        return math.sqrt(self.sqdist/(self.tot-1.0))

