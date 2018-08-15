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

if __name__ == "__main__":
    import numpy as np
    data = np.random.normal(30, 10, 10000)
    online_mean = Welford()
    for val in data:
        online_mean.update(val)
    welford_mean = online_mean.final_mean()
    welford_stdev = online_mean.final_stdev()
    actual_mean = np.mean(data)
    actual_stdev = np.std(data, ddof=1)
    print "Mean"
    print "Welfords: %s Actual: %s"%(welford_mean, actual_mean)
    print "Stdev"
    print "Welfords: %s Actual: %s"%(welford_stdev, actual_stdev)



