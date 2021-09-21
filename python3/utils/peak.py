""" peak.py

Classes and functions to deal with peaks in genomic coordinates

Written by Michael Wolfe
"""

import numpy as np

class Peak(object):

    @classmethod
    def from_line(cls, line):
        linearr = line.rstrip().split("\t")
        while len(linearr) < 10:
            linearr.append(-1)
        return(cls(chrm=linearr[0], start=int(linearr[1]), end=int(linearr[2]),
               name=linearr[3], score=float(linearr[4]), strand=linearr[5],
               signalval=float(linearr[6]), pval=float(linearr[7]), qval=float(linearr[8]),
               peak=int(linearr[9])))

    def __init__(self, chrm=".", start=-1, end=-1, name=".", score=-1, 
                 strand=".", signalval=-1, pval=-1, qval=-1, peak=-1):

        self.chrm = chrm
        self.start = start
        self.end = end
        self.name = name
        self.score = score
        self.strand = strand
        self.signalval = signalval
        self.pval = pval
        self.qval = qval
        self.peak = peak
        self.density = None
        self.condition = None
    
    def __str__(self):
        return ("%s\t"*9)%(self.chrm, self.start, self.end, self.name,
                           self.score, self.strand, self.signalval, self.pval,
                           self.qval) + "%s"%self.peak
    def __len__(self):
        return self.end - self.start

    def add_density(self, array):
        self.density = array

    def find_density_center(self):
        """ This assumes that the density is only what is contained within the
        peak and no NaNs or infs are in that array.
        """
        # find the first location in the array where cumulative sum/sum is 
        # over 50 %
        # first mask nans:
        nanmask = np.isfinite(self.density)
        index_vals = np.where(nanmask)[0]
        center_index = np.where(self.density[nanmask].cumsum()/self.density[nanmask].sum() > 0.5)[0].min()
        return int(self.start + index_vals[center_index])

    def find_geometric_center(self):
        return int(self.start + (self.end-self.start)/2)

    def find_height_center(self):
        if self.peak >= 0:
            center = self.start+self.peak
        else:
            center = self.find_geometric_center()
        return int(center)

    def add_condition(self, val):
        self.condition = val



class PeakList(object):
    def __init__(self):
        self.data = []

    def add_Peak(self, peak):
        self.data.append(peak)

    def from_narrowPeak_file(self, filename):
        with open(filename, mode="r") as f:
            for line in f:
                self.data.append(Peak.from_line(line))


    def write_narrowPeak_file(self, filename):
        with open(filename, mode = 'w') as f:
            for peak in self.data:
                f.write(str(peak) + "\n")
    def generator(self):
        for peak in self.data:
            yield peak

    def to_array(self, array):
        for peak in self.data:
            array[peak.start:peak.end] = True

    def from_array(self, array):
        changes = np.abs(np.diff(np.concatenate(([0], array.view(np.int8), [0]))))
        start_stops = np.where(changes)[0].reshape(-1,2)
        for start, stop in start_stops:
            self.add_Peak(Peak(start=start, end=stop))
    
    def filter_peaks(self, filter_func):
        new_peak_list = PeakList()
        new_peak_list.data = filter(filter_func, self.data)
        return new_peak_list

    def __len__(self):
        return len(self.data)

class PeakCluster(PeakList):

    def __init__(self, start = -1, end=-1):
        self.data = []
        self.start = start
        self.end = end
        self.annotations = []

    def update_size(self):
        for peak in self.data:
            if peak.start < self.start:
                self.start = peak.start
            if peak.end < self.end:
                self.end = peak.end

    def add_annotation(self, entry):
        self.annotations.append(entry)
