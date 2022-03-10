"""Set of classes designed to deal with Fimo gff outputs. Only handles the case
where ONE motif was searched
"""

import numpy as np
from pprint import pprint

class FimoLine(object):

    def __init__(self, line=None):
        if line is not None:
            self.parse(line)
        else:
            self.patname=''
            self.tfname=''
            self.seqname=''
            self.start=None
            self.stop=None
            self.strand=''
            self.score=None
            self.pvalue=None
            self.qvalue=None
            self.matchedseq=''

    def parse(self, line):
        linearr = line.rstrip().split('\t')
        self.patname = linearr[0]
        self.tfname = linearr[1]
        self.seqname = linearr[2]
        # converts to normal 0-based coordinates
        self.start = int(linearr[3])-1
        self.stop = int(linearr[4])
        self.strand = linearr[5]
        try:
            self.score = float(linearr[6])
        except ValueError:
            self.score = None
        try:
            self.pvalue = float(linearr[7])
        except ValueError:
            self.pvalue = None
        try:
            self.qvalue = float(linearr[8])
        except ValueError:
            self.qvalue=None
        self.matchedseq=linearr[9]

class FimoSeq(object):

    def __init__(self,name=''):
        self.data = []
        self.name=name

    def __iter__(self):
        for line in self.data:
            yield line

    def append(self, fimoline):
        self.data.append(fimoline)

    def __len__(self):
        return len(self.data)

    def find_lines(self, findfunc, findall=True):
        matches = list(filter(findfunc, self.data))
        new_seq = FimoSeq(self.name)
        if len(matches) == 0:
            return new_seq
        if findall:
            for match in matches:
                new_seq.append(match)
        else:
            new_seq.append(match[0])
        return new_seq

class FimoFile(object):

    def __init__(self):
        self.data = {}
        self.names= []

    def parse(self, fname):
        with open(fname) as inf:
            # skip first line
            inf.readline()
            for i,line in enumerate(inf):
                # skip comments
                if line.startswith("#"):
                    continue
                if line == "\n":
                    continue
                this_line = FimoLine(line)
                if this_line.seqname in self.data:
                    self.data[this_line.seqname].append(this_line)
                else:
                    this_entry = FimoSeq(this_line.seqname)
                    this_entry.append(this_line)
                    self.data[this_line.seqname] = this_entry
                    self.names.append(this_line.seqname)

    def pull_entry(self, name):
        return self.data[name]

    def get_design_matrix(self, rec_db):
        # set up array of zeros with n_records rows. First column if for misses,
        #  second column is for hits.
        X = np.zeros((len(rec_db),2))
        for rec_name,rec_idx in rec_db.record_name_lut.items():
            # if this record's name is in the fimo hits, set its index in X to 1
            if rec_name in self.names:
                X[rec_idx,1] = 1
            else:
                X[rec_idx,0] = 1
        return X

    def __iter__(self):
        for name in self.names:
            yield self.data[name]
