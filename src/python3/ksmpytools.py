"""Set of classes designed to deal with Fimo gff outputs. Only handles the case
where ONE motif was searched
"""

import re
import numpy as np
from pprint import pprint
import ipdb

class Line:

    def __init__(self):
        self.motif_id=''
        self.seq_id=''
        self.motif_name=''
        self.seqname=None
        self.match=None
        self.seqpos=''
        self.coord=None
        self.strand=None
        self.score=None


class KSMLine(Line):

    def __init__(self, line=None):
        super().__init__()

        if line is not None:
            self.parse(line)

    def parse(self, line):
        linearr = line.rstrip().split('\t')
        self.motif_id=linearr[0]
        self.seq_id=linearr[1]
        self.motif_name=linearr[2]
        self.seqname=linearr[3]
        self.match=linearr[4]
        self.seqpos=linearr[5]
        self.coord=linearr[6]
        self.strand=linearr[7]
        self.score=linearr[8]


class KSMSeq(object):

    def __init__(self,name=''):
        self.data = []
        self.name=name

    def __iter__(self):
        for line in self.data:
            yield line

    def append(self, ksmline):
        self.data.append(ksmline)

    def __len__(self):
        return len(self.data)

    def find_lines(self, findfunc, findall=True):
        matches = list(filter(findfunc, self.data))
        new_seq = KSMSeq(self.name)
        if len(matches) == 0:
            return new_seq
        if findall:
            for match in matches:
                new_seq.append(match)
        else:
            new_seq.append(match[0])
        return new_seq


class KSMFile(object):

    def __init__(self):
        self.data = {}
        self.names= []

    def parse(self, fname):
        with open(fname, "r") as inf:
            # skip first three lines
            [inf.readline() for _ in range(3)]
            for i,line in enumerate(inf):
                # skip comments
                if line.startswith("#"):
                    continue
                if line == "\n":
                    continue
                this_line = KSMLine(line)
                self.insert_entry(this_line)

    def insert_entry(self, ksmline):
        if ksmline.seqname in self.data:
            self.data[ksmline.seqname].append(ksmline)
        else:
            this_entry = KSMSeq(ksmline.seqname)
            this_entry.append(ksmline)
            self.data[ksmline.seqname] = this_entry
            self.names.append(ksmline.seqname)

    def pull_entry(self, name):
        return self.data[name]

    def filter_by_id(self, identifier_list):
        newfimo = KSMFile()
        for name in self.names:
            for hit in self.pull_entry(name).data:
                if hit.motif_name in identifier_list:
                    newfimo.insert_entry(hit)
        return newfimo
 
    def gather_hits_dict(self, qval_thresh=1.0):
        hits = {}
        for seqname,hit_info in self.data.items():
            #print(dir(hit_info.data[0]))
            for hit in hit_info.data:
                motif_name = hit.motif_name
                if motif_name in hits:
                    hits[motif_name].append(seqname)
                else:
                    hits[motif_name] = [seqname]
        return hits

    def get_list(self):
        motif_list = []
        for seqname,hit_info in self.data.items():
            motif_name = hit_info.data[0].tfname
            motif_list.append(motif_name)
        motif_set = set(motif_list)
        return list(motif_set)

    def get_design_matrix(self, rec_db, qval_thresh=1.0, motif_list=None):
        var_lut = {}
        # set up array of zeros with n_records rows and n_motifs columns
        print("gathering hits dict")
        motif_hits = self.gather_hits_dict(qval_thresh)
        print("done gathering hits dict")
        X_list = []
        hit = 1
        for (i,motif) in enumerate(motif_list):
            X_i = motif.get_X(rec_db, motif_hits)
            X_list.append(X_i)
            var_lut[i] = {'motif_idx': i, 'hits': hit}
        X = np.stack(X_list, axis=1)
        #X = np.zeros((len(rec_db), len(motif_hits)))
        #for (i,motif) in enumerate(motif_list):
        #    motif_name = motif.alt_name
        #    rec_name_list = motif_hits[motif_name]
        #    for rec_name,rec_idx in rec_db.record_name_lut.items():
        #        # if this record's name is in the fimo hits, set its index in X to 1
        #        if rec_name in rec_name_list:
        #            X[rec_idx,i] = 1
        #    # matches var_lut format in shape motifs
        #    hit = 1
        #    var_lut[i] = {'motif_idx': i, 'hits': hit}
        return (X,var_lut)

    def __iter__(self):
        for name in self.names:
            yield self.data[name]

    def __len__(self):
        return len(self.data)


