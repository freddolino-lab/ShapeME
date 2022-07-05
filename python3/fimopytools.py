"""Set of classes designed to deal with Fimo gff outputs. Only handles the case
where ONE motif was searched
"""

import re
import numpy as np
from pprint import pprint

class MemeLine:

    def __init__(self):
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


class FimoLine(MemeLine):

    def __init__(self, line=None):
        super().__init__()

        if line is not None:
            self.parse(line)

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
                self.insert_entry(this_line)

    def insert_entry(self, fimoline):
        if fimoline.seqname in self.data:
            self.data[fimoline.seqname].append(fimoline)
        else:
            this_entry = FimoSeq(fimoline.seqname)
            this_entry.append(fimoline)
            self.data[fimoline.seqname] = this_entry
            self.names.append(fimoline.seqname)

    def pull_entry(self, name):
        return self.data[name]

    def filter_by_id(self, identifier_list):
        newfimo = FimoFile()
        for name in self.names:
            hit = self.pull_entry(name).data[0]
            if hit.patname in identifier_list:
                newfimo.insert_entry(hit)
        return newfimo
 
    def gather_hits_dict(self, pval_thresh=1.0):
        hits = {}
        for seqname,hit_info in self.data.items():
            #print(dir(hit_info.data[0]))
            if hit_info.data[0].pvalue is not None:
                if np.all(
                    np.asarray([this.pvalue for this in hit_info.data]) > pval_thresh
                ):
                    continue
            motif_name = hit_info.data[0].tfname
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

    def get_design_matrix(self, rec_db, pval_thresh=1.0):
        var_lut = {}
        # set up array of zeros with n_records rows and n_motifs columns
        motif_hits = self.gather_hits_dict(pval_thresh)
        X = np.zeros((len(rec_db), len(motif_hits)))
        for (i,(motif_name,rec_name_list)) in enumerate(motif_hits.items()):
            for rec_name,rec_idx in rec_db.record_name_lut.items():
                # if this record's name is in the fimo hits, set its index in X to 1
                if rec_name in rec_name_list:
                    X[rec_idx,i] = 1
            # matches var_lut format in shape motifs
            hit = 1
            var_lut[i] = {'motif_idx': i, 'hits': hit}
        return (X,var_lut)

    def __iter__(self):
        for name in self.names:
            yield self.data[name]

    def __len__(self):
        return len(self.data)


class StremeLine(MemeLine):

    def __init__(self, line=None):
        super().__init__()

        if line is not None:
            self.parse(line)

    def parse(self, line):
        linearr = line.rstrip().split('\t')
        self.patname = linearr[0]
        self.tfname = linearr[1]
        self.seqname = linearr[3]
        # converts to normal 0-based coordinates
        self.start = None
        self.stop = None
        self.strand = None
        self.qvalue = None
        self.matchedseq = None
        try:
            self.score = float(linearr[4])
        except ValueError:
            self.score = None
        try:
            self.pvalue = float(linearr[2])
        except ValueError:
            self.pvalue = None


class StremeSeq(object):

    def __init__(self,name=''):
        self.data = []
        self.name=name

    def __iter__(self):
        for line in self.data:
            yield line

    def append(self, stremeline):
        self.data.append(stremeline)

    def __len__(self):
        return len(self.data)

    def find_lines(self, findfunc, findall=True):
        matches = list(filter(findfunc, self.data))
        new_seq = StremeSeq(self.name)
        if len(matches) == 0:
            return new_seq
        if findall:
            for match in matches:
                new_seq.append(match)
        else:
            new_seq.append(match[0])
        return new_seq


class StremeFile(object):

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
                this_line = StremeLine(line)
                if this_line.seqname in self.data:
                    self.data[this_line.seqname].append(this_line)
                else:
                    this_entry = StremeSeq(this_line.seqname)
                    this_entry.append(this_line)
                    self.data[this_line.seqname] = this_entry
                    self.names.append(this_line.seqname)

    def pull_entry(self, name):
        return self.data[name]

    def gather_hits_dict(self, pval_thresh=1.0):
        hits = {}
        for seqname,hit_info in self.data.items():
            #print(dir(hit_info.data[0]))
            if hit_info.data[0].pvalue is not None:
                if np.all(np.asarray([this.pvalue for this in hit_info.data]) > pval_thresh):
                    continue
            motif_name = hit_info.data[0].tfname
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

    def get_design_matrix(self, rec_db, pval_thresh=1.0):
        var_lut = {}
        # set up array of zeros with n_records rows and n_motifs columns
        motif_hits = self.gather_hits_dict(pval_thresh)
        X = np.zeros((len(rec_db), len(motif_hits)))
        for (i,(motif_name,rec_name_list)) in enumerate(motif_hits.items()):
            for rec_name,rec_idx in rec_db.record_name_lut.items():
                # if this record's name is in the streme hits, set its index in X to 1
                if rec_name in rec_name_list:
                    X[rec_idx,i] = 1
            # matches var_lut format in shape motifs
            hit = 1
            var_lut[i] = {'motif_idx': i, 'hits': hit}
        return (X,var_lut)

    def __iter__(self):
        for name in self.names:
            yield self.data[name]

                   
