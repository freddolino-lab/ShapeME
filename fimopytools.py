"""Set of classes designed to deal with Fimo gff outputs. Only handles the case
where ONE motif was searched
"""

class FimoLine(object):

    def __init__(self, line=None):
        if line is not None:
            self.parse(line)
        else:
            self.patname=''
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
        self.seqname = linearr[1]
        # converts to normal 0-based coordinates
        self.start = int(linearr[2])-1
        self.stop = int(linearr[3])
        self.strand = linearr[4]
        try:
            self.score = float(linearr[5])
        except ValueError:
            self.score = None
        try:
            self.pvalue = float(linearr[6])
        except ValueError:
            self.pvalue = None
        try:
            self.qvalue = float(linearr[7])
        except ValueError:
            self.qvalue=None
        self.matchedseq=linearr[8]

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
        matches = filter(findfunc, self.data)
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
            inf.readline()
            for line in inf:
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

    def __iter__(self):
        for name in self.names:
            yield self.data[name]
