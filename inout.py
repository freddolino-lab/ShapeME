import numpy as np
import dnashapeparams as dsp
import logging

def complement(sequence):
    """Complement a nucleotide sequence
    >>> complement("AGTC")
    'TCAG'
    >>> complement("AGNT")
    'TCNA'
    >>> complement("AG-T")
    'TC-A'
    """
    # create a dictionary to act as a mapper
    comp_dict = {'A': 'T', 'G':'C', 'C':'G', 'T': 'A', 'N':'N', '-':'-'}
    # turn the sequence into a list
    sequence = list(sequence)
    # remap it to the compelmentary sequence using the mapping dict
    sequence = [comp_dict[base] for base in sequence]
    # join the new complemented sequence list into a string
    sequence = ''.join(sequence)
    return sequence

class FastaEntry(object):
    """ 
    Stores all the information for a single fasta entry. 

    An example of a fasta entry is below:

    >somechromosomename
    AGAGATACACACATATA...ATACAT #typically 50 bases per line

    Args:
        header (str): The complete string for the header ">somechromosomename" 
                      in the example above. Defaults to ">"
        seq (str): The complete string for the entire sequence of the entry

    Attributes:
        header (str): The complete string for the header ">somechromosomename" 
                      in the example above.
        seq (str): The complete string for the entire sequence of the entry

    """
    def __init__(self, header = ">", seq = ""):
        self.header = header
        self.seq = seq
        self.length = None

    def __str__(self):
        return "<FastaEntry>" + self.chrm_name() + ":" + str(len(self))

    def write(self,fhandle):
        fhandle.write(self.header+"\n")
        for i in range(0,len(self), 70):
            try:
                fhandle.write(self.seq[i:i+70]+"\n")
            except IndexError:
                fhandle.write(self.seq[i:-1] + "\n")

    def __iter__(self):
        for base in seq:
            yield base


    def set_header(self, header):
        self.header = header

    def set_seq(self, seq, rm_na=None):
        if rm_na:
            for key in rm_na.keys():
                seq = [rm_na[key] if x == key else float(x) for x in seq]
        self.seq = seq

    def __len__(self):
        if self.length:
            return self.length
        else:
            return(len(self.seq))

    def pull_seq(self, start, end, circ=False, rc=False):
        """ 
        Obtain a subsequence from the fasta entry sequence

        Args:
            start (int)    : A start value for the beginning of the slice. Start
                             coordinates should always be within the fasta entry
                             sequence length.
            end   (int)    : An end value for the end of the slice. If circ is
                             True then end coordinates can go beyond the fasta
                             entry length.
            circ  (boolean): Flag to allow the end value to be specified beyond
                             the length of the sequence. Allows one to pull
                             sequences in a circular manner.
        Returns:
            A subsequence of the fasta entry as specified by the start and end

        Raises:
            ValueError: If start < 0 or >= fasta entry sequence length.
            ValueError: If circ is False and end > fasta entry sequence length.
        """

        seq_len = len(self)
        if start < 0 or start >= seq_len:
            if circ and start < 0:
                start = seq_len + start
                end = seq_len + end
            else:
                raise ValueError("Start %s is outside the length of the sequence %s"%(start,seq_len))
        if end > seq_len:
            if circ:
                seq = self.seq[start:seq_len] + self.seq[0:(end-seq_len)]
            else: 
                raise ValueError("End %s is outside length of sequence %s"%(end,seq_len))
        else:
            seq = self.seq[start:end].upper()
        if rc:
            return complement(seq)[::-1]
        else:
            return seq

    def chrm_name(self):
        """
        Pulls the chromosome name from the header attribute.

        Assumes the header is of the type ">chromosomename" and nothing else
        is in the header.

        Returns:
            chromosome name
        """
        return self.header[1:]
                

class FastaFile(object):
    """ 
    Stores all the information for a single fasta file.

    An example of a fasta file is below

    >somechromosomename
    AGAGATACACACATATA...ATACAT 
    GGGAGAGAGATCTATAC...AGATAG
    >anotherchromosomename
    AGAGATACACACATATA...ATACAT #typically 50 bases per line

    Attributes:
        data (dict): where the keys are the chromosome names and the entries are
                     FastaEntry objects for each key
    """

    def __init__(self):
        self.data = {}
        self.names = []


    def __iter__(self):
        for name in self.names:
            yield self.pull_entry(name)

    def read_whole_file(self, fhandle):
        """ 
        Read an entire fasta file into memory and store it in the data attribute
        of FastaFile

        Args:
            fhandle (File)    : A python file handle set with mode set to read
        Returns:
            None

        Raises:
            ValueError: If fasta file does not start with a header ">"
        """

        line = fhandle.readline().strip()
        if line[0] != ">":
            raise ValueError("File is missing initial header!")
        else:
            curr_entry = FastaEntry(header = line.rstrip().split()[0])
        line = fhandle.readline().strip()
        curr_seq = []
        while line != '':
            if line[0] == ">":
                curr_entry.set_seq(''.join(curr_seq))
                self.data[curr_entry.chrm_name()] = curr_entry
                self.names.append(curr_entry.chrm_name())
                curr_seq = []
                curr_entry = FastaEntry(line)
            else:
                curr_seq.append(line)

            line = fhandle.readline().strip()

        curr_entry.set_seq(''.join(curr_seq))
        self.data[curr_entry.chrm_name()] = curr_entry
        self.names.append(curr_entry.chrm_name())

    def read_whole_datafile(self, fhandle, delim=","):
        """ 
        Read an entire fasta file into memory and store it in the data attribute
        of FastaFile. This handles comma seperated data in place of sequence

        Args:
            fhandle (File)    : A python file handle set with mode set to read
        Returns:
            None

        Raises:
            ValueError: If fasta file does not start with a header ">"
        """

        line = fhandle.readline().strip()
        if line[0] != ">":
            raise ValueError("File is missing initial header!")
        else:
            curr_entry = FastaEntry(header = line.rstrip().split()[0])
        line = fhandle.readline().strip()
        curr_seq = []
        while line != '':
            if line[0] == ">":
                curr_entry.set_seq(curr_seq, rm_na={"NA":np.nan})
                self.data[curr_entry.chrm_name()] = curr_entry
                self.names.append(curr_entry.chrm_name())
                curr_seq = []
                curr_entry = FastaEntry(line)
            else:
                line = line.split(delim)
                curr_seq.extend(line)

            line = fhandle.readline().strip()

        curr_entry.set_seq(curr_seq, rm_na={"NA":np.nan})
        self.data[curr_entry.chrm_name()] = curr_entry
        self.names.append(curr_entry.chrm_name())

    def pull_entry(self, chrm):
        """
        Pull a FastaEntry out of the FastaFile

        Args:
            chrm (str): Name of the chromosome that needs pulled
        Returns:
            FastaEntry object
        """
        try:
            return self.data[chrm]
        except KeyError:
            raise KeyError("Entry for %s does not exist in fasta file"%chrm)

    def add_entry(self, entry):
        """
        Add a FastaEntry to the object

        Args:
            entry (FastaEntry): FastaEntry to add
        Returns:
            None
        """
        this_name = entry.chrm_name()
        if this_name not in self.data.keys():
            self.names.append(this_name)
        self.data[this_name]= entry

    def chrm_names(self):
        return self.names

    def write(self, fhandle):
        """ 
        Write the contents of self.data into a fasta format

        Args:
            fhandle (File)    : A python file handle set with mode set to write
        Returns:
            None

        """
        for chrm in self.chrm_names():
            entry = self.pull_entry(chrm)
            entry.write(fhandle)


class SeqDatabase(object):
    """Class to store input information from tab seperated value with
    fasta name and a score

    Attributes:
        data (dict): a dictionary with fasta names as keys and scores as values
        names (list): A list of fasta names
        params (list): A list of dsp.ShapeParams for each sequence
    """

    def __init__(self, names=None):
        self.names = names
        self.values = []
        self.params = []
        self.vectors = []
        self.center_spread = None

    def __iter__(self):
        """ Iter method iterates over the parameters

        Acts as a generator

        Yields:
            dsp.ShapeParam object for each sequence in self.name order
        """
        for param in self.params:
            yield param

    def __len__(self):
        return len(self.names)

    def get_values(self):
        """ Get method for the values attribute

        Returns:
            self.values- this is a mutable list
        """
        return self.values

    def get_names(self):
        """ Get method for the names attribute

        Returns:
            self.names- this is a mutable list
        """
        return self.names

    def discretize_RZ(self):
        # first convert values to robust Z score
        self.values = np.array(self.values)
        median = np.median(self.values)
        mad = np.median(np.abs((self.values-median)))*1.4826
        self.values = (self.values-median)/mad
        bins = [-2*mad + median, -1*mad + median, 1*mad + median, 2*mad + median]
        logging.warning("Discretizing on bins: %s"%bins)
        self.values = np.digitize(self.values, bins)


    def discretize_quant(self,nbins=10):
        quants = np.arange(0,100, 100.0/nbins)
        values = np.array(self.values)
        bins = []
        for quant in quants:
            bins.append(np.percentile(values, quant))
        logging.warning("Discretizing on bins: %s"%bins)
        self.values = list(np.digitize(values, bins))

    def category_subset(self, category):
        """Subset the Sequence database based on category membership

        Args:
            category (int): category to select

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        locs = np.where(np.array(self.values) == category)[0]
        new_db = SeqDatabase(names=[self.names[x] for x in locs])
        new_db.params = [self.params[x] for x in locs]
        new_db.values = [self.values[x] for x in locs]
        if self.vectors:
            new_db.vectors= self.vectors
        return new_db

    def shuffle(self):
        """ Get a shuffled view of the data. NOT THREAD SAFE

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        size = len(self)
        subset = np.random.permutation(size)
        new_db = SeqDatabase(names=[self.names[x] for x in subset])
        new_db.params = [self.params[x] for x in subset]
        new_db.values = [self.values[x] for x in subset]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in subset]
        return new_db


    def random_subset(self, size):
        """ Take a random subset of the data. NOT THREAD SAFE

        Args:
            size (float): percentage of data to subset
            prng (np.random.prng): a numpy random number generator

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        total_num = int(np.floor(size*len(self)))
        subset = np.random.permutation(len(self))
        subset = subset[0:total_num]
        new_db = SeqDatabase(names=[self.names[x] for x in subset])
        new_db.params = [self.params[x] for x in subset]
        new_db.values = [self.values[x] for x in subset]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in subset]
        return new_db


    def read(self, infile, dtype=int):
        """Method to read a sequence file in FIRE/TEISER/iPAGE format

        Args:
            infile (str): input data file name
            dtype (func): a function to convert the values, defaults to int

        Modifies:
            names- adds in the name of each sequence
            values- adds in the value for each sequence
            params - makes a new dsp.ShapeParams object for each sequence
        """
        with open(infile) as inf:
            line = inf.readline()
            for line in inf:
                linearr = line.rstrip().split("\t")
                self.names.append(linearr[0])
                self.values.append(dtype(linearr[1]))
                self.params.append(dsp.ShapeParams(data={},names=[]))


    def normalize_params(self):
        """Method to normalize each parameter to a robustZ score based on
        all parameters in the database.

        This will ignore any Nans in any sequences parameter scores. First
        calculates center (median of all scores) and spread (MAD of all scores)
        and uses the dsp.ShapeParamSeq.normalize_values method to normalize
        each parameter

        Modifies:
            params - makes all dsp.ShapeParams.params into a robustZ score
        """
        # figure out center and spread
        for seqparam in self:
            all_this_param = {}
            for this_param in seqparam:
                this_val = all_this_param.get(this_param.name, [])
                this_val.extend(this_param.params)
                all_this_param[this_param.name] = this_val
        cent_spread = {}
        for name in all_this_param.keys():
            these_vals = all_this_param[name]
            these_vals = np.array(these_vals)
            these_vals = these_vals[np.isfinite(these_vals)]
            center = np.median(these_vals)
            spread = np.median(np.abs(these_vals - center))*1.4826
            cent_spread[name] = (center, spread)

        # normalize params
        for seqparam in self:
            for param in seqparam:
                center, spread = cent_spread[param.name]
                param.normalize_values(center, spread)
        self.center_spread = cent_spread

    def pre_compute_windows(self, wsize, slide_by = 1, wstart=0, wend= None):
        for seq in self:
            this_seqs = []
            for window in seq.sliding_windows(wsize, slide_by, wstart, wend):
                window.as_vector(cache=True)
                this_seqs.append(window)
            self.vectors.append(this_seqs)

    def iterate_through_precompute(self):
        for vals in self.vectors:
            yield vals

    def calculate_enrichment(self, discrete):
        """ Calculate the enrichment of a motif for each category

        Args:
        discrete (np.array) - a vector containing motif matches, 1 true 0 false

        Returns:
        dict holding enrichment for each category
        """

        enrichment = {}
        values = np.array(self.values)
        discrete = np.array(discrete)
        for value in np.unique(self.values):
            two_way = []
            two_way.append(np.sum(np.logical_and(values == value, discrete == 1 )))
            two_way.append(np.sum(np.logical_and(values != value, discrete == 1)))
            two_way.append(np.sum(np.logical_and(values == value, discrete == 0)))
            two_way.append(np.sum(np.logical_and(values != value, discrete == 0)))
            enrichment[value] = two_way
        return enrichment

    def mutual_information(self, discrete):
        """Method to calculate the MI between the values in the database and
        an external vector of discrete values of the same length
        
        Uses log2 so MI is in bits

        Args:
            discrete (np.array): a number array of integer values the same
                                 length as the database
        Returns:
            mutual information between discrete and self.values
        """
        these_vals = self.values
        total = len(self.values) + 0.0
        MI = 0
        for val in np.unique(these_vals):
            for val2 in np.unique(discrete):
                p_x = np.sum(these_vals == val)/total
                p_y = np.sum(discrete == val2)/total
                p_x_y = np.sum(np.logical_and(these_vals == val, discrete == val2))/total
                if p_x_y == 0 or p_x == 0 or p_y == 0:
                    MI+= 0
                else:
                    MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
        return MI

    def shannon_entropy(self, discrete):
        """Method to calculate the entropy between the values in the database and
        an external vector of discrete values of the same length
        
        Uses log2 so entropy is in bits

        Args:
            discrete (np.array): a number array of integer values the same
                                 length as the database
        Returns:
            entropy between discrete and self.values
        """
        these_vals = self.values
        total = len(self.values) + 0.0
        entropy = 0
        for val in np.unique(these_vals):
            for val2 in np.unique(discrete):
                p_x_y = np.sum(np.logical_and(these_vals == val, discrete == val2))/total
                if p_x_y == 0:
                    entropy+= 0
                else:
                    entropy += p_x_y*np.log2(p_x_y)
        return -entropy
