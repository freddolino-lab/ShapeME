import numpy as np
import dnashapeparams as dsp
import logging

def robust_z_csp(array):
    """Method to get the center and spread of an array based on the robustZ

    This will ignore any Nans or infinites in the array.

    Args:
        array (np.array)- 1 dimensional numpy array
    Returns:
        tuple - center (median) spread (MAD)

    """
    these_vals = array[np.isfinite(array)]
    center = np.median(these_vals)
    spread = np.median(np.abs(these_vals - center))*1.4826
    return (center, spread)

def identity_csp(array):
    """Method to get the center and spread of an array that keeps the array
    the same when used to normalize

    Args:
        array (np.array)- 1 dimensional numpy array
    Returns:
        tuple - center (0) spread (1)

    """
    return (0, 1)


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
        for base in self.seq:
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
        vectors (list): A list of all motifs precomputed
        center_spread (dict): The center and spread used to normalize shape data
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
        """Length method returns the total number of sequences in the database
        as determined by the length of the names attribute
        """
        return len(self.names)

    def get_values(self):
        """ Get method for the values attribute

        Returns:
            self.values- after reading this is a numpyarray
        """
        return self.values

    def get_names(self):
        """ Get method for the names attribute

        Returns:
            self.names- this is a mutable list
        """
        return self.names

    def discretize_RZ(self):
        """ Discretize data using 5 bins according to the robust Z score
        
        Prints bins divisions to the logger

        Modifies:
            self.values (list): converts the values into their new categories
        """
        # first convert values to robust Z score
        values = get_values(self)
        median = np.median(values)
        mad = np.median(np.abs((values-median)))*1.4826
        values = (values-median)/mad
        bins = [-2*mad + median, -1*mad + median, 1*mad + median, 2*mad + median]
        logging.warning("Discretizing on bins: %s"%bins)
        self.values = np.digitize(values, bins)


    def discretize_quant(self,nbins=10):
        """ Discretize data into n equally populated bins according to the
        n quantiles
        
        Prints bins divisions to the logger

        Modifies:
            self.values (list): converts the values into their new categories
        """
        quants = np.arange(0,100, 100.0/nbins)
        values = self.get_values()
        bins = []
        for quant in quants:
            bins.append(np.percentile(values, quant))
        logging.warning("Discretizing on bins: %s"%bins)
        self.values = np.digitize(values, bins)

    def category_subset(self, category):
        """Subset the Sequence database based on category membership

        Args:
            category (int): category to select

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        values = self.get_values()
        locs = np.where(values == category)[0]
        new_db = SeqDatabase(names=[self.names[x] for x in locs])
        new_db.params = [self.params[x] for x in locs]
        new_db.values = [self.values[x] for x in locs]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in locs]
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
        new_db.values = self.get_values()[subset]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in subset]
        return new_db


    def random_subset(self, size):
        """ Take a random subset of the data. NOT THREAD SAFE

        Args:
            size (float): percentage of data to subset

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        total_num = int(np.floor(size*len(self)))
        subset = np.random.permutation(len(self))
        subset = subset[0:total_num]
        new_db = SeqDatabase(names=[self.names[x] for x in subset])
        new_db.params = [self.params[x] for x in subset]
        new_db.values = self.get_values()[subset]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in subset]
        return new_db

    def random_subset_by_class(self, size):
        """ Take a random subset with proportional class representation. 
        NOT THREAD SAFE

        Args:
            size (float): percentage of data to subset

        Returns:
            new SeqDatabase object, all attributes are shared with original
            object, so this acts more like a numpy view
        """
        vals = self.get_values()
        indices = []
        for val in np.unique(vals):
            this_subset = np.where(vals == val)[0]
            total_num = int(np.floor(len(this_subset)*size))
            selection = np.random.permutation(len(this_subset))
            for val in selection[0:total_num]:
                indices.append(this_subset[val])
        new_db = SeqDatabase(names=[self.names[x] for x in indices])
        new_db.params = [self.params[x] for x in indices]
        new_db.values = self.get_values()[indices]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in indices]
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
            self.values = np.array(self.values)

    def set_center_spread(self, center_spread):
        """Method to set the center spread for the database for each
        parameter

        TODO check to make sure keys match all parameter names
        """
        self.center_spread = center_spread


    def determine_center_spread(self, method=robust_z_csp):
        """Method to get the center spread for each parameter based on
        all parameters in the database.

        This will ignore any Nans in any sequences parameter scores. First
        calculates center (median of all scores) and spread (MAD of all scores)

        Modifies:
            self.center_spread - populates center spread with calculated values
        """
        all_this_param = {}
        # figure out center and spread
        for seqparam in self:
            for this_param in seqparam:
                this_val = all_this_param.get(this_param.name, [])
                this_val.extend(this_param.params)
                all_this_param[this_param.name] = this_val
        cent_spread = {}
        for name in all_this_param.keys():
            these_vals = all_this_param[name]
            these_vals = np.array(these_vals) 
            cent_spread[name] = method(these_vals)
        self.set_center_spread(cent_spread)

    def normalize_params(self):
        """Method to normalize each parameter based on self.center_spread

        Modifies:
            params - makes all dsp.ShapeParams.params into a robustZ score
        """

        # normalize params
        for seqparam in self:
            for param in seqparam:
                center, spread = self.center_spread[param.name]
                param.normalize_values(center, spread)


    def unnormalize_params(self):
        """Method to unnormalize each parameter from its RobustZ score back to
        its original value

        Modifies:
            params - unnormalizes all the param values
        """
        # unnormalize params
        for seqparam in self:
            for param in seqparam:
                center, spread = self.center_spread[param.name]
                param.unnormalize_values(center, spread)

    def pre_compute_windows(self, wsize, slide_by = 1, wstart=0, wend= None):
        """Method to precompute all nmers. Uses the same syntax as 
        the sliding windows method.


        Modifies:
            self.vectors - creates a list of lists where there is a single
                           entry for each list holding all motifs coming
                           from that sequence (motif = dsp.ShapeParams)
        """
        for seq in self:
            this_seqs = []
            for window in seq.sliding_windows(wsize, slide_by, wstart, wend):
                window.as_vector(cache=True)
                this_seqs.append(window)
            self.vectors.append(this_seqs)

    def iterate_through_precompute(self):
        """Method to iterate through all precomputed motifs


        Yields: 
            vals (list): a list of motifs for that sequence
        """
        for vals in self.vectors:
            yield vals

    def calculate_enrichment(self, discrete):
        """ Calculate the enrichment of a motif for each category

        Args:
        discrete (np.array) - a vector containing motif matches, 1 true 0 false

        Returns:
            dict holding enrichment for each category as a two way table
        """

        enrichment = {}
        values = self.get_values()
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
        return mutual_information(self.get_values(), discrete)

    def joint_entropy(self, discrete):
        """Method to calculate the joint entropy between the values in the
        database and an external vector of discrete values of the same length
        
        Uses log2 so entropy is in bits

        Args:
            discrete (np.array): a number array of integer values the same
                                 length as the database
        Returns:
            entropy between discrete and self.values
        """
        return joint_entropy(self.get_values(), discrete)

    def shannon_entropy(self):
        return entropy(self.get_values())

class ShapeMotifFile(object):
    """ Class to store a dna shape motif file .dsm for writing and reading
        Currently no error checking on input file format

        Attributes:
            motifs (list) - list of motif dicts
            cent_spreads(list) - list of center spread dict for each motif
    """
    def __init__(self):
        self.motifs = []
        self.cent_spreads = []

    def __iter__(self):
        for motif in self.motifs:
            yield motif

    def __len__(self):
        return len(self.motifs)

    def unnormalize(self):
        for i, motif in enumerate(self):
            motif['seed'].unnormalize_values(self.cent_spreads[i])

    def normalize(self):
        for i, motif in enumerate(self):
            motif['seed'].normalize_values(self.cent_spreads[i])


    def add_motifs(self, motifs):
        """ Method to add additional motifs

        Currently extends to current list
        """
        self.motifs.extend(motifs)

    def create_transform_line(self, motif, cats):
        """ Method to create a transform line from a motif and category

        Args
            motif(dict) - motif dict
            cats(SeqDatabase) - final sequence database
        Returns
            string of line to be written
        """
        string = "Transform"
        for name in motif['seed'].names:
            string += "\t%s:(%f,%f)"%(name, cats.center_spread[name][0], cats.center_spread[name][1])
        string +="\n"
        return string

    def read_transform_line(self,linearr):
        """ Method to read a transform line from an input file

        Args
            linearr(list) - line split by tabs without Transform in the beg
        Returns
            cent_spread (dict) - parsed from line with names as keys
            names (list) - list of names in file order
        """
        cent_spread = {}
        names = []
        for val in linearr:
            name, vals = val.split(":")
            center,spread = vals.split(",")
            center = float(center.replace("(", ""))
            spread = float(spread.replace(")", ""))
            cent_spread[name] = (center, spread)
            names.append(name)
        return cent_spread, names

    def read_motif_line(self,linearr):
        """ Method to read a motif line from an input file

        Args
            linearr(list) - line split by tabs without Motif in the beg
        Returns
            motif_dict (dict) - dictionary holding all the motif fields
        """
        motif_dict = {}
        for val in linearr:
            key, value = val.split(":")
            try:
                motif_dict[key] = float(value)
            except ValueError:
                motif_dict[key] = value
        return motif_dict

    def read_data_lines(self,lines, names):
        """ Method to read a set of data lines from an input file

        Args
            lines(list) - list of unsplit lines
            names(list) - list of names in proper order as data
        Returns
            motif (dsp.ShapeParams) - holds the particular motif
        """
        data_dict = {}
        for name in names:
            data_dict[name] = []
        for line in lines:
            linearr = line.split(",")
            for i,val in enumerate(linearr):
                data_dict[names[i]].append(float(val))
        motif = dsp.ShapeParams()
        for name in names:
            motif.add_shape_param(dsp.ShapeParamSeq(name=name, 
                                  params=data_dict[name]))
        return motif

    def read_file(self, infile):
        """ Method to read the full file

        Args
            infile (str) - name of input file
            names(list) - list of names in proper order as data
        Modifies
            self.motifs adds a motif per motif in file
            self.center_spread adds center and spread per motif in file
        """
        in_motif=False
        cent_spreads = []
        with open(infile) as f:
            for line in f:
                if not in_motif:
                    while line.rstrip() == "":
                        continue
                    if line.rstrip().startswith("Transform"):
                        in_motif = True
                        lines = []
                        cent_spread, names = self.read_transform_line(line.rstrip().split("\t")[1:])
                        continue
                else:
                    if line.startswith("Motif"):
                        this_motif = self.read_motif_line(line.rstrip().split("\t")[1:])
                    elif line.rstrip() == "":
                        this_motif["seed"] = self.read_data_lines(lines, names)
                        self.motifs.append(this_motif)
                        self.cent_spreads.append(cent_spread)
                        in_motif = False
                    else:
                        lines.append(line)

    def create_motif_line(self,motif):
        """ Method to create a motif line from a motif dict

        Args
            motif(dict) - motif dict
        Returns
            string of line to be written
        """
        string = "Motif"
        string += "\tname:%s"%(motif["name"])
        string += "\tthreshold:%f"%(motif["threshold"])
        string += "\tlength:%i"%(len(motif["seed"]))
        if motif.has_key("mi"):
            string +="\tmi:%f"%(motif['mi'])
        if motif.has_key("motif_entropy"):
            string +="\tmotif_entropy:%f"%(motif['motif_entropy'])
        if motif.has_key("category_entropy"):
            string +="\tcategory_entropy:%f"%(motif['category_entropy'])
        if motif.has_key("zscore"):
            string +="\tZ-score:%f"%(motif['zscore'])
        if motif.has_key("robustness"):
            string +="\trobustness:%s"%(motif['robustness'])

        string += "\n"
        return string

    def create_data_lines(self,motif):
        """ Method to create data lines from a motif 

        Args
            motif(dict) - motif dict
        Returns
            string of lines to be written
        """
        string = ""
        for col in motif['seed'].matrix().transpose():
            string+= ",".join(["%f"%val for val in col])
            string += "\n"
        return string

    def write_file(self, outfile, cats):
        """ Method to write a file form object

        Args
            outfile(str) - name of outputfile
            cats (SeqDatabase) - whole database used
        """
        with open(outfile, mode="w") as f:
            for i, motif in enumerate(self.motifs):
                if not motif.has_key("name"):
                    motif["name"] = "motif_%i"%(i)
                f.write(self.create_transform_line(motif, cats))
                f.write(self.create_motif_line(motif))
                f.write(self.create_data_lines(motif))
                f.write("\n")

def entropy(array):
    """Method to calculate the entropy of any discrete numpy array
        
    Uses log2 so entropy is in bits

    Args:
        array (np.array): an array of discrete categories
    Returns:
        entropy of array
    """
    entropy = 0
    total = array.size 
    for val in np.unique(array):
        num_this_class = np.sum(array == val)
        p_i = num_this_class/float(total)
        if p_i == 0:
            entropy += 0
        else:
            entropy += p_i*np.log2(p_i)
    return -entropy


def joint_entropy(arrayx, arrayy):
    """Method to calculate the joint entropy H(X,Y) for two discrete numpy
    arrays
    
    Uses log2 so entropy is in bits. Arrays must be same length

    Args:
        arrayx (np.array): an array of discrete values
        arrayy (np.array): an array of discrete values
    Returns:
        entropy between discrete and self.values
    """
    total_1 = arrayx.size 
    total_2 = arrayy.size
    if total_1 != total_2:
        raise ValueError("Array sizes must be the same %s %s"%(total_1, total_2))
    else:
        total = total_1 + 0.0

    entropy = 0
    for x in np.unique(arrayx):
        for y in np.unique(arrayy):
            p_x_y = np.sum(np.logical_and(arrayx == x, arrayy == y))/total
            if p_x_y == 0:
                entropy+= 0
            else:
                entropy += p_x_y*np.log2(p_x_y)
    return -entropy

def conditional_entropy(arrayx, arrayy):
    """Method to calculate the conditional entropy H(X|Y) of the two arrays
    
    Uses log2 so entropy is in bits. Arrays must be same length

    Args:
        arrayx (np.array): an array of discrete values
        arrayy (np.array): an array of discrete values
    Returns:
        entropy between discrete and self.values
    """
    total_1 = arrayx.size 
    total_2 = arrayy.size
    if total_1 != total_2:
        raise ValueError("Array sizes must be the same %s %s"%(total_1, total_2))
    else:
        total = total_1 + 0.0

    entropy = 0
    for x in np.unique(arrayx):
        for y in np.unique(arrayy):
            p_x_y = np.sum(np.logical_and(arrayx == x, arrayy == y))/total
            p_x = np.sum(arrayx == x)/total
            if p_x_y == 0 or p_x == 0:
                entropy+= 0
            else:
                entropy += p_x_y*np.log2(p_x/p_x_y)
    return -entropy


def mutual_information(arrayx, arrayy):
    """Method to calculate the mutual information between two discrete
    numpy arrays I(X;Y)
        
    Uses log2 so mutual information is in bits

    Args:
        arrayx (np.array): an array of discrete categories
        arrayy (np.array): an array of discrete categories
    Returns:
        mutual information between array1 and array2
    """

    total_x = arrayx.size 
    total_y = arrayy.size
    if total_x != total_y:
        raise ValueError("Array sizes must be the same %s %s"%(total_x, total_y))
    else:
        total = total_x + 0.0
    MI = 0
    for x in np.unique(arrayx):
        for y in np.unique(arrayy):
            p_x = np.sum(arrayx == x)/total
            p_y = np.sum(arrayy == y)/total
            p_x_y = np.sum(np.logical_and(arrayx == x, arrayy == y))/total
            if p_x_y == 0 or p_x == 0 or p_y == 0:
                MI+= 0
            else:
                MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
    return MI


def conditional_mutual_information(arrayx, arrayy, arrayz):
    """Method to calculate the conditional mutual information I(X;Y|Z)
        
    Uses log2 so mutual information is in bits. This is O(X*Y*Z) where
    X Y and Z are the number of unique categories in each array

    Args:
        arrayx (np.array): an array of discrete categories
        arrayy (np.array): an array of discrete categories
        arrayz (np.array): an array of discrete categories
    Returns:
        conditional mutual information arrayx;arrayy | arrayz
    """

    total_x = arrayx.size 
    total_y = arrayy.size
    total_z = arrayz.size
    if total_x != total_y or total_y != total_z:
        raise ValueError("Array sizes must be the same %s %s %s"%(total_x, total_y, total_z))
    else:
        total = total_x + 0.0
    CMI = 0
    for z in np.unique(arrayz):
        subset = arrayz == z
        total_subset = np.sum(subset) + 0.0
        p_z = total_subset/total
        this_MI = 0
        for x in np.unique(arrayx):
            for y in np.unique(arrayy):
                p_x = np.sum(arrayx[subset] == x)/total_subset
                p_y = np.sum(arrayy[subset] == y)/total_subset
                p_x_y = np.sum(np.logical_and(arrayx[subset] == x, arrayy[subset] == y))/total_subset
                if p_x_y == 0 or p_x == 0 or p_y == 0:
                    this_MI+= 0
                else:
                    this_MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
        CMI += p_z*this_MI
    return CMI

