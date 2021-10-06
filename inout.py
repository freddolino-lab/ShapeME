import numpy as np
import dnashapeparams as dsp
import logging
from numba import jit,prange
import welfords
from scipy import stats
from collections import OrderedDict


def run_query_over_ref(y_vals, query_shapes, query_weights, threshold,
                       ref, R, W, dist_func, max_count=4, alpha=0.1):

    # R for record number, 2 for one forward count and one reverse count
    hits = np.zeros((R,2))

    optim_generate_peak_array(
        ref = ref,
        query = query_shapes,
        weights = query_weights,
        threshold = threshold,
        results = hits,
        R = R,
        W = W,
        dist = dist_func,
        max_count = max_count,
        alpha = alpha,
    )

    # sort the counts such that for each record, the
    #  smaller of the two numbers comes first.
    hits = np.sort(hits, axis=1)
    unique_hits = np.unique(hits, axis=0)

    this_mi = mutual_information(y_vals, hits, unique_hits)

    return this_mi,hits


@jit(nopython=True, parallel=True)
def optim_generate_peak_array(ref, query, weights, threshold,
                              results, R, W, dist, max_count, alpha):
    """Does same thing as generate_peak_vector, but hopefully faster
    
    Args:
    -----
    ref : np.array
        The windows attribute of an inout.RecordDatabase object. Will be an
        array of shape (R,L,S,W,2), where R is the number of records,
        L is the window size, S is the number of shape parameters, and
        W is the number of windows for each record.
    query : np.array
        A slice of the records and windows axes of the windows attribute of
        an inout.RecordDatabase object to check for matches in ref.
        Should be an array of shape (L,S,2), where 2 is for the 2 strands.
    weights : np.array
        Weights to be applied to the distance
        calculation. Should be an array of shape (L,S,1).
    threshold : float
        Minimum distance to consider a match.
    results : 2d np.array
        Array of shape (R,2), where R is the number of records in ref.
        This array should be populated with zeros, and will be incremented
        by 1 when matches are found. The final axis is of length 2 so that
        we can do the reverse-complement and the forward.
    R : int
        Number of records
    W : int
        Number of windows for each record
    dist : function
        The distance function to use for distance calculation.
    max_count : int
        The maximum number of hits to count for each strand.
    alpha : float
        Between 0.0 and 1.0, sets the lower limit for the tranformed weights
        prior to normalizing the sum of weights to one and calculating distance.
    """
    
    for r in prange(R):
        f_maxed = False
        r_maxed = False
        for w in range(W):
            
            if f_maxed and r_maxed:
                break

            ref_seq = ref[r,:,:,w,:]

            distances = dist(query, ref_seq, weights, alpha)

            if (not f_maxed) and (distances[0] < threshold):
                # if a window has a distance low enough,
                #   add 1 to this result's index
                results[r,0] += 1
                if results[r,0] == max_count:
                    f_maxed = True

            if (not r_maxed) and (distances[1] < threshold):
                results[r,1] += 1
                if results[r,1] == max_count:
                    r_maxed = True


@jit(nopython=True)
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))

@jit(nopython=True)
def manhattan_distance(vec1, vec2, w=1):
    return np.sum(np.abs(vec1 - vec2) * w)

@jit(nopython=True)
def constrained_manhattan_distance(vec1, vec2, w=1):
    w_exp = np.exp(w)
    w = w_exp/np.sum(w_exp)
    return np.sum(np.abs(vec1 - vec2) * w)

@jit(nopython=True)
def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

@jit(nopython=True)
def constrained_inv_logit_manhattan_distance(vec1, vec2, w=1, a=0.1):
    w_floor_inv_logit = a + (1-a) * inv_logit(w)
    w_trans = w_floor_inv_logit/np.sum(w_floor_inv_logit)
    w_abs_diff = np.abs(vec1 - vec2) * w_trans
    #NOTE: this seems crazy, but it's necessary instead of np.sum(arr, axis=(0,1))
    #  in order to get jit(nopython=True) to work
    first_sum = np.sum(w_abs_diff, axis=0)
    second_sum = np.sum(first_sum, axis=0)
    return second_sum

@jit(nopython=True)
def hamming_distance(vec1, vec2):
    return np.sum(vec1 != vec2)

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

    def write(self,fhandle, wrap = 70, delim = None):
        fhandle.write(self.header+"\n")
        if delim:
            convert = lambda x: delim.join([str(val) for val in x])
        else:
            convert = lambda x: x
        for i in range(0,len(self), wrap):
            try:
                fhandle.write(convert(self.seq[i:i+wrap])+"\n")
            except IndexError:
                fhandle.write(convert(self.seq[i:-1]) + "\n")

    def __iter__(self):
        for base in self.seq:
            yield base


    def set_header(self, header):
        self.header = header

    def set_seq(self, seq, rm_na=None):
        if rm_na:
            for key in list(rm_na.keys()):
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
        self.data = OrderedDict()
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
                curr_entry.set_seq(curr_seq, rm_na={"NA":np.nan, "nan":np.nan})
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
        if this_name not in list(self.data.keys()):
            self.names.append(this_name)
        self.data[this_name]= entry

    def chrm_names(self):
        return self.names

    def write(self, fhandle, wrap = 70, delim = None):
        """ 
        Write the contents of self.data into a fasta format

        Args:
            fhandle (File)    : A python file handle set with mode set to write
        Returns:
            None

        """
        for chrm in self.chrm_names():
            entry = self.pull_entry(chrm)
            entry.write(fhandle, wrap, delim)

def parse_shape_fasta(infile):
    """Specifically for reading shape parameter fasta files,
    which have comma-delimited positions in their sequences.
    """

    shape_info = {}
    first_line = True

    with open(infile, 'r') as f:

        for line in f:

            if line.startswith(">"):

                if not first_line:
                    store_record(shape_info, rec_seq, rec_name)

                rec_name = line.strip('>').strip()
                rec_seq = []
                first_line = False

            else:
                rec_seq.extend(line.strip().split(','))

    store_record(shape_info, rec_seq, rec_name)

    return(shape_info)
                
def store_record(info, rec_seq, rec_name):
    for i,val in enumerate(rec_seq):
        try:
            rec_seq[i] = float(val)
        except:
            rec_seq[i] = np.nan

    info[rec_name] = np.asarray(rec_seq)


class RecordDatabase(object):
    """Class to store input information from tab separated value with
    fasta name and a score

    Attributes:
    -----------
    shape_name_lut : dict
        A dictionary with shape names as keys and indices of the S
        axis of data as values.
    rec_name_lut : dict
        A dictionary with record names as keys and indices of the R
        axis of data as values.
    y : np.array
        A vector. Binary if looking at peak presence/absence,
        categorical if looking at signal magnitude.
    X : np.array
        Array of shape (R,P,S,2), where R is the number of records in
        the input data, P is the number of positions (the length of)
        each record, and S is the number of shape parameters present.
        The final axis is of length 2, one index for each strand.
    windows : np.array
        Array of shape (R,L,S,W,2), where R and S are described above
        for the data attribute, L is the length of each window,
        and W is the number of windows each record was chunked into.
        The final axis is of length 2, one index for each strand.
    weights : np.array
        Array of shape (R,L,S,W), where the axis lengths are described
        above for the windows attribute. Values are weights applied
        during calculation of distance between a pair of sequences.
    thresholds : np.array
        Array of shape (R,W), where R is the number of records in the
        database and W is the number of windows each record was
        chuncked into.
    """

    def __init__(self, infile=None, shape_dict=None, y=None, X=None,
                 shape_names=None, record_names=None, weights=None, windows=None,
                 shift_params=["HelT", "Roll"],
                 exclude_na=True):

        self.record_name_lut = {}
        self.shape_name_lut = {}
        if X is None:
            self.X = None
        else:
            if shape_names is None:
                raise("X values were given without names for the shapes. Reinitialize, setting the shape_names argument")
            self.X = X
            self.shape_name_lut = {name:i for i,name in enumerate(shape_names)}
        if y is not None:
            #if record_names is None:
            #    raise("y values were given without names for the records. Reinitialize, setting the record_names argument")
            self.y = y
            #self.record_name_lut = {name:i for i,name in enumerate(record_names)}
        if weights is not None:
            self.weights = weights
        if windows is not None:
            self.windows = windows
        if infile is not None:
            self.read_infile(infile)
        if shape_dict is not None:
            self.read_shapes(
                shape_dict,
                shift_params=shift_params,
                exclude_na=exclude_na,
            )

    def __len__(self):
        """Length method returns the total number of records in the database
        as determined by the length of the records attribute.
        """

        return len(self.y)

    def records_per_bin(self):
        """Prints the number of records in each category of self.y
        """

        distinct_cats = np.unique(self.y)
        print({cat:len(np.where(self.y == cat)[0]) for cat in distinct_cats})

    def iter_records(self):
        """Iter method iterates over the shape records

        Acts as a generator

        Yields:
        -------
        rec_shapes : np.array
            A 2d array of shape (P,S), where P is the length of each record
            and S is the number of shapes,
            for each index in the first axis of self.X.
            So, each record's shapes.
        """
        for rec_shapes in self.X:
            yield rec_shapes

    def iter_shapes(self):
        """Iter method iterates over each type of shape parameter's X vals

        Acts as a generator

        Yields:
        -------
        shapes : np.array
            A 2d array of shape (R,P), where R is the number of records in
            the database and P is the length of each record
        """
        shape_count = self.X.shape[2]
        for s in range(shape_count):
            yield self.X[:,:,s]

    def iter_y(self):
        """Iter method iterates over the ground truth records

        Acts as a generator

        Yields:
        -------
        val : int
        """
        for val in self.y:
            yield val

    def discretize_quant(self,nbins=10):
        """Discretize data into n equally populated bins according to the
        n quantiles
        
        Prints bin divisions to the logger

        Modifies:
        ---------
        self.y : np.array
            converts the values into their new categories
        """

        quants = np.arange(0, 100, 100.0/nbins)
        values = self.y
        bins = []
        for quant in quants:
            bins.append(np.percentile(values, quant))
        logging.warning("Discretizing on bins: {}".format(bins))
        self.y = np.digitize(values, bins)

    def read_infile(self, infile):#, keep_inds=None):
        """Method to read a sequence file in FIRE/TEISER/iPAGE format

        Args:
            infile (str): input data file name

        Modifies:
            record_name_lut - creates lookup table for associating
                record names in records with record indices in y and X.
            y - adds in the value for each sequence
        """
        scores = []
        with open(infile) as inf:
            line = inf.readline()
            for i,line in enumerate(inf):
                #if keep_inds is not None:
                #    if not i in keep_inds:
                #        continue
                linearr = line.rstrip().split("\t")
                self.record_name_lut[linearr[0]] = i
                scores.append(linearr[1])
            self.y = np.asarray(
                scores,
                dtype=int,
            )

    def read_shapes(self, shape_dict, shift_params=["HelT","Roll"], exclude_na=True):
        """Parses info in shapefiles and inserts into database

        Args:
        -----
        shape_dict: dict
            Dictionary, values of which are shape parameter names,
            keys of which are shape parameter fasta files.

        Modifies:
        ---------
        shape_name_lut
        X
        """

        self.normalized = False
        shape_idx = 0
        shape_count = len(shape_dict)

        for shape_name,shape_infname in shape_dict.items():

            if not shape_name in self.shape_name_lut:
                self.shape_name_lut[shape_name] = shape_idx
                s_idx = shape_idx
                shape_idx += 1

            this_shape_dict = parse_shape_fasta(shape_infname)

            if self.X is None:

                record_count = len(this_shape_dict)
                for rec_idx,rec_data in enumerate(this_shape_dict.values()):
                    if rec_idx > 0:
                        break
                    record_length = len(rec_data)

                self.X = np.zeros((record_count,record_length,shape_count,2))

            for rec_name,rec_data in this_shape_dict.items():
                r_idx = self.record_name_lut[rec_name]
                if shape_name in shift_params:
                    #while len(rec_data) < self.X.shape[1]:
                    rec_data = np.append(rec_data, np.nan)
                    rec_data = np.append(np.nan, rec_data)
                    self.X[r_idx,:,s_idx,0] = rec_data[1:]
                    self.X[r_idx,:,s_idx,1] = rec_data[:-1]
                else:
                    self.X[r_idx,:,s_idx,0] = rec_data
                    self.X[r_idx,:,s_idx,1] = rec_data

        if exclude_na:
            # identifies which positions have at least one NA for any shape
            complete_positions = ~np.any(np.isnan(self.X), axis=(0,2,3))
            # grabs complete cases from X
            self.X = self.X[:,complete_positions,:,:]

    #def set_center_spread(self, center_spread):
    #    """Method to set the center spread for the database for each
    #    parameter

    #    TODO check to make sure keys match all parameter names
    #    """
    #    self.center_spread = center_spread

    def determine_center_spread(self, method=robust_z_csp):
        """Method to get the center spread for each shape based on
        all records in the database.

        This will ignore any Nans in any shape sequence scores. First
        calculates center (median of all scores) and spread (MAD of all scores)

        Modifies:
            self.center_spread - populates center spread with calculated values
        """

        shape_cent_spreads = []
        # figure out center and spread
        for shape_recs in self.iter_shapes():
            shape_cent_spreads.append(method(shape_recs))

        self.shape_centers = np.array([x[0] for x in shape_cent_spreads])
        self.shape_spreads = np.array([x[1] for x in shape_cent_spreads])

    def normalize_shape_values(self):
        """Method to normalize each parameter based on self.center_spread

        Modifies:
            X - makes each index of the S axis a robust z score 
        """

        # normalize shape values
        if not self.normalized:
            self.X = (
                ( self.X
                - self.shape_centers[:,np.newaxis] )
                / self.shape_spreads[:,np.newaxis]
            )
            self.normalized = True
        else:
            print("X vals are already normalized. Doing nothing.")

    def unnormalize_shape_values(self):
        """Method to unnormalize each parameter from its RobustZ score back to
        its original value

        Modifies:
            X - unnormalizes all the shape values
        """
    
        # unnormalize shape values
        if self.normalized:
            self.X = ( self.X * self.shape_spreads ) + self.shape_centers
            self.normalized = False
        else:
            print("X vals are not normalized. Doing nothing.")

    def initialize_weights(self):
        """Provides the weights attribute, with a beta distrubuted
        weight over the length of the windows. Normalized such
        that the weights for all shapes/positions in a given
        record/window sum to one.
        """

        #L = self.windows.shape[1]
        #S = self.windows.shape[2]
        #self.weights = np.full_like(self.windows, 1.0/L/S)
        weights = np.zeros_like(self.windows[0,:,:,0,0])

        #x_vals = np.linspace(0,1,self.windows.shape[1])
        #w = stats.beta.pdf(x_vals, 2, 2)
        #w = w/np.sum(w)/self.windows.shape[2]
        #w_list = [w for _ in range(self.windows.shape[2])]
        #w = np.stack(w_list,axis=1)
        #
        #self.weights = np.zeros_like(self.windows)
        #for rec in range(self.weights.shape[0]):
        #    for win in range(self.weights.shape[-1]):
        #        self.weights[rec,:,:,win] = w
        return weights

    def compute_windows(self, wsize):
        """Method to precompute all nmers.

        Modifies:
            self.windows
        """

        #(R,L,S,W)
        rec_num, rec_len, shape_num, strand_num = self.X.shape
        window_count = rec_len - wsize
        
        self.windows = np.zeros((
            rec_num, wsize, shape_num, window_count, strand_num
        ))

        for i,rec in enumerate(self.iter_records()):
            for j in range(window_count):
                self.windows[i, :, :, j, :] = self.X[i, j:(j+wsize), :, :]

    def permute_records(self):

        rand_order = np.random.permutation(self.windows.shape[0])

        permuted_windows = self.windows[rand_order,...]
        permuted_shapes = self.X[rand_order,...]
        permuted_vals = self.y[rand_order,...]

        shape_count = self.X.shape[-1]
        rev_lut = {val:k for k,val in self.shape_name_lut.items()}

        permuted_records = RecordDatabase(
            y = permuted_vals,
            X = permuted_shapes,
            shape_names = [rev_lut[idx] for idx in range(shape_count)],
            windows = permuted_windows,
        )
        return(permuted_records)

    def set_initial_threshold(self, dist, weights, alpha=0.1,
                              threshold_sd_from_mean=2.0,
                              seeds_per_seq=1, max_seeds=10000):
        """Function to determine a reasonable starting threshold given a sample
        of the data

        Args:
        -----
        seeds_per_seq : int
        max_seeds : int

        Returns:
        --------
        threshold : float
            A  threshold that is the
            mean(distance)-2*stdev(distance))
        """

        online_mean = welfords.Welford()
        total_seeds = []
        seed_counter = 0
        shuffled_db = self.permute_records()

        for i,record_windows in enumerate(shuffled_db.windows):

            rand_order = np.random.permutation(record_windows.shape[2])
            curr_seeds_per_seq = 0
            for index in rand_order:
                window = record_windows[:,:,index]
                if curr_seeds_per_seq >= seeds_per_seq:
                    break
                total_seeds.append((window, weights))
                curr_seeds_per_seq += 1
                seed_counter += 1
            if seed_counter >= max_seeds:
                break

        logging.info(
            "Using {} random seeds to determine threshold from pairwise distances".format(
                len(total_seeds)
            )
        )
        distances = []
        for i,seed_i in enumerate(total_seeds):
            for j, seed_j in enumerate(total_seeds):
                if i >= j:
                    continue
                distances = dist(
                    seed_i[0],
                    seed_j[0],
                    seed_i[1],
                    alpha,
                )
                online_mean.update(distances[0]) # just use + strand for initial thresh

        mean = online_mean.final_mean()
        stdev = online_mean.final_stdev()

        logging.info("Threshold mean: {} and stdev {}".format(mean,stdev))
        thresh = max(mean - threshold_sd_from_mean * stdev, 0)
        logging.info("Setting initial threshold for each seed to {}".format(thresh))

        return thresh

    def mutual_information(self, arr):
        """Method to calculate the MI between the values in the database and
        an external vector of discrete values of the same length
        
        Uses log2 so MI is in bits

        Args:
        -----
            arr : np.array
                A 1D numpy array of integer values the same length as the database

        Returns:
        --------
            mutual information between discrete and self.values
        """
        return mutual_information(self.y, arr)

    def flatten_in_windows(self, attr):

        arr = getattr(self, attr)
        flat = arr.reshape(
            (
                arr.shape[0],
                arr.shape[1]*arr.shape[2],
                arr.shape[3]
            )
        ).copy()
        return flat

    def compute_mi(self, dist, max_count, alpha, weights, threshold):
        
        rec_num,win_len,shape_num,win_num,strand_num = self.windows.shape
        #mi_arr = np.zeros((rec_num,win_num))
        #hits_arr = np.zeros((rec_num,win_num,rec_num,strand_num))
        results = []

        for r in range(rec_num):
            for w in range(win_num):

                query = self.windows[r,:,:,w,0]
                query = query[...,None]

                mi,hits = run_query_over_ref(
                    y_vals = self.y,
                    query_shapes = query,
                    query_weights = weights,
                    threshold = threshold,
                    ref = self.windows,
                    R = rec_num,
                    W = win_num,
                    dist_func = dist,
                    max_count = max_count,
                    alpha = alpha,
                )
                results.append(
                    {
                        'seq': query,
                        'mi': mi,
                        'hits': hits,
                        'row_index': r,
                        'col_index': w
                    }
                )

        return results

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
        if names is None:
            self.names = []
        else:
            self.names = names
        self.values = []
        self.params = []
        self.vectors = []
        self.flat_windows = None
        self.center_spread = None

    def __iter__(self):
        """ Iter method iterates over the parameters

        Acts as a generator

        Yields:
            dsp.ShapeParam object for each sequence in self.name order
        """
        for param in self.params:
            yield param

    def __getitem__(self, item):
        return(self.names[item], self.values[item], self.params[item], self.vectors[item])

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
        mad = np.median(np.abs((values-median))) * 1.4826
        values = (values-median)/mad
        # I don't quite understand this method.
        # Why is the middle bin so wide?
        # And, why mad-standardize values, then digitize on these bins, which
        #   are shifted by the median?
        bins = [-2*mad + median, -1*mad + median, 1*mad + median, 2*mad + median]
        logging.warning("Discretizing on bins: {}".format(bins))
        self.values = np.digitize(values, bins)


    def discretize_quant(self,nbins=10):
        """ Discretize data into n equally populated bins according to the
        n quantiles
        
        Prints bins divisions to the logger

        Modifies:
            self.values (list): converts the values into their new categories
        """
        quants = np.arange(0, 100, 100.0/nbins)
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

    def random_subset_by_class(self, size, split=False):
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
        other_indices = []
        for val in np.unique(vals):
            this_subset = np.where(vals == val)[0]
            total_num = int(np.floor(len(this_subset)*size))
            selection = np.random.permutation(len(this_subset))
            for val in selection[0:total_num]:
                indices.append(this_subset[val])
            if split:
                for val in selection[total_num:len(this_subset)]:
                    other_indices.append(this_subset[val])
        new_db = SeqDatabase(names=[self.names[x] for x in indices])
        new_db.params = [self.params[x] for x in indices]
        new_db.values = self.get_values()[indices]
        if self.vectors:
            new_db.vectors= [self.vectors[x] for x in indices]
        if split: 
            new_db2 = SeqDatabase(names=[self.names[x] for x in other_indices])
            new_db2.params = [self.params[x] for x in other_indices]
            new_db2.values = self.get_values()[other_indices]
            if self.vectors:
                new_db2.vectors= [self.vectors[x] for x in other_indices]
            return new_db, new_db2
        else:
            return new_db

    def read(self, infile, dtype=int, keep_inds=None):
        """Method to read a sequence file in FIRE/TEISER/iPAGE format

        Args:
            infile (str): input data file name
            dtype (func): a function to convert the values, defaults to int
            keep_inds (list): a list of indices to store

        Modifies:
            names - adds in the name of each sequence
            values - adds in the value for each sequence
            params - makes a new dsp.ShapeParams object for each sequence
        """
        with open(infile) as inf:
            line = inf.readline()
            for i,line in enumerate(inf):
                if keep_inds is not None:
                    if not i in keep_inds:
                        continue
                linearr = line.rstrip().split("\t")
                self.names.append(linearr[0])
                self.values.append(dtype(linearr[1]))
                self.params.append(dsp.ShapeParams(data={},names=[]))
            self.values = np.asarray(self.values)


    def write(self, outfile):
        """ Method to write out the category file in FIRE/TEISER/IPAGE format"""
        with open(outfile, mode="w")  as outf:
            outf.write("name\tval\n")
            for name, val in zip(self.names, self.values):
                outf.write("%s\t%s\n"%(name, val))


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
        for name in list(all_this_param.keys()):
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

    def shape_vectors_to_3d_array(self):
        """Method to iterate through precomputed windows for each parameter
        and flatten all parameters into a single long vector.

        Modifies:
            self.flat_windows : np.array
                A 3d numpy array of shape (N, L*P, W), where N is the number
                of records in the original input data,
                L*P is the length of each window
                times the number of shape parameters, and W is the number
                of windows computed for each record.
        """

        param_names = self.params[0].names
        param_count = len(param_names)
        window_count = len(self.vectors[0])
        record_count = len(self)
        window_size = len(self.vectors[0][0].data['EP'].get_values())

        self.flat_windows = np.zeros(
            (
                record_count,
                window_size*param_count,
                window_count
            )
        )
        for i,val in enumerate(self.vectors):
            for j,window in enumerate(val):
                self.flat_windows[i,:,j] = window.as_vector()

#    def shape_vectors_to_2d_array(self):
#        """Method to iterate through precomputed windows for each parameter
#        and flatten all parameters into a single long vector.
#
#        Modifies:
#            self.flat_windows : np.array
#                A 3d numpy array of shape (N*W, L*P), where N*W is the number
#                of records in the original input data times the number of windows
#                computed for each record and L*P*W is the length of each window
#                times the number of shape parameters times.
#        """
#
#        param_names = self.params[0].names
#        param_count = len(param_names)
#        window_count = len(self.vectors[0])
#        record_count = len(self)
#        window_size = len(self.vectors[0][0].data['EP'].get_values())
#
#        self.flat_2d_windows = np.zeros(
#            (
#                record_count*window_count,
#                window_size*param_count
#            )
#        )
#        this_idx = 0
#        for i,val in enumerate(self.vectors):
#            for j,window in enumerate(val):
#                self.flat_2d_windows[this_idx,:] = window.as_vector()
#                this_idx += 1

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
        if "mi" in motif:
            string +="\tmi:%f"%(motif['mi'])
        if "motif_entropy" in motif:
            string +="\tmotif_entropy:%f"%(motif['motif_entropy'])
        if "category_entropy" in motif:
            string +="\tcategory_entropy:%f"%(motif['category_entropy'])
        if "zscore" in motif:
            string +="\tZ-score:%f"%(motif['zscore'])
        if "robustness" in motif:
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
                if not "name" in motif:
                    motif["name"] = "motif_%i"%(i)
                f.write(self.create_transform_line(motif, cats))
                f.write(self.create_motif_line(motif))
                f.write(self.create_data_lines(motif))
                f.write("\n")
    def to_tidy(self, outfile):
        """ Method to write file in a tidy format for data analysis

        Args
            outfile(str) - name of outputfile
        """
        with open(outfile, mode = "w") as f:
            header = ",".join(self.motifs[0]['seed'].names)
            header += ",bp,name\n"
            f.write(header)
            for i, motif in enumerate(self.motifs):
                if not "name" in motif:
                    motif["name"] = "motif_%i"%(i)
            for i, col in enumerate(motif['seed'].matrix().transpose()):
                string = ""
                string += ",".join(["%f"%val for val in col])
                string += ",%d,%s\n"%(i,motif["name"])
                f.write(string)

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

@jit(nopython=True)
def mutual_information(arrayx, arrayy, uniquey):
    """Method to calculate the mutual information between two discrete
    numpy arrays I(X;Y)
        
    Uses log2 so mutual information is in bits

    Args:
        arrayx (np.array): an array of discrete categories
        arrayy (np.array): an array of discrete categories
        uniquey (np.array): array of distinct values found in arrayy
    Returns:
        mutual information between array1 and array2
    """

    total_x = arrayx.size 
    total_y = arrayy.size
    total = total_x
    MI = 0
    for x in np.unique(arrayx):
        # p(x_i)
        row_is_x = arrayx == x
        p_x = np.sum(row_is_x)/total
        for y in uniquey:
            # p(y_j)
            row_is_y = (arrayy == y)[:,0]
            p_y = np.sum(row_is_y)/total
            # p(x_i,y_j)
            p_x_y = np.sum(np.logical_and(row_is_x, row_is_y))/total
            if p_x_y == 0 or p_x == 0 or p_y == 0:
                MI += 0
            else:
                MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
    return MI

@jit(nopython=True)
def conditional_mutual_information(arrayx, uniquex, arrayy, uniquey, arrayz, uniquez):
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
    total = total_x
    #if total_x != total_y or total_y != total_z:
    #    raise ValueError(
    #        "Array sizes must be the same {} {} {}".format(
    #            total_x,
    #            total_y,
    #            total_z
    #        )
    #    )
    #else:
    #    total = total_x
    CMI = 0
    # CMI will look at each position of arr_x and arr_y that are of value z in arr_z
    for z in uniquez:
        # set the indices we will look at in arr_x and arr_y
        subset = (arrayz == z)[:,0]
        # set number of vals == z in arr_z as denominator
        total_subset = np.sum(subset)
        p_z = total_subset/total
        this_MI = 0

        for x in uniquex:
            for y in uniquey:
                # calculate the probability that the indices of arr_x and arr_y
                #  corresponding to those in arr_z == z are equal to x or y.
                # so essentially, in english, we're saying the following:
                #  given that arr_z is what it is, what is the MI between
                #  arr_x and arr_y?
                row_is_y = (arrayy[subset] == y)[:,0]
                row_is_x = arrayx[subset] == x
                p_x = np.sum(row_is_x)/total_subset
                p_y = np.sum(row_is_y)/total_subset
                p_x_y = np.sum(
                    np.logical_and(
                        arrayx[subset] == x,
                        row_is_y
                    )
                ) / total_subset
                if p_x_y == 0 or p_x == 0 or p_y == 0:
                    this_MI += 0
                else:
                    this_MI += p_x_y*np.log2(p_x_y/(p_x*p_y))

        CMI += p_z*this_MI

    return CMI

