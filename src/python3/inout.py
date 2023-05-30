import numpy as np
import cvlogistic
import dnashapeparams as dsp
#import shapemotifvis as smv
import fimopytools as fimo
import logging
#from numba import jit,prange
import welfords
from scipy import stats
from scipy import sparse
from collections import OrderedDict
import pickle
import json
import sys
import glob
import re
import copy
import time
import subprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import cluster 
from sklearn import metrics
from math import log
from scipy.stats import contingency
from statsmodels.stats import rates
import tempfile
import os

from matplotlib import pyplot as plt

EPSILON = np.finfo(float).eps

class ReadMotifException(Exception):
    def __init__(self):
        self.message = f"ERROR: passed a filename to Motifs() but did not set "\
            f"the motif type. Set motif_type to either \"shape\" or \"sequence\"."
        super().__init__(self.message)

class SetNamesException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NoSeqFaException(Exception):
    def __init__(self):
        self.message = f"ERROR: you requested finding of sequence motifs by "\
            f"setting the --find_seq_motifs command line option, but you did not "\
            f"provide the --seq_fasta option."
        super().__init__(self.message)

class RustBinaryException(Exception):
    def __init__(self, cmd):
        self.message = f"ERROR: infer_motifs binary execution exited with "\
            f"non-zero exit status.\n"\
            f"The attempted command was as follows:\n{cmd}"
        super().__init__(self.message)

class StremeClassException(Exception):
    def __init__(self, val, line):
        self.value = val
        self.line = line
        self.message = f"ERROR: using streme to find sequence motifs "\
            f"only implemented for binary input of value 0 and 1. "\
            f"Value at {self.line} has value {self.value}."
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

class SeqMotifOptionException(Exception):
    """Exception raised for error in sequence motif CLI usage

    Attributes:
        meme_fname - name of meme file passed at command line
        message - explanation of the error
    """

    def __init__(self, fname):

        self.meme_fname = fname
        self.message = f"ERROR: you specified a known motifs file, "\
                f"{self.meme_fname}, AND to search for new motifs by including "\
                f"--find_seq_motifs at the command line. You must choose one or the "\
                f"other if you want to work with sequence motifs. If you want "\
                f"only to deal with shape motifs, include neither command line option."
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'


def evaluate_match_object(mo):
    if mo is not None:
        result = float(mo.group())
    else:
        result = None
    return result


def construct_records(
        in_direc,
        shape_names,
        shape_files,
        in_fname,
        exclude_na=True,
        shift_params = ["Roll", "HelT"],
):
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(shape_names, shape_files)
    }
    records = RecordDatabase(
        os.path.join(in_direc,in_fname),
        shape_fname_dict,
        shift_params = shift_params,
        exclude_na = exclude_na,
    )
    return records


def read_shape_motifs(fname, shape_lut, alt_name_base=None):
    """Reads json file (fname) containing Motifs from rust, wrangles
    data into appropriate shapes for next steps of infer_motifs.py
    """

    with open(fname, 'r') as f:
        rust_mi_results = json.load(f)

    motif_results = []
    for i,motif in enumerate(rust_mi_results):
        motif_id = f"SHAPE-{i+1}"
        if alt_name_base is not None:
            alt_name = f"{alt_name_base}-{i+1}"
        else: alt_name = "None"
        this_motif = wrangle_rust_motif(motif, shape_lut, motif_id, alt_name)
        motif_results.append(this_motif)

    return motif_results


def read_fimo_file(fname):
    """Reads json file (fname) containing Motifs from rust, wrangles
    data into appropriate shapes for next steps of infer_motifs.py
    """

    motif_results = []
    for motif in seq_motifs:
        motif_results.append(wrangle_seq_motif(motif))

    return motif_results


def parse_meme_file(fname, evalue_thresh=np.Inf):

    motif_list = []

    alphabet_len_pat = re.compile(r'(?<=alength\= )\d+')
    motif_width_pat = re.compile(r'(?<=w\= )\d+')
    eval_pat = re.compile(r'(?<=[ES]\= )\S+')
    nsites_pat = re.compile(r'(?<=nsites\= )\d+')
    threshold_pat = re.compile(r'(?<=threshold\= )\S+')
    ami_pat = re.compile(r'(?<=adj_mi\= )\S+\.\d+')
    robustness_pat = re.compile(r'(?<=robustness\= )(\d+)\/(\d+)')
    zscore_pat = re.compile(r'(?<=zscore\= )\S+\.\d+')

    # start not in_motif
    in_motif = False
    with open(fname, 'r') as f:

        for line in f:

            if line.startswith("ALPHABET= "):
                row_lut = {i:v for i,v in enumerate(line.strip("ALPHABET=").strip())}

            if not line.startswith("MOTIF"):
                # if the line doesn't start with MOTIF and
                # if we're not currently parsing a motif, move on
                if not in_motif:
                    continue
                # if the line doesn't start with MOTIF and
                # if we ARE currently parsing a motif, do the following
                else:
                    # determine which column of data_arr we need to update
                    col_idx = mwidth - pos_left
                    # decrement mwidth so that we can know when to set
                    # in_motif back to False
                    pos_left -= 1

                    # update data_arr with this position's motif data
                    data_arr[:,col_idx] = [float(num) for num in line.strip().split()]

                    # if this is the final position in the motif, set the motif
                    # into motif_list
                    if pos_left == 0:
                        in_motif = False
                        motif_list.append(
                            Motif(
                                identifier = motif_id,
                                alt_name = motif_name,
                                row_lut = row_lut,
                                motif = data_arr,
                                evalue = evalue,
                                motif_type = "sequence",
                            )
                        )
                    
            else:
                if not in_motif:
                    in_motif = True
                # parse motif name line
                _,motif_id,motif_name = line.strip().split(' ')
                # next line is matrix description
                description_line = f.readline()
                # gather info from the description line
                alen = int(alphabet_len_pat.search(description_line).group())
                mwidth = int(motif_width_pat.search(description_line).group())
                pos_left = mwidth
                print(description_line)
                eval_match = eval_pat.search(description_line)
                evalue = float(eval_match.group())
                nsites = int(nsites_pat.search(description_line).group())
                data_arr = np.zeros((alen, mwidth))

    passing_motifs = [motif for motif in motif_list if motif.evalue <= evalue_thresh]
    print(
        f"Done parsing {fname}.\n"\
        f"Of the {len(motif_list)} motifs found, {len(passing_motifs)} passed "\
        f"the e-value threshold of {evalue_thresh}."
    )
                
    return passing_motifs

 
def get_precision_recall(yhat, y, plot_prefix=None):
    ## NOTE: needs expanded for multiclass predictions
    pos_probs = yhat[:, 1]
    prec, recall, _ = metrics.precision_recall_curve(y, pos_probs)
    auc = metrics.auc(recall, prec)
    no_skill = len(y[y==1]) / len(y)

    if plot_prefix is not None:
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
        plt.plot(recall, prec, marker='.', label='Motifs')
        plt.savefig(plot_prefix + ".png")
        plt.savefig(plot_prefix + ".pdf")

    return (prec, recall, auc, no_skill)


def lasso_regression(x, y, c, multi_class="multinomial", penalty="l1", solver="saga",
            max_iter=10000, fit_intercept=True):

    clf_f = LogisticRegression(
        C=c,
        multi_class=multi_class,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        fit_intercept=fit_intercept,
    ).fit(x,y)
    return clf_f


def choose_l1_penalty(x, y, Cs=100, cv=5,
                      multi_class="multinomial", solver="saga",
                      max_iter=10000, fit_intercept=True):

    clf = LogisticRegressionCV(
        Cs=Cs,
        cv=cv,
        multi_class=multi_class,
        penalty='l1',
        solver=solver,
        max_iter=max_iter,
        fit_intercept=fit_intercept,
    ).fit(x, y)

    best_c = cvlogistic.find_best_c(clf)
    return best_c


def wrangle_rust_motif(motif, shape_lut, identifier, alt_name=None):
    """Take information in motif dictionary and reshapes arrays to create
    ndarrays using numpy
    """

    shapes = np.asarray(
        motif['params']['params']['data']
    ).reshape(motif['params']['params']['dim'])

    weights = np.asarray(
        motif['weights']['weights_norm']['data']
    ).reshape(motif['weights']['weights_norm']['dim']) 

    hits = np.asarray(
        motif['hits']['data']
    ).reshape(motif['hits']['dim'])

    dists = np.asarray(
        motif['dists']['data']
    ).reshape(motif['dists']['dim'])

    positions = {'fwd':{}, 'rev':{}}
    for i,vals in enumerate(motif['positions']):
        positions['fwd'][i] = vals['fwd']
        positions['rev'][i] = vals['rev']

    return Motif(
        alt_name = alt_name,
        identifier = identifier,
        row_lut = {v:k for k,v in shape_lut.items()},
        motif = shapes,
        mi =  motif['mi'],
        hits = hits,
        dists = dists,
        weights = weights,
        threshold = motif['threshold'],
        positions = positions,
        zscore = motif['zscore'],
        robustness = motif['robustness'],
        motif_type = "shape",
    )


def consolidate_optim_batches(fname_list):

    opt_results = []
    for fname in fname_list:
        with open(fname, 'rb') as f:
            opt_results.extend(pickle.load(f))

    return opt_results


#@jit(nopython=True)
#def bin_hits(hits_arr, distinct_hits, binned_hits):
#    for bin_id in range(distinct_hits.shape[0]):
#        rows_y = (hits_arr == distinct_hits[bin_id,:])[:,0]
#        binned_hits[rows_y] = bin_id
        

def run_query_over_ref(y_vals, query_shapes, query_weights, threshold,
                       ref, R, W, dist_func, max_count=4, alpha=0.1,
                       parallel=True):

    # R for record number, 2 for one forward count and one reverse count
    hits = np.zeros((R,2))
    
    if parallel:
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
    else:
        optim_generate_peak_array_series(
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
    #unique_hits = np.unique(hits, axis=0)
    this_mi = adjusted_mutual_information(y_vals, hits)

    return this_mi,hits

#@jit(nopython=True, parallel=False)
#def optim_generate_peak_array_series(ref, query, weights, threshold,
#                              results, R, W, dist, max_count, alpha):
#    """Does same thing as generate_peak_vector, but hopefully faster
#    
#    Args:
#    -----
#    ref : np.array
#        The windows attribute of an inout.RecordDatabase object. Will be an
#        array of shape (R,L,S,W,2), where R is the number of records,
#        L is the window size, S is the number of shape parameters, and
#        W is the number of windows for each record.
#    query : np.array
#        A slice of the records and windows axes of the windows attribute of
#        an inout.RecordDatabase object to check for matches in ref.
#        Should be an array of shape (L,S,2), where 2 is for the 2 strands.
#    weights : np.array
#        Weights to be applied to the distance
#        calculation. Should be an array of shape (L,S,1).
#    threshold : float
#        Minimum distance to consider a match.
#    results : 2d np.array
#        Array of shape (R,2), where R is the number of records in ref.
#        This array should be populated with zeros, and will be incremented
#        by 1 when matches are found. The final axis is of length 2 so that
#        we can do the reverse-complement and the forward.
#    R : int
#        Number of records
#    W : int
#        Number of windows for each record
#    dist : function
#        The distance function to use for distance calculation.
#    max_count : int
#        The maximum number of hits to count for each strand.
#    alpha : float
#        Between 0.0 and 1.0, sets the lower limit for the tranformed weights
#        prior to normalizing the sum of weights to one and calculating distance.
#    """
#    
#    for r in range(R):
#        f_maxed = False
#        r_maxed = False
#        for w in range(W):
#            
#            if f_maxed and r_maxed:
#                break
#
#            ref_seq = ref[r,:,:,w,:]
#
#            distances = dist(query, ref_seq, weights, alpha)
#
#            if (not f_maxed) and (distances[0] < threshold):
#                # if a window has a distance low enough,
#                #   add 1 to this result's index
#                results[r,0] += 1
#                if results[r,0] == max_count:
#                    f_maxed = True
#
#            if (not r_maxed) and (distances[1] < threshold):
#                results[r,1] += 1
#                if results[r,1] == max_count:
#                    r_maxed = True

def testing_optim_generate_peak_array(ref, query, weights, threshold,
                              results, R, W, dist, max_count, alpha, dists, lt):
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
    
    for r in range(R):
        f_maxed = False
        r_maxed = False
        for w in range(W):
            
            if f_maxed and r_maxed:
                break

            ref_seq = ref[r,:,:,w,:]

            distances = dist(query, ref_seq, weights, alpha)
            dists[w,:] = distances
            lt[w,:] = distances < threshold

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


#@jit(nopython=True, parallel=True)
#def optim_generate_peak_array(ref, query, weights, threshold,
#                              results, R, W, dist, max_count, alpha):
#    """Does same thing as generate_peak_vector, but hopefully faster
#    
#    Args:
#    -----
#    ref : np.array
#        The windows attribute of an inout.RecordDatabase object. Will be an
#        array of shape (R,L,S,W,2), where R is the number of records,
#        L is the window size, S is the number of shape parameters, and
#        W is the number of windows for each record.
#    query : np.array
#        A slice of the records and windows axes of the windows attribute of
#        an inout.RecordDatabase object to check for matches in ref.
#        Should be an array of shape (L,S,2), where 2 is for the 2 strands.
#    weights : np.array
#        Weights to be applied to the distance
#        calculation. Should be an array of shape (L,S,1).
#    threshold : float
#        Minimum distance to consider a match.
#    results : 2d np.array
#        Array of shape (R,2), where R is the number of records in ref.
#        This array should be populated with zeros, and will be incremented
#        by 1 when matches are found. The final axis is of length 2 so that
#        we can do the reverse-complement and the forward.
#    R : int
#        Number of records
#    W : int
#        Number of windows for each record
#    dist : function
#        The distance function to use for distance calculation.
#    max_count : int
#        The maximum number of hits to count for each strand.
#    alpha : float
#        Between 0.0 and 1.0, sets the lower limit for the tranformed weights
#        prior to normalizing the sum of weights to one and calculating distance.
#    """
#    
#    for r in prange(R):
#        f_maxed = False
#        r_maxed = False
#        for w in range(W):
#            
#            if f_maxed and r_maxed:
#                break
#
#            ref_seq = ref[r,:,:,w,:]
#
#            distances = dist(query, ref_seq, weights, alpha)
#
#            if (not f_maxed) and (distances[0] < threshold):
#                # if a window has a distance low enough,
#                #   add 1 to this result's index
#                results[r,0] += 1
#                if results[r,0] == max_count:
#                    f_maxed = True
#
#            if (not r_maxed) and (distances[1] < threshold):
#                results[r,1] += 1
#                if results[r,1] == max_count:
#                    r_maxed = True
#
#
#@jit(nopython=True)
#def euclidean_distance(vec1, vec2):
#    return np.sqrt(np.sum((vec1 - vec2)**2))
#
#@jit(nopython=True)
#def manhattan_distance(vec1, vec2, w=1):
#    return np.sum(np.abs(vec1 - vec2) * w)
#
#@jit(nopython=True)
#def constrained_manhattan_distance(vec1, vec2, w=1):
#    w_exp = np.exp(w)
#    w = w_exp/np.sum(w_exp)
#    return np.sum(np.abs(vec1 - vec2) * w)
#
#@jit(nopython=True)
#def inv_logit(x):
#    return np.exp(x) / (1 + np.exp(x))
#
#@jit(nopython=True)
#def constrained_inv_logit_manhattan_distance(vec1, vec2, w=1, a=0.1):
#    w_floor_inv_logit = a + (1-a) * inv_logit(w)
#    w_trans = w_floor_inv_logit/np.sum(w_floor_inv_logit)
#    w_abs_diff = (np.abs(vec1 - vec2)) * w_trans
#    #NOTE: this seems crazy, but it's necessary instead of np.sum(arr, axis=(0,1))
#    #  in order to get jit(nopython=True) to work
#    first_sum = np.sum(w_abs_diff, axis=0)
#    second_sum = np.sum(first_sum, axis=0)
#    return second_sum
#
#@jit(nopython=True)
#def hamming_distance(vec1, vec2):
#    return np.sum(vec1 != vec2)

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


class Motif:

    def __init__(
            self, identifier, row_lut, motif,
            alt_name=None, mi=None, hits=None, dists=None,
            weights=None, threshold=None, positions=None,
            zscore=None, robustness=None, evalue=None,
            motif_type=None, enrichments=None, nsites=None,
    ):
        self.identifier = identifier
        self.alt_name = alt_name
        self.row_lut = row_lut
        self.motif = motif
        self.mi = mi
        self.hits = hits
        self.dists = dists
        self.weights = weights
        self.threshold = threshold
        self.positions = positions
        self.zscore = zscore
        self.robustness = robustness
        self.evalue = evalue
        self.motif_type = motif_type
        self.enrichments = enrichments
        self.nsites = nsites

    def __str__(self):
        outstr = self.create_data_header_line()
        outstr += self.create_data_lines()
        if self.motif_type == "shape":
            outstr += self.create_weights_header_line()
            outstr += self.create_weights_lines()
        outstr += "\n"
        return outstr

    def get_enrichments(self, categories, cat_inds):
        '''Calculates and stores motif enrichments in each
        category in rec_db. Modifies self.enrichments
        '''
        self.enrichments = {}
        enr = self.enrichments

        hit_cats,hit_inds = np.unique(
            self.hits,
            return_inverse=True,
            axis=0,
        )

        (hit_vals,cat_vals),contingency_tab = contingency.crosstab(
            hit_inds,
            cat_inds,
        )
        null_tab = np.round(
            contingency.expected_freq(contingency_tab)
        ).astype('int')

        enr["row_hit_vals"] = hit_cats[hit_vals]
        enr["col_cat_vals"] = cat_vals
        enr["ratio"] = np.zeros_like(null_tab, dtype=np.float64)
        enr["pvals"] = np.zeros_like(null_tab, dtype=np.float64)
        enr["test_stats"] = np.zeros_like(null_tab, dtype=np.float64)
        for i in range(null_tab.shape[0]):
            for j in range(null_tab.shape[1]):
                result = rates.test_poisson_2indep(
                    contingency_tab[i,j],
                    np.sum(contingency_tab),
                    null_tab[i,j],
                    np.sum(null_tab),
                )
                enr["ratio"][i,j] = result.ratio
                (enr["test_stats"][i,j], enr["pvals"][i,j]) = result.tuple

        enr["log2_ratio"] = np.log2(
            np.clip(enr["ratio"], EPSILON, np.Inf)
        )

    def create_data_header_line(self):
        """ Method to create a motif header line from a motif

        Returns:
        --------
        string of line to be written
        """
        string = f"MOTIF {self.identifier} {self.alt_name}\n"
        if self.motif_type == "shape":
            string += f"shape-value matrix:"
        else:
            string += f"letter-probability matrix:"
        string += f" alength= {self.motif.shape[0]} w= {self.motif.shape[1]}"
        if self.motif_type == "sequence":
            if self.hits is None:
                string += ""
            else:
                string += f" nsites= {int(np.sum(self.hits))}"
        if self.threshold is not None:
            string += f" threshold= {self.threshold:.3f}"
        if self.mi is not None:
            string += f" adj_mi= {self.mi:.3f}"
        if self.zscore is not None:
            string += f" z-score= {self.zscore:.2f}"
        if self.robustness is not None:
            robustness = f"{self.robustness[0]}/{self.robustness[1]}"
            string += f" robustness= {robustness}"
        if self.evalue is not None:
            string += f" E= {self.evalue}"

        string += "\n"
        return string

    def create_data_lines(self):
        """ Method to create data lines from a motif 

        Returns:
        --------
        string of lines to be written
        """
        string = ""
        for col_idx in range(self.motif.shape[1]):
            string += " "
            string += " ".join(
                [f"{val:.6f}" for val in self.motif[:,col_idx]]
            )
            string += "\n"
        return string

    def create_weights_header_line(self):
        """ Method to create a weights header line from a motif

        Returns:
        --------
        string of line to be written
        """
        string = "weights matrix:\n" 
        return string

    def create_weights_lines(self):
        """ Method to create data lines from a motif 

        Returns:
        --------
        string of lines to be written
        """
        string = ""
        for col_idx in range(self.weights.shape[1]):
            string += " "
            string += " ".join(
                [f"{val:.5f}" for val in self.weights[:,col_idx]]
            )
            string += "\n"
        return string

    def get_rust_dict(self):
        motif_dict = {
            "params": {
                "params": {
                    "v": 1,
                    "dim": list(self.motif.shape),
                    "data": list(self.motif.flatten()),
                }
            },
            "weights": {
                "weights": {
                    "v": 1,
                    "dim": list(self.weights.shape),
                    "data": list(self.weights.flatten()),
                },
                "weights_norm": {
                    "v": 1,
                    "dim": list(self.weights.shape),
                    "data": list(self.weights.flatten()),
                }
            },
            "threshold": self.threshold,
        }

        return motif_dict


class Motifs:

    def __init__(
            self, fname=None, motif_type=None, shape_lut=None,
            max_count=None, alt_name_base=None, evalue_thresh=np.Inf,
    ):

        self.X = None
        self.motif_type = motif_type
        self.var_lut = None
        self.max_count = max_count
        self.bic = None
        self.motifs = []
        self.transform = {}
        self.shape_row_lut = {}
        self.seq_row_lut = {}

        if fname is not None:
            if motif_type == "shape":
                if shape_lut is None:
                    raise Exception(
                        f"You must pass a shape lookup table to read "\
                        f"a json file of motifs."
                    )
                self.motifs = read_shape_motifs(fname, shape_lut, alt_name_base)
                self.motif_type = motif_type
            elif motif_type == "sequence":
                self.motifs = parse_meme_file(fname, evalue_thresh=evalue_thresh)
                self.motif_type = motif_type
            else:
                raise ReadMotifException()

    def __getitem__(self, index):
        return self.motifs[index]

    def __iter__(self):
        for motif in self.motifs:
            yield motif

    def __len__(self):
        return len(self.motifs)

    def __str__(self):
        shapes_str = self.get_shape_str()
        outstr = "ALPHABET= ACGT\n"
        outstr += f"SHAPES= {shapes_str}\n"
        for motif in self.motifs:
            outstr += motif.create_data_header_line()
            outstr += "\n"
        return outstr

    def get_shape_str(self):
        shape_tuples = list([ (v,k) for k,v in self.shape_row_lut.items() ])
        sorted_shape_names = [
            y[0] for y in sorted(shape_tuples, key = lambda x:x[1])
        ]
        shapes_str = ' '.join(sorted_shape_names)
        return shapes_str

    def split_seq_and_shape_motifs(self):
        seq_motifs = Motifs()
        seq_motifs.motifs = [copy.deepcopy(_) for _ in self if _.motif_type == "sequence"]
        seq_motifs.motif_type = "sequence"
        seq_motifs.seq_row_lut = self.seq_row_lut
        shape_motifs = Motifs()
        shape_motifs.motifs = [copy.deepcopy(_) for _ in self if _.motif_type == "shape"]
        shape_motifs.motif_type = "shape"
        shape_motifs.shape_row_lut = self.shape_row_lut
        return(seq_motifs, shape_motifs)

    def write_shape_motifs_as_rust_output(self, out_fname):
        rust_dicts = []
        for motif in self:
            rust_dicts.append(motif.get_rust_dict())
        with open(out_fname, "w") as f:
            json.dump(rust_dicts, f)

    def set_transforms_from_meme_line(self, line):
        """Method to place shape centers and spreads
        into self.transform

        Args:
        -----
        line : str
            Line after "Shape transformations"
        """
        pass

    def set_transforms_from_db(self, rec_db):
        """Method to place shape centers and spreads
        into self.transform

        Args:
        -----
        rec_db : RecordDatabase
            the record database used
        """
        shape_tuples = list(rec_db.shape_name_lut.items())
        sorted_shapes = sorted(shape_tuples, key = lambda x:x[1])
        for name,idx in sorted_shapes:
            center = rec_db.shape_centers[idx]
            spread = rec_db.shape_spreads[idx]
            self.transform[name] = (center,spread)

    def read_file(self, fname):
        """Reads a MEME-like file, potentially with mixed
        sequence and shape motifs.

        Args:
        -----
        fname : str
            Name of meme-like file containing motifs
        """

        motif_list = []

        alphabet_len_pat = re.compile(r'(?<=alength\= )\d+')
        motif_width_pat = re.compile(r'(?<=w\= )\d+')
        eval_pat = re.compile(r'(?<=E\= )\S+')
        nsites_pat = re.compile(r'(?<=nsites\= )\d+')
        threshold_pat = re.compile(r'(?<=threshold\= )\S+')
        ami_pat = re.compile(r'(?<=adj_mi\= )\S+\.\d+')
        zscore_pat = re.compile(r'(?<=zscore\= )\S+\.\d+')
        robustness_pat = re.compile(r'(?<=robustness\= )(\d+)\/(\d+)')

        # start not in_motif
        in_motif = False
        with open(fname, 'r') as f:

            for line in f:

                if line.startswith("ALPHABET= "):
                    self.seq_row_lut = {
                        i:v for i,v in enumerate(line.strip("ALPHABET=").strip())
                    }

                if line.startswith("SHAPES= "):
                    self.shape_row_lut = {
                        i:v for i,v
                        in enumerate(line.strip("SHAPES=").strip().split(" "))
                    }

                if line.startswith("Shape transformations"):
                    # go to next line
                    line = f.readline()
                    transforms = {}
                    elements = line.strip().split(" ")
                    for e in elements:
                        shape_info = e.split(":")
                        center,spread = shape_info[1].split(",")
                        transforms[shape_info[0]] = (float(center), float(spread))
                    self.transform = transforms

                ##################################################################
                ##################################################################
                ## needs updated to read meme files with both seq AND shape ######
                ##################################################################
                ##################################################################
                if not line.startswith("MOTIF"):
                    # if the line doesn't start with MOTIF and
                    # if we're not currently parsing a motif, move on
                    if not in_motif:
                        continue
                    # if the line doesn't start with MOTIF and
                    # if we ARE currently parsing a motif, do the following
                    else:
                        # determine which column of data_arr we need to update
                        col_idx = mwidth - pos_left
                        # subtract from mwidth so that we can know when to set
                        # in_motif back to False
                        pos_left -= 1

                        if in_data:
                            # update data_arr with this position's motif data
                            data_arr[:,col_idx] = [
                                float(num) for num in line.strip().split()
                            ]
                        elif in_weights:
                            # update weights_arr with this position's motif data
                            weights_arr[:,col_idx] = [
                                float(num) for num in line.strip().split()
                            ]

                        # if this is the final position in the motif,
                        # check whether we're entering weights, or whether
                        # we've left the motif
                        if pos_left == 0:
                            # we're definitely not in_data
                            in_data = False
                            # check next line to see whether we're entering weights
                            line = f.readline()
                            # if we're in_weights, set to True and re-set pos_left
                            if line.startswith("weights matrix"):
                                in_weights = True
                                pos_left = mwidth
                            # if we've exited the motif entirely, set in_motif and
                            # in_weights to False and put info into Motif
                            else:
                                in_motif = False
                                in_weights = False
                                motif_list.append(
                                    Motif(
                                        identifier = motif_id,
                                        alt_name = motif_name,
                                        row_lut = row_lut,
                                        motif = data_arr,
                                        evalue = evalue,
                                        motif_type = motif_type,
                                        mi = ami,
                                        weights = weights_arr,
                                        zscore = zscore,
                                        robustness = robustness,
                                        nsites = nsites,
                                        threshold = threshold,
                                    )
                                )
                        
                else:
                    if not in_motif:
                        in_motif = True
                        in_data = True
                    # parse motif name line
                    _,motif_id,motif_name = line.strip().split(' ')
                    # next line is matrix description
                    description_line = f.readline()
                    # gather info from the description line
                    if description_line.startswith("shape-value"):
                        motif_type = "shape"
                        row_lut = self.shape_row_lut
                    elif description_line.startswith("letter-probability"):
                        motif_type = "sequence"
                        row_lut = self.seq_row_lut

                    mo = alphabet_len_pat.search(description_line)
                    alen = int(evaluate_match_object(mo))

                    mo = motif_width_pat.search(description_line)
                    mwidth = int(evaluate_match_object(mo))
                    pos_left = mwidth

                    eval_match = eval_pat.search(description_line)
                    evalue = evaluate_match_object(eval_match)

                    mo = nsites_pat.search(description_line)
                    nsites = evaluate_match_object(mo)

                    mo = threshold_pat.search(description_line)
                    threshold = evaluate_match_object(mo)

                    mo = ami_pat.search(description_line)
                    ami = evaluate_match_object(mo)

                    mo = zscore_pat.search(description_line)
                    zscore = evaluate_match_object(mo)

                    mo = robustness_pat.search(description_line)
                    if mo is not None:
                        robustness = [int(val) for val in mo.group(1,2)]
                    else:
                        robustness = None

                    data_arr = np.zeros((alen, mwidth))
                    if motif_type == "shape":
                        weights_arr = np.zeros((alen, mwidth))
                    else:
                        weights_arr = None

        self.motifs = motif_list

    def plot_shapes_and_weights(self):
        pass

    def write_enrichment_heatmap(self):
        pass

    def write_file(self, fname, rec_db):
        """ Method to write a file from Motifs object

        Args:
        -----
        fname : str
            name of outputfile
        rec_db : RecordDatabase
            whole database used
        """
        shapes_str = self.get_shape_str()
        with open(fname, mode="w") as f:

            f.write("MEME version 4\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write(f"SHAPES= {shapes_str}\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")
            f.write(rec_db.create_transform_lines())

            for i, motif in enumerate(self.motifs):
                f.write(motif.create_data_header_line())
                f.write(motif.create_data_lines())
                if motif.motif_type == "shape":
                    f.write(motif.create_weights_header_line())
                    f.write(motif.create_weights_lines())
                f.write("\n")

    def supplement_robustness(self, rec_db, binary, my_env=None):

        ami_pat = re.compile(r'(?<=adj_mi\= )\S+\.\d+')
        robustness_pat = re.compile(r'(?<=robustness\= )\((\d+), (\d+)')
        zscore_pat = re.compile(r'(?<=zscore\= )\S+\.\d+')

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_direc = tmp_dir.name
        y_name = os.path.join(tmp_direc, "tmp_y.npy")
        np.save(y_name, rec_db.y.astype(np.int64))
        #print(rec_db.y.shape)

        binary += f" {y_name}"

        for motif in self.motifs:

            hits_name = os.path.join(tmp_direc, "tmp_hits.npy")
            #print(motif.hits.shape)
            np.save(hits_name, motif.hits.flatten().astype(np.int64))

            cmd = binary + f" {hits_name}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                env=my_env,
            )
            if result.returncode != 0:
                raise(Exception(
                    f"Supplementing sequence motifs with robustness failed:\n"\
                    f"STDOUT: {result.stdout.decode()}\n"
                    f"ERROR: {result.stderr.decode()}\n"\
                ))
            output = result.stdout.decode()
            try:
                motif.mi = float(ami_pat.search(output).group())
                passes = int(robustness_pat.search(output).group(1))
                attempts = int(robustness_pat.search(output).group(2))
                motif.robustness = (passes, attempts)
                motif.zscore = float(zscore_pat.search(output).group())
            except:
                raise(Exception(
                    f"Something went wrong in supplementing robustness:\n\n"\
                    f"Looked for {ami_pat} in:\n"\
                    f"STDOUT: {result.stdout.decode()}\n\n"\
                    f"ERROR: {result.stderr.decode()}\n\n"\
                ))

    #def to_tidy(self, outfile):
    #    """ Method to write file in a tidy format for data analysis

    #    Args
    #        outfile(str) - name of outputfile
    #    """
    #    with open(outfile, mode = "w") as f:
    #        header = ",".join(self.motifs[0]['seed'].names)
    #        header += ",bp,name\n"
    #        f.write(header)
    #        for i, motif in enumerate(self.motifs):
    #            if not "name" in motif:
    #                motif["name"] = "motif_%i"%(i)
    #        for i, col in enumerate(motif['seed'].matrix().transpose()):
    #            string = ""
    #            string += ",".join(["%f"%val for val in col])
    #            string += ",%d,%s\n"%(i,motif["name"])
    #            f.write(string)
 
    def get_enrichments(self, rec_db):
        '''
        Args:
        -----
        rec_db: RecordDatabase

        Modifies self inplace
        '''

        for motif in self.motifs:

            categories,cat_inds = np.unique(
                rec_db.y,
                return_inverse=True,
            )
            motif.get_enrichments(categories, cat_inds)

            #hit_cat_list = np.unique(motif["hits"], axis=0)
            #hit_names = [f"hit_{hit_idx}" for hit_idx,hit in enumerate(hit_cat_list)]
            #form_hit_names = [f"hit_{hit_idx}" for hit_idx,hit in enumerate(hit_cat_list) if hit_idx > 0]
            #motif["cat_vars"] = form_cat_names
            #motif["hit_vars"] = form_hit_names

            #covar_names = hit_names + cat_names
            ##col_names = ["count"] + covar_names
            #n = len(hit_cat_list)*len(categories)

            #counts = np.zeros((n,1))#, dtype=np.uint64)

            #cat_X = np.zeros((n, len(categories)))#, dtype=np.uint8)
            #hit_X = np.zeros((n, len(hit_cat_list)))#, dtype=np.uint8)

            #y_idx = 0
            #for cat_idx,category in enumerate(categories):
            #    is_this_cat = records.y == category
            #    for hit_idx,hits in enumerate(hit_cat_list):
            #        is_this_hit = np.all(motif["hits"] == hits, axis=1)
            #        is_this = np.bitwise_and(is_this_cat, is_this_hit)
            #        this_count = np.sum(is_this)

            #        counts[y_idx, 0] = this_count
            #        cat_X[y_idx, cat_idx] = 1
            #        hit_X[y_idx, hit_idx] = 1
            #        y_idx += 1

            #covars = np.append(hit_X, cat_X, axis=1)
            ##data = np.append(counts, covars, axis=1)
            ##tmp_df = pd.DataFrame(data=data, columns=col_names)
            #tmp_df = pd.DataFrame(data=covars, columns=covar_names)
            #interaction_terms = []
            #for form_cat_name in form_cat_names:
            #    for form_hit_name in form_hit_names:
            #        term = f"{form_hit_name}:{form_cat_name}"
            #        tmp_df[term] = (
            #            tmp_df["form_hit_name"]
            #            * tmp_df["form_cat_name"]
            #        )
            #        interaction_terms.append(term)
            #motif["count_df"] = tmp_df.drop(["hit_0", "cat_0"], axis=1)
####    ##############################################################
            #model = linear_model.LogisticRegression(
            #    penalty = None,
            #    multi_class = "multinomial",
            #    fit_intercept = True,
            #)
            #        
            #contingency_fit = model.fit(
            #    motif["count_df"].loc[[]]
            #    motif["count_df"].count.values,
            #)

            #with localconverter(ro.default_converter + pandas2ri.converter):
            #    r_count_df = ro.conversion.py2rpy(motif["count_df"])

            #formula = "count ~ "
            ## place +-separated category names as covars
            #formula += " + ".join(form_cat_names)
            #formula += " + "
            ## place +-separated hit names as covars
            #formula += " + ".join(form_hit_names)
            #row_col_formula = formula
            ## place interaction terms as covars
            #for cat_name in cat_names:
            #    if cat_name == "cat_0":
            #        continue
            #    for hit_name in hit_names:
            #        if hit_name == "hit_0":
            #            continue
            #        covar_name = f" + {cat_name}:{hit_name}"
            #        formula += covar_name

            #print(f"Fitting category counts to the formula: {formula}")
            #motif["contingency_fit"] = brms.brm(
            #    stats.as_formula(formula),
            #    data = r_count_df,
            #    family = "poisson",
            #    #prior = base.c(
            #    #    brms.set_prior("normal(-50,20)", cl="b", coef="cat_1.0"),
            #    #    brms.set_prior("normal(20,3)", cl="b", coef="hit_0"),
            #    #)
            #)
            #print(f"Fitting category counts to the formula: {row_col_formula}")
            #motif["null_fit"] = brms.brm(
            #    stats.as_formula(row_col_formula),
            #    data = r_count_df,
            #    family = "poisson",
            #    #prior = base.c(
            #    #    brms.set_prior("normal(-50,20)", cl="b", coef="cat_1.0"),
            #    #    brms.set_prior("normal(20,3)", cl="b", coef="hit_0"),
            #    #)
            #)

            #c_df = base.as_data_frame(motif["contingency_fit"])
            #i_df = base.as_data_frame(motif["null_fit"])
            #with localconverter(ro.default_converter + pandas2ri.converter):
            #    motif["contingency_fit_samples"] = ro.conversion.rpy2py(c_df)
            #    motif["null_fit_samples"] = ro.conversion.rpy2py(i_df)


    def merge_with_motifs(self, other):

        orig_covar_num = self.X.shape[1]
        new_motif_idx = len(self)

        self.X = np.append(self.X, other.X, axis=1)
        self.var_lut = self.var_lut.copy()
        self.motifs.extend(other.motifs)

        for other_motif_key,other_motif_info in other.var_lut.items():
            new_key = other_motif_key + orig_covar_num
            other_motif_info['motif_idx'] = new_motif_idx
            self.var_lut[new_key] = other_motif_info
            new_motif_idx += 1
        
    def new_with_motifs(self, other):
        '''copies self and merges with other Motifs object.
        Returns a new Motifs object
        '''
        print("Merging two Motifs objects into a single, new Motifs object")
        new_motifs = copy.deepcopy(self)
        other_motifs = copy.deepcopy(other)
        new_motifs.merge_with_motifs(other_motifs)
        
        return new_motifs

    def get_distinct_ids(self):
        return set([motif.identifier for motif in self])

    def get_X(self, max_count=None, fimo_fname=None, rec_db=None):
        if self.motif_type == "shape":
            self.prep_shape_logit_reg_data(max_count)
        elif self.motif_type == "sequence":
            self.prep_sequence_logit_reg_data(fimo_fname, rec_db)

    def prep_sequence_logit_reg_data(self, fimo_fname, rec_db):

        fimo_file = fimo.FimoFile()
        fimo_file.parse(fimo_fname)
        ids_in_self = self.get_distinct_ids()
        retained_hits = fimo_file.filter_by_id(ids_in_self)

        self.X,self.var_lut = retained_hits.get_design_matrix(
            rec_db,
            motif_list = self.motifs,
        )
        try:
            for i,motif in enumerate(self.motifs):
                motif.hits = self.X[:,i][:,None]
        except:
            logging.error(
                f"Problem creating X array for sequence motif logistic regression.\n"\
                f"X array shape: {self.X.shape}\n"\
                f"motif ids: {ids_in_self}\n"\
                f"motif lut: {self.var_lut}\n"\
            )
            sys.exit(1)


    def filter_motifs(self, coefs):
        '''Determines which coeficients were shrunk to zero during LASSO regression
        and removes motifs for which all covariates in X were zero. Returns
        a filtered set of motifs, a new array of X values (motif hits covariates),
        and a new var_lut to map columns of the new X array to motif information.

        Returns:
        --------
        Returns the filtered array of coefficients corresponding to each
        coefficient for each motif in the updates motifs in self.

        Modifies:
        ---------
        Modifies self in place
        '''

        # keys are X arr indices, vals are dict
        # of {'motif_idx': motif index in list of motifs,
        #     'hits': the class of hit this covariate represents, i.e., [0,1], [1,1], etc.}
        new_lut = {}
        # construct lookup table where motif index is key, and value is list
        #  of column indices for that motif in coefs
        motif_lut = {}

        #print("------------------------------")
        #print(var_lut)
        #print("------------------------------")
        for k,coef in self.var_lut.items():
            # if this motif index isn't yet in the lut, place it in and give it a list
            if not coef['motif_idx'] in motif_lut:
                # k+1 here, since coefs will have the intercept at index 0
                # and k is the index in the covariates array
                motif_lut[coef['motif_idx']] = [k+1]
            # if this motif idx is already present, append col to list
            else:
                motif_lut[coef['motif_idx']].append(k+1)

        retain = []
        # keep the first column (intercept)
        retain_coefs = [0]
        # make nrow-by-zero array to start appending covariates from coeficiens with 
        # predictive value
        retained_X = np.zeros((self.X.shape[0],0))
        #print(retained_X.shape)
        # now go through coefs columns to see whether any motif has all zeros
        #print("------------------------------")
        #print(motif_lut)
        #print("------------------------------")
        for motif_idx,motif_coef_inds in motif_lut.items():
            # instantiate a list to carry bools
            motif_any_nonzero = []
            this_motif_coef_inds = []
            for coef_idx in motif_coef_inds:
                # are any of these values non-zero?
                #print(f"motif_idx: {motif_idx}")
                #print(f"coef_idx: {coef_idx}")
                #print(f"coefs: {coefs[:,coef_idx]}")
                has_non_zero = np.any(coefs[:,coef_idx] != 0)
                #print(f"has_non_zero: {has_non_zero}")
                if has_non_zero:
                    retained_X = np.append(
                        retained_X,
                        self.X[:,coef_idx-1][:,None],
                        axis=1,
                    )
                    this_col_idx = retained_X.shape[1] - 1
                    new_lut[this_col_idx] = {
                        'motif_idx': len([_ for _ in retain if _]),
                        'hits': self.var_lut[coef_idx-1]['hits'],
                    }
                motif_any_nonzero.append(has_non_zero)
                this_motif_coef_inds.append(coef_idx)
            
            # if any column for this motif contained any non-zero values, retain it
            # and all its coefficients
            #print(f"motif_any_nonzero: {motif_any_nonzero}")
            if np.any(motif_any_nonzero):
                retain.append(True)
                for mo_idx in this_motif_coef_inds:
                    retain_coefs.append(mo_idx)
            else:
                retain.append(False)
                print(
                    f"WARNING: all regression coefficients for motif at "\
                    f"index {motif_idx} were shrunken to 0 during LASSO regression. "\
                    f"The motif has been removed from further consideration."
                )
            
        #print(f"retain: {retain}")
        # keep motifs for which at least one coefficient was non-zero
        retained_motifs = [self[i] for i,_ in enumerate(retain) if _]

        retained_coefs = np.zeros((coefs.shape[0], len(retain_coefs)))
        for (new_i,filt_i) in enumerate(retain_coefs):
            retained_coefs[:,new_i] = coefs[:,filt_i]

        self.motifs = retained_motifs
        self.X = retained_X
        self.var_lut = new_lut

        return(retained_coefs)


    def prep_shape_logit_reg_data(self, max_count):
        """Converts motif hit categories to X matrix of
        variables for logistic regression.
        """

        n = max_count + 1
        max_cat = int(n*n - n*(n-1)/2) - 1 # minus one to get rid of [0,0] category
        possible_cats = [_ for _ in range(max_cat)]
        rec_num = self[0].hits.shape[0]

        self.X = np.zeros((rec_num, max_cat*len(self)),dtype="uint8")

        self.var_lut = {}
        col_idx = 0

        for motif_idx,motif in enumerate(self):
            for i in range(n):
                for j in range(n):
                    # skip [0,0] case since that will be in intercept
                    if (i == 0) and (j == 0):
                        continue
                    # don't do lower triangle
                    if j < i:
                        continue
                    
                    hit = [i,j]
                    # get the appropriate motif

                    rows = np.all(motif.hits == hit, axis=1)
                    self.X[rows,col_idx] = 1

                    self.var_lut[col_idx] = {
                        'motif_idx': motif_idx,
                        'hits': hit,
                    }

                    col_idx += 1


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

    def write(self, fhandle, wrap = 70, delim = None):
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

    def __getitem__(self, sliced):
        if isinstance(sliced, int):
            sliced = tuple([sliced])
        subset = FastaFile()
        seq_names = [self.names[idx] for idx in sliced]
        seq_data = { seq_name:self.data[seq_name] for seq_name in seq_names }
        subset.names = seq_names
        subset.data = seq_data
        return subset

    def __len__(self):
        return len(self.names)

    def split_kfold(self, k, yvals, rng_seed=None):
        """Takes stratified samples of indices of records in self to
        grease the wheels of doing k-fold crossvalidation.

        Args:
        -----
        k : int
            The number of folds into which to split the data.
        yvals : 1D np array
            Categorical y-values for setting up stratifed splitting.

        Returns:
        --------
        folds : list
            List of length k
        """

        if rng_seed is None:
            rng_seed = int(time.time())

        folds = []

        skf = StratifiedKFold(
            n_splits = k,
            shuffle = True,
            # set for reproducibility
            random_state = rng_seed,
        )

        skf_inds = skf.split(self.names, yvals)

        folds = []

        for fold,(train_inds,test_inds) in enumerate(skf_inds):
            train_seq = self[train_inds]
            test_seqs = self[test_inds]
            train_y = yvals[train_inds]
            test_y = yvals[test_inds]

            folds.append(((train_seq,train_y),(test_seqs,test_y)))

        return folds


    def sample(self, n, yvals, rng_seed=None):
        """Useful for down-sampling the records in self. Sampling is stratified
        by values in yvals, so yvals must be categorical for this to work as desired.

        Args:
        -----
        n : int
            The final number of (randomly sampled) records to return. Sampling
            is stratified by the classes found in yvals
        yvals : 1D np.array
            categorical assignments for each record in self.
        rng_seed : int
            Sets seed to random number generator for reproducible sampling.
        """

        if rng_seed is None:
            rng_seed = int(time.time())

        total = len(self)
        if total <= n:
            logging.error(
                f"To sample from a FastaFile, n must be less than the "\
                f"number of records in the file. You set n to {n}, but "\
                f"there are {total} records. Exiting now."
            )
            sys.exit(1)
        inds = list(range(total))
        distinct_cats = np.unique(yvals)
        strat_w = np.zeros_like(yvals)
        for cat in distinct_cats:
            mask = yvals == cat
            n_cat = np.sum(mask)
            strat_w[mask] = n_cat / total

        strat_w = strat_w / strat_w.sum()

        # stratified random sample of record indices
        rng = np.random.default_rng(rng_seed)
        samp_inds = rng.choice(inds, size=n, replace=False, p=strat_w)

        sampled_records = self[samp_inds]
        sampled_yvals = yvals[samp_inds]
        return samp_inds,sampled_records,sampled_yvals


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
                 shape_names=None, record_names=None, weights=None,
                 shift_params=["HelT", "Roll"],
                 exclude_na=True):

        self.record_name_list = []
        self.record_name_lut = {}
        self.shape_name_lut = {}
        self.normalized = False
        if X is None:
            self.X = None
        else:
            if shape_names is None:
                raise SetNamesException(
                    f"ERROR: X values were given without names for the shapes. "\
                    f"Reinitialize, setting the shape_names argument."
                )
            self.X = X
            self.shape_name_lut = {name:i for i,name in enumerate(shape_names)}
        if y is not None:
            if record_names is None:
                raise SetNamesException(
                    "ERROR: y values were given without names for the records. "\
                    f"Reinitialize, setting the record_names argument"
                )
            self.y = y
            self.record_name_lut = {name:i for i,name in enumerate(record_names)}
        if weights is not None:
            self.weights = weights
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


    def write_to_files(self, out_direc, fname_base):
        """Writes shapes to fasta files and scores to txt file.

        Returns:
        --------
        A list with the follwing format:
            [score_fname, [shape_fname1, shape_fname2, ...]]
        """

        score_fname = os.path.join(out_direc, fname_base) + ".txt"
        with open(score_fname, "w") as score_f:
            score_f.write("name\tscore")
            for rec_name,rec_idx in self.record_name_lut.items():
                val = self.y[rec_idx]
                score_f.write(f"\n{rec_name}\t{val}")

        shape_fnames = []
        for shape_name,shape_idx in self.shape_name_lut.items():
            shape_fname = os.path.join(out_direc, fname_base) + f".fa.{shape_name}"
            shape_fnames.append(shape_fname)

            with open(shape_fname, "w") as shape_f:
                firstline = True
                for rec_name,rec_idx in self.record_name_lut.items():

                    if firstline:
                        record_str = f">{rec_name}\n"
                        firstline = False
                    else:
                        record_str = f"\n>{rec_name}\n"

                    seq = self.X[rec_idx,:,shape_idx,0]
                    seq_str = ",".join([f"{val:.2f}" for val in seq])

                    record_str += seq_str
                    shape_f.write(record_str)
        return (score_fname, shape_fnames)


    def set_records_inplace(self, inds):
        self.X = self.X[inds,...]
        self.y = self.y[inds,...]
        try:
            self.weights = self.weights[inds,...]
        except AttributeError:
            weights = np.ones_like(self.X)
            self.weights = weights / weights.sum()

        rev_rec_lut = {v:k for k,v in self.record_name_lut.items()}
        rec_names = [ rev_rec_lut[idx] for idx in inds ]
        self.record_name_lut = { name:i for i,name in enumerate(rec_names) }
        
        rev_shape_lut = {v:k for k,v in self.shape_name_lut.items()}
        shape_names = [ rev_shape_lut[idx] for idx in rev_shape_lut.keys() ] 
        self.shape_name_lut = { name:i for i,name in enumerate(shape_names) }


    def subset_records(self, inds):
        X = self.X[inds,...]
        y = self.y[inds,...]
        try:
            weights = self.weights[inds,...]
        except AttributeError:
            weights = np.ones_like(X)
            weights = weights / weights.sum()

        rev_rec_lut = {v:k for k,v in self.record_name_lut.items()}
        rec_names = [ rev_rec_lut[idx] for idx in inds ]

        rev_shape_lut = {v:k for k,v in self.shape_name_lut.items()}
        shape_names = [ rev_shape_lut[idx] for idx in range(len(rev_shape_lut)) ] 

        db = RecordDatabase(
            y = y,
            X = X,
            shape_names = shape_names,
            record_names = rec_names,
            weights = weights,
        )
        return db

    def split_kfold(self, k, seqs=None, rng_seed=None):
        """Makes this database into a list of 2-tuples. Each 2-tuple is 
        a paired set of training/testing data. The first element of each 2-tuple
        is itself a 2-tuple of (training_shapes, training_sequences), and the
        second element of each 2-tuple is itself a 2-tuple of
        (test_shapes, test_sequences). The length of the returned list
        of 2-tuples is equal to k.

        Args:
        -----
        k : int
            The number of folds into which to split the data.
        seqs : FastaFile or None
            If doing both sequence and shape motif inference, a FastaFile
            object can be used for this argument so as to do k-fold splitting
            of the sequences and shapes together.

        Returns:
        --------
        folds : list
            List of length k
        """

        if rng_seed is None:
            rng_seed = int(time.time())

        folds = []

        skf = StratifiedKFold(
            n_splits = k,
            shuffle = True,
            # set for reproducibility
            random_state = rng_seed,
        )

        skf_inds = skf.split(self.X, self.y)

        folds = []

        for fold,(train_inds,test_inds) in enumerate(skf_inds):
            train_shapes = self.subset_records(train_inds)
            test_shapes = self.subset_records(test_inds)

            if seqs is not None:
                train_seqs = seqs[train_inds]
                test_seqs = seqs[test_inds]
            else:
                train_seqs = None
                test_seqs = None

            folds.append(((train_shapes,train_seqs),(test_shapes,test_seqs)))

        return folds

    def sample(self, n, inplace=False, rng_seed=None):
        """Useful for down-sampling the records in self. Sampling is stratified
        by values in self.y, so y must be categorical for this to work as desired.

        Args:
        -----
        n : int
            The final number of (randomly sampled) records to return. Sampling
            is stratified by the classes found in self.y
        inplace : bool
            Sets whether to modify self in place, or whether to return
            a copy of the sampled data from self.
        """

        if rng_seed is None:
            rng_seed = int(time.time())

        total = len(self)
        if total <= n:
            logging.error(
                f"To sample from a database, n must be less than the "\
                f"number of records in the database. You set n to {n}, but "\
                f"there are {total} records. Exiting now."
            )
            sys.exit(1)
        inds = list(range(total))
        distinct_cats = np.unique(self.y)
        strat_w = np.zeros_like(self.y)
        for cat in distinct_cats:
            mask = self.y == cat
            n_cat = np.sum(mask)
            strat_w[mask] = n_cat / total

        strat_w = strat_w / strat_w.sum()

        # stratified random sample of record indices
        rng = np.random.default_rng(rng_seed)
        samp_inds = rng.choice(inds, size=n, replace=False, p=strat_w)

        if inplace:
            self.set_records_inplace(samp_inds)
            return samp_inds
        else:
            sampled_db = self.subset_records(samp_inds)
            return (sampled_db,samp_inds)

    def seqs_per_bin(self):
        """ Method to determine how many sequences are in each category

        Returns:
            outstring - a string enumerating the number of seqs in each category
        """
        string = ""
        for value in np.unique(self.y):
            user_cat = self.category_lut[value]
            string += "\nCat {}: {}".format(
                user_cat, np.sum(self.y ==  value)
            )
        return string


    def create_transform_lines(self):
        """ Method to create a transform line

        Returns:
        --------
        string of line to be written
        """

        shape_tuples = list(self.shape_name_lut.items())
        sorted_shapes = sorted(shape_tuples, key = lambda x:x[1])
        string = f"Shape transformations\n"
        transformations = []
        for name,idx in sorted_shapes:
            center = self.shape_centers[idx]
            spread = self.shape_spreads[idx]
            transformations.append(f"{name}:{center},{spread}")
        string += " ".join(transformations)
        string += "\n\n"
        return string

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

    def quantize_quant(self,nbins=10):
        """Quantize data into n equally populated bins according to the
        nbins quantiles
        
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
        logging.warning("Quantizing on bins: {}".format(bins))
        # subtract 1 to ensure categories start with 0
        self.y = np.digitize(values, bins) - 1
        return bins

    def discretize_quant(self, nbins=10):
        """Discretize data into nbins bins according to K-means clustering
        
        Prints bin divisions to the logger

        Modifies:
        ---------
        self.y : np.array
            converts the values into their new categories
        """

        values = self.y
        k_means = cluster.KMeans(n_clusters=nbins)
        k_means.fit(values.reshape(-1,1))
        self.y[...] = k_means.labels_
        print(self.y)


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
                self.record_name_list.append(linearr[0])
                scores.append(float(linearr[1]))
            self.y = np.asarray(
                scores,
                dtype=float,
            )

    def set_category_lut(self):
        y_copy = self.y.copy()
        distinct_cats = np.sort(np.unique(self.y))
        self.category_lut = {}
        for (i,category) in enumerate(distinct_cats):
            self.category_lut[i] = category
            y_copy[self.y == category] = i

        print(f"Category lookup table\n{self.category_lut}")
        self.y = y_copy

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

        for i,rec_name in enumerate(self.record_name_list):
            self.record_name_lut[rec_name] = i

        for shape_name,shape_infname in shape_dict.items():

            #print(shape_infname)
            #print(shape_name)

            if not shape_name in self.shape_name_lut:
                self.shape_name_lut[shape_name] = shape_idx
                s_idx = shape_idx
                shape_idx += 1

            this_shape_dict = parse_shape_fasta(shape_infname)
            #print(this_shape_dict)

            if self.X is None:

                record_count = len(this_shape_dict)
                for rec_idx,rec_data in enumerate(this_shape_dict.values()):
                    if rec_idx > 0:
                        break
                    record_length = len(rec_data)

                self.X = np.zeros((record_count,record_length,shape_count,2))
                #print(self.X.shape)

            for rec_name,rec_data in this_shape_dict.items():
                #print(rec_name)
                r_idx = self.record_name_lut[rec_name]

                if shape_name in shift_params:
                    #while len(rec_data) < self.X.shape[1]:
                    rec_data = np.append(rec_data, np.nan)
                    rec_data = np.append(np.nan, rec_data)
                    fwd_data = rec_data[1:]
                    rev_data = rec_data[1:]
                    rev_data = rev_data[::-1]
                    #if (
                    #    (len(fwd_data) != record_length)
                    #    | (len(rev_data) != record_length)
                    #):
                    #    logging.error(
                    #        f"ERROR: the record named {rec_name} in file "\
                    #        f"{shape_infname} is not "\
                    #        f"the same length as the other records. "\
                    #        f"{rec_name} is {len(fwd_data)}, but record_length "\
                    #        f"is {record_length}. "\
                    #        f"All records must be "\
                    #        f"the same length. Exiting without inferring motifs."
                    #    )
                    #    sys.exit(1)

                    self.X[r_idx,:,s_idx,0] = fwd_data
                    self.X[r_idx,:,s_idx,1] = rev_data
                else:
                    if len(rec_data) != record_length:
                        logging.error(
                            f"ERROR: the record named {rec_name} in file "\
                            f"{shape_infname} is not "\
                            f"the same length as the other records. "\
                            f"{rec_name} is {len(rec_data)}, but record_length "\
                            f"is {record_length}. "\
                            f"All records must be "\
                            f"the same length. Exiting without inferring motifs."
                        )
                        sys.exit(1)
                    self.X[r_idx,:,s_idx,0] = rec_data
                    self.X[r_idx,:,s_idx,1] = rec_data[::-1]

        if exclude_na:

            # remove the first- and final two bases from each shape/record
            self.X = self.X[:,2:-2,:,:]
            # identifies which positions have at least one NA for any record
            complete_records = ~np.any(np.isnan(self.X), axis=(1,2,3))
            incomplete_records = ~complete_records
            # remove na-containing records from X and y
            self.X = self.X[complete_records, ...]
            self.y = self.y[complete_records]
            self.record_name_list = [
                name for (name,complete)
                in zip(self.record_name_list, complete_records)
                if complete
            ]

            self.record_name_lut = {}
            for i,rec_name in enumerate(self.record_name_list):
                self.record_name_lut[rec_name] = i

            has_na = np.any(np.isnan(self.X))

            if has_na:
                try:
                    raise Exception()
                except Exception as e:
                    logging.error(
                        f"ERROR: after clipping the first two, and final two bases "\
                        f"from all records in the input data, NaN values "\
                        f"remained! Exiting the script now. Thoroughly examine "\
                        f"your input data for non-canonical bases and other "\
                        f"potential causes of this issue."
                    )
                    sys.exit(1)
            self.complete_records = complete_records

    #def set_center_spread(self, centers, spreads):
    #    """Sets the centers and spreads for the shape parameters.

    #    Args:
    #    -----
    #    centers : dict
    #        keys are shape names, values are center for each shape
    #    spreads : dict
    #        keys are shape names, values are spread for each shape
    #    """
    #    self.shape_name_lut.items()
    #    shape_count = self.X.shape[2]
    #    for s in range(shape_count):
    #        yield self.X[:,:,s]


    def determine_center_spread(self, method=robust_z_csp):
        """Method to get the center and spread for each shape based on
        all records in the database.

        This will ignore any Nans in any shape sequence scores. First
        calculates center (median of all scores) and spread (MAD of all scores)

        Modifies:
        ---------
        self.shape_centers - populates center with calculated values
        self.shape_spreads - populates spread with calculated values
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
        """Provides the weights attribute. Normalized such
        that the weights for all shapes/positions in a given
        record sum to one.
        """
        self.weights = np.ones_like(self.X) / self.X.size


    def compute_windows(self, wsize):
        """Method to precompute all nmers.

        Modifies:
            self.windows
        """

        #(R,L,S,W)
        rec_num, rec_len, shape_num, strand_num = self.X.shape
        window_count = rec_len - wsize + 1
        
        self.windows = np.zeros((
            rec_num, wsize, shape_num, window_count, strand_num
        ))

        for i,rec in enumerate(self.iter_records()):
            for j in range(window_count):
                self.windows[i, :, :, j, :] = self.X[i, j:(j+wsize), :, :]

    def permute_records(self):

        rand_order = np.random.permutation(self.X.shape[0])

        permuted_shapes = self.X[rand_order,...]
        permuted_vals = self.y[rand_order,...]
        permuted_record_names = [ self.record_name_list[idx] for idx in rand_order ]
        self.record_name_list = permuted_record_names
        
        for i,rec_name in enumerate(permuted_record_names):
            self.record_name_lut[rec_name] = i

        self.X = permuted_shapes
        self.y = permuted_vals


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
            f"Using {len(total_seeds)} random seeds to determine "\
            f"threshold from pairwise distances"
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

        logging.info(f"Threshold mean: {mean} and stdev {stdev}")
        thresh = max(mean - threshold_sd_from_mean * stdev, 0)
        logging.info(
            f"Setting initial threshold for each seed "\
            f"to {thresh}"
        )

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


class FIREfile(object):

    def __init__(self):
        self.data = {}
        self.names = []

    def __iter__(self):
        for name in self.names:
            yield (name, self.data[name])

    def __len__(self):
        return len(self.names)

    def __add__(self, other):
        newfile = FIREfile()
        for name, score in self:
            newfile.add_entry(name, score)
        for name, score in other:
            newfile.add_entry(name, score)
        return newfile

    def add_entry(self, name, score):
        self.data[name] = score
        self.names.append(name)

    def pull_value(self, name):
        return self.data[name]

    def discretize_quant(self, nbins=10):
        # first pull all the values
        all_vals = [val for name, val in self]
        all_vals = np.array(all_vals)
        quants = np.arange(0,100, 100.0/nbins)
        bins = []
        for quant in quants:
            bins.append(np.percentile(all_vals, quant))
        all_vals = np.digitize(all_vals, bins)
        for new_val, (name, old_val) in zip(all_vals, self):
            self.data[name] = new_val

    def shuffle(self):
        size = len(self)
        subset = np.random.permutation(size)
        shuffled_names = [self.names[val] for val in subset]
        self.names = shuffled_names


    def write(self, fname):
        with open(fname, mode="w") as outf:
            outf.write("name\tscore\n")
            for name, score in self:
                outf.write("%s\t%s\n"%(name, score))


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


#@jit(nopython=True)
#def entropy_ln(array):
#    """Method to calculate the entropy of any discrete numpy array
#
#    Args:
#        array (np.array): an array of discrete categories
#        uniquey : unique values in y
#
#    Returns:
#        entropy of array
#    """
#    entropy = 0
#    total = array.shape[0]
#    for val in np.unique(array):
#        num_this_class = np.sum(array == val)
#        p_i = num_this_class/total
#        if p_i == 0:
#            entropy += 0
#        else:
#            entropy += p_i*np.log(p_i)
#    return -entropy


def entropy(array, logfunc=np.log2):
    """Method to calculate the entropy of any discrete numpy array

    Args:
        array (np.array): an array of discrete categories
        uniquey : unique values in y
        logfunc : function
            defaults to np.log2, but sometimes you may want natural log (np.log)
    Returns:
        entropy of array
    """
    entropy = 0
    total = array.shape[0]
    for val in np.unique(array):
        num_this_class = np.sum(array == val)
        p_i = num_this_class/total
        if p_i == 0:
            entropy += 0
        else:
            entropy += p_i*logfunc(p_i)
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


def get_contingency_matrix(y_val_arr, hits_cats):

    y_classes, y_idx = np.unique(y_val_arr, return_inverse=True)
    hits_classes, hits_idx = np.unique(hits_cats, return_inverse=True)

    n_y_classes = y_classes.shape[0]
    n_hits_classes = hits_classes.shape[0]
    
    contingency = sparse.coo_matrix(
        (np.ones(y_idx.shape[0]), (y_idx, hits_idx)),
        shape=(n_y_classes, n_hits_classes),
        dtype=np.int64,
    )
    contingency = contingency.tocsr()
    contingency.sum_duplicates()
    return contingency.toarray()

def mutual_information_contingency(contingency):
    '''Calculates mutual information using contingency matrix

    Args:
    -----
    contingency: np.ndarray
        contingency matrix returned by get_contingency_matrix function
    '''

    # get non-zeros
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]
    log_contingency_nm = np.log(nz_val)

    contingency_sum = contingency.sum()
    contingency_nm = nz_val / contingency_sum

    p_i = np.ravel(contingency.sum(axis=1))
    p_j = np.ravel(contingency.sum(axis=0))

    # get outer product for non-zeros
    outer = (
        p_i.take(nzx).astype(np.int64, copy=False)
        * p_j.take(nzy).astype(np.int64, copy=False)
    )
    log_outer = -np.log(outer) + log(p_i.sum()) + log(p_j.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)
    

def adjusted_mutual_information(y_vals, hits):
    '''Calculated adjusted mutual information, which accounts for
    the effect that increasing the number of categories has on
    increasing mutual information between two vectors simply by
    random chance.

    Args:
    -----
    y_vals : np.array
    hits : np.array

    Returns:
    --------
    ami : float
        Adjusted mutual information, the max of which is 1.0
    '''

    distinct_hits,hits_cats = np.unique(hits, return_inverse=True, axis=0)
    distinct_y_vals = np.unique(y_vals)
    # special case where there is only one category in each vector,
    #  just return 0.0.
    if (
        distinct_y_vals.shape[0] == distinct_hits.shape[0] == 1
        or distinct_y_vals.shape[0] == distinct_hits.shape[0] == 0
    ):
        return 0.0
    contingency = get_contingency_matrix(y_vals, hits_cats)
    mi = mutual_information_contingency(contingency)
    expect_mi = exp_mi.expected_mutual_information(contingency, y_vals.shape[0])
    h_y, h_hits = entropy_ln(y_vals), entropy_ln(hits_cats)
    mean_h = np.mean([h_y, h_hits])
    denominator = mean_h - expect_mi
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        enominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - expect_mi) / denominator
    return ami

def conditional_adjusted_mutual_information(y_vals, hits_a, hits_b):
    """Method to calculate the conditional adjusted mutual information
        
    Args:
        y_vals (np.array): an array of discrete categories from the input data
        hits_a (np.array): number of hits in each record on each strand for motif a
        hits_b (np.array): number of hits in each record on each strand for motif b
    Returns:
        conditional adjusted mutual information
    """

    CMI = 0
    total = y_vals.shape[0]
    # CMI will look at each position of arr_x and arr_y that are of value z in arr_z
    distinct_b,b_cats = np.unique(hits_b, return_inverse=True, axis=0)
    distinct_b = np.unique(b_cats)
    for b in distinct_b:
        # set the indices we will look at in y_vals and hits_a
        subset = (b_cats == b)

        total_subset = np.sum(subset)
        p_z = total_subset/total

        y_cond_b = y_vals[subset]
        a_cond_b = hits_a[subset,:]

        ami_cond_b = adjusted_mutual_information(y_cond_b, a_cond_b)

        CMI += p_z*ami_cond_b

    return CMI


#@jit(nopython=True)
#def mutual_information(arrayx, arrayy, uniquey):
#    """Method to calculate the mutual information between two discrete
#    numpy arrays I(X;Y)
#        
#    Uses log2 so mutual information is in bits
#
#    Args:
#        arrayx (np.array): an array of discrete categories
#        arrayy (np.array): an array of discrete categories
#        uniquey (np.array): array of distinct values found in arrayy
#    Returns:
#        mutual information between array1 and array2
#    """
#
#    total_x = arrayx.size 
#    total_y = arrayy.size
#    total = total_x
#    MI = 0
#    for x in np.unique(arrayx):
#        # p(x_i)
#        row_is_x = arrayx == x
#        p_x = np.sum(row_is_x)/total
#        for y in uniquey:
#            # p(y_j)
#            row_is_y = (arrayy == y)[:,0]
#            p_y = np.sum(row_is_y)/total
#            # p(x_i,y_j)
#            p_x_y = np.sum(np.logical_and(row_is_x, row_is_y))/total
#            if p_x_y == 0 or p_x == 0 or p_y == 0:
#                MI += 0
#            else:
#                MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
#    return MI
#
#
#@jit(nopython=True)
#def conditional_mutual_information(arrayx, uniquex, arrayy, uniquey, arrayz, uniquez):
#    """Method to calculate the conditional mutual information I(X;Y|Z)
#        
#    Uses log2 so mutual information is in bits. This is O(X*Y*Z) where
#    X Y and Z are the number of unique categories in each array
#
#    Args:
#        arrayx (np.array): an array of discrete categories
#        arrayy (np.array): an array of discrete categories
#        arrayz (np.array): an array of discrete categories
#    Returns:
#        conditional mutual information arrayx;arrayy | arrayz
#    """
#
#    total_x = arrayx.size 
#    total_y = arrayy.size
#    total_z = arrayz.size
#    total = total_x
#    #if total_x != total_y or total_y != total_z:
#    #    raise ValueError(
#    #        "Array sizes must be the same {} {} {}".format(
#    #            total_x,
#    #            total_y,
#    #            total_z
#    #        )
#    #    )
#    #else:
#    #    total = total_x
#    CMI = 0
#    # CMI will look at each position of arr_x and arr_y that are of value z in arr_z
#    for z in uniquez:
#        # set the indices we will look at in arr_x and arr_y
#        subset = (arrayz == z)[:,0]
#        # set number of vals == z in arr_z as denominator
#        total_subset = np.sum(subset)
#        p_z = total_subset/total
#        this_MI = 0
#
#        for x in uniquex:
#            for y in uniquey:
#                # calculate the probability that the indices of arr_x and arr_y
#                #  corresponding to those in arr_z == z are equal to x or y.
#                # so essentially, in english, we're saying the following:
#                #  given that arr_z is what it is, what is the MI between
#                #  arr_x and arr_y?
#                row_is_y = (arrayy[subset] == y)[:,0]
#                row_is_x = arrayx[subset] == x
#                p_x = np.sum(row_is_x)/total_subset
#                p_y = np.sum(row_is_y)/total_subset
#                p_x_y = np.sum(
#                    np.logical_and(
#                        arrayx[subset] == x,
#                        row_is_y
#                    )
#                ) / total_subset
#                if p_x_y == 0 or p_x == 0 or p_y == 0:
#                    this_MI += 0
#                else:
#                    this_MI += p_x_y*np.log2(p_x_y/(p_x*p_y))
#
#        CMI += p_z*this_MI
#
#    return CMI

