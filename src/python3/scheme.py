#!/usr/bin/env python3

"""
The main driver script for k-fold cross-validated
motif inference.
"""

import subprocess
import argparse
import toml
import os
import sys
import numpy as np
from pathlib import Path

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)

from convert_narrowpeak_to_fire import make_kfold_datasets
import inout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crossval_fold', action="store", type=int, required=True,
        help="Number of folds into which to split data for k-fold cross-validation",
        default=5)
    parser.add_argument('--score_file', action='store', type=str, required=True,
        help='input text file with names and scores for training data')
    parser.add_argument('--shape_files', nargs="+", type=str, required=True,
        help='input files with shape scores')
    parser.add_argument('--shape_names', nargs="+", type=str, required=True,
        help='shape names (MUST BE IN SAME ORDER AS CORRESPONDING SHAPE FILES)')
    parser.add_argument('--out_prefix', type=str, required=True,
        help="Prefix to apply to output files.")
    parser.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    parser.add_argument('--kmer', type=int,
        help='kmer size to search for shape motifs. Default=%(default)d', default=15)
    parser.add_argument('--max_count', type=int, default=1,
        help=f"Maximum number of times a motif can match "\
            f"each of the forward and reverse strands in a reference. "\
            f"Default: %(default)d")
    parser.add_argument('--continuous', type=int, default=None,
        help="number of bins to discretize continuous input data with")
    parser.add_argument('--threshold_sd', type=float, default=2.0, 
        help=f"std deviations below mean for seed finding. "\
            f"Only matters for greedy search. Default=%(default)f")
    parser.add_argument('--init_threshold_seed_num', type=int, default=500, 
        help=f"Number of randomly selected seeds to compare to records "\
            f"in the database during initial threshold setting. Default=%(default)d")
    parser.add_argument('--init_threshold_recs_per_seed', type=int, default=20, 
        help=f"Number of randomly selected records to compare to each seed "\
            f"during initial threshold setting. Default=%(default)d")
    parser.add_argument('--init_threshold_windows_per_record', type=int, default=2, 
        help=f"Number of randomly selected windows within a given record "\
            f"to compare to each seed during initial threshold setting. "\
            f"Default=%(default)d")
    parser.add_argument("--max_batch_no_new_seed", type=int, default=10,
        help=f"Sets the number of batches of seed evaluation with no new motifs "\
            f"added to the set of motifs to be optimized prior to truncating the "\
            f"initial search for motifs.")
    parser.add_argument('-nprocs', type=int, default=1,
        help="number of processors. Default: %(default)d")
    parser.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help=f"Sets the upper and lower limits on the match "\
            f"threshold during optimization. Defaults to 0 for the "\
            f"lower limit and 10 for the upper limit.")
    parser.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help=f"Sets the upper and lower limits on the shapes' z-scores "\
            f"during optimization. Defaults to -4 for the lower limit "\
            f"and 4 for the upper limit.")
    parser.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, "\
            f"pre-normalized weights during optimization. Defaults to -4 "\
            f"for the lower limit and 4 for the upper limit.")
    parser.add_argument('--temperature', type=float, default=0.4,
        help=f"Sets the temperature argument for simulated annealing. "\
            f"Default: %(default)f")
    parser.add_argument('--t_adj', type=float, default=0.001,
        help=f"Fraction by which temperature decreases each iteration of "\
            f"simulated annealing. Default: %(default)f")
    parser.add_argument('--stepsize', type=float, default=0.25,
        help=f"Sets the stepsize argument simulated annealing. This "\
            f"defines how far a given value can be modified for iteration i "\
            f"from its value at iteration i-1. A higher value will "\
            f"allow farther hops. Default: %(default)f")
    parser.add_argument('--opt_niter', type=int, default=10000,
        help=f"Sets the number of simulated annealing iterations to "\
            f"undergo during optimization. Default: %(default)d.")
    parser.add_argument('--alpha', type=float, default=0.0,
        help=f"Lower limit on transformed weight values prior to "\
            f"normalization to sum to 1. Default: %(default)f")
    parser.add_argument('--batch_size', type=int, default=2000,
        help=f"Number of records to process seeds from at a time. Set lower "\
            f"to avoid out-of-memory errors. Default: %(default)d")
    parser.add_argument('--find_seq_motifs', action="store_true",
        help=f"Add this flag to call sequence motifs using streme in addition "\
            f"to calling shape motifs.")
    parser.add_argument("--no_shape_motifs", action="store_true",
        help=f"Add this flag to turn off shape motif inference. "\
            f"This is useful if you basically want to use this script "\
            f"as a wrapper for streme to just find sequence motifs.")
    parser.add_argument("--seq_fasta", type=str, default=None,
        help=f"Name of fasta file (located within in_direc, do not include the "\
            f"directory, just the file name) containing sequences in which to "\
            f"search for motifs")
    parser.add_argument('--seq_motif_positive_cats', required=False, default="1",
        action="store", type=str,
        help=f"Denotes which categories in `--infile` (or after quantization "\
            f"for a continous signal in the number of bins denoted by the "\
            f"`--continuous` argument) to use as the positive "\
            f"set for sequence motif calling using streme. Example: "\
            f"\"4\" would use category 4 as the positive set, whereas "\
            f"\"3,4\" would use categories 3 and 4 as "\
            f"the positive set.")
    parser.add_argument('--streme_thresh', default = 0.05,
        help="Threshold for including motifs identified by streme. Default: %(default)f")
    parser.add_argument("--seq_meme_file", type=str, default=None,
        help=f"Name of meme-formatted file (file must be located in data_dir) "\
            f"to be used for searching for known sequence motifs of interest in "\
            f"seq_fasta")
    parser.add_argument("--shape_rust_file", type=str, default=None,
        help=f"Name of json file containing output from rust binary")
    parser.add_argument("--write_all_files", action="store_true",
        help=f"Add this flag to write all motif meme files, regardless of whether "\
            f"the model with shape motifs, sequence motifs, or both types of "\
            f"motifs was most performant.")
    parser.add_argument("--exhaustive", action="store_true", default=False,
        help=f"Add this flag to perform and exhaustive initial search for seeds. "\
            f"This can take a very long time for datasets with more than a few-thousand "\
            f"binding sites. Setting this option will override the "\
            f"--max_rounds_no_new_seed option.")
    parser.add_argument("--max_n", type=int, action="store", default=np.Inf,
        help=f"Sets the maximum number of fasta records to use for motif inference. "\
            f"This is useful when runs are taking prohibitively long.")
    parser.add_argument("--log_level", type=str, default="INFO",
        help=f"Sets log level for logging module. Valid values are DEBUG, "\
                f"INFO, WARNING, ERROR, CRITICAL.")

    data_dir = args.data_dir
    kfold = args.crossval_folds
    find_seq_motifs = args.find_seq_motifs
    no_shape_motifs = args.no_shape_motifs
    in_fname = os.path.join(in_direc, args.score_file)

    # assemble the prefix for output direc name
    outdir_pre = ""
    if not no_shape_motifs:
        outdir_pre += "shape"
        if find_seq_motifs:
            outdir_pre += "_and_seq"
    else:
        if find_seq_motifs:
            outdir_pre += "seq"
        else:
            sys.exit(
                f"You included --no_shape_motifs without including --find_seq_motifs. "\
                f"No motifs will be found. Exiting now."
            )

    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(args.shape_names, args.shape_files)
    }
    records = inout.RecordDatabase(
        in_fname,
        shape_fname_dict,
        shift_params = ["Roll", "HelT"],
    )

    # make k-fold data
    for fold in range(kfold):
        out_dir = os.path.join(data_dir, f"{outdir_pre}_fold_{fold}_output")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            sys.exit(
                f"The intended output directory, {out_dir}, already "\
                f"exists. Either rename the existing directory or remove it. "\
                f"Nothing was done for fold {fold}. Exiting now."
            )

        make_kfold_datasets(kfold, outfasta, finalfire, args.outpre)



if __name__ == '__main__':
    main()


