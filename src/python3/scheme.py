#!/usr/bin/env python3

"""
The main driver script for k-fold cross-validated
motif inference.
"""

import subprocess
import argparse
import os
import sys
import numpy as np
from pathlib import Path
import logging
import shlex

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)

from convert_narrowpeak_to_fire import make_kfold_datasets
import inout

def read_score_file(infile):
    """Read a score file 

    Args:
    -----
        infile : str
            input data file name

    Returns:
    --------
        y : 1D np array
            value for each sequence
    """
    scores = []
    with open(infile) as inf:
        line = inf.readline()
        for i,line in enumerate(inf):
            linearr = line.rstrip().split("\t")
            scores.append(float(linearr[1]))
        y = np.asarray(
            scores,
            dtype=float,
        )
    return y

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crossval_folds', action="store", type=int, required=True,
        help="Number of folds into which to split data for k-fold cross-validation",
        default=5)
    parser.add_argument('--score_file', action='store', type=str, required=True,
        help='input text file with names and scores for training data')
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
    parser.add_argument('--nprocs', type=int, default=1,
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
        help=f"Name of fasta file (located within data_dir, do not include the "\
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

    args = parser.parse_args()
    return args

def set_outdir_pref(no_shape_motifs, find_seq_motifs):
    """Assemble output directory name prefix
    """
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
    return outdir_pre


def main():

    args = parse_args()

    data_dir = args.data_dir
    shape_names = ["EP", "HelT", "MGW", "ProT", "Roll"]
    seq_fasta = os.path.join(data_dir, args.seq_fasta)
    kfold = args.crossval_folds
    find_seq_motifs = args.find_seq_motifs
    no_shape_motifs = args.no_shape_motifs
    in_fname = os.path.join(data_dir, args.score_file)
    max_n = args.max_n

    loglevel = args.log_level
    numeric_level = getattr(logging, loglevel.upper(), None)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=numeric_level,
        stream=sys.stdout,
    )

    # assemble the prefix for output direc name
    outdir_pre = set_outdir_pref(no_shape_motifs, find_seq_motifs)
    with open(seq_fasta,"r") as seq_f:
        seqs = inout.FastaFile()
        seqs.read_whole_file(seq_f)

    yvals = read_score_file(in_fname)

    # down-sample number of records if that's what we've chosen to do
    if max_n < len(seqs):
        retained_indices,seqs,yvals = seqs.sample(max_n, yvals)
        seq_fasta = os.path.splitext(seq_fasta)[0] + f"_downsampled.fa"
        in_fname = os.path.splitext(in_fname)[0] + f"_downsampled.txt"
        seq_names = seqs.names
        # write files with down-sampled records
        with open(seq_fasta, "w") as of:
            seqs.write(of)
        with open(in_fname, "w") as of:
            of.write("name\tscore")
            for name,yval in zip(seq_names,yvals):
                of.write(f"\n{name}\t{yval}")

    # get list of ((train_seq,train_y),(test_seq,test_y)) tuples for each fold
    folds = seqs.split_kfold( kfold, yvals )

    # write the data to files for each fold, run motif inference and evaluation
    # on each fold
    for k,fold in enumerate(folds):

        out_dir = os.path.join(data_dir, f"{outdir_pre}_fold_{k}_output")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            logging.error(
                f"The intended output directory, {out_dir}, already "\
                f"exists. We try not to clobber existing data. "\
                f"Either rename the existing directory or remove it. "\
                f"Nothing was done for fold {k}. Exiting now."
            )
            sys.exit(1)

        train_base = f"fold_{k}_train"
        test_base = f"fold_{k}_test"
        train_seqs = fold[0][0]
        train_scores = fold[0][1]
        test_seqs = fold[1][0]
        test_scores = fold[0][1]

        train_seq_fasta = f"{train_base}.fa"
        test_seq_fasta = f"{test_base}.fa"

        train_score_fname = f"{train_base}.txt"
        test_score_fname = f"{test_base}.txt"

        train_shape_fnames = ""
        test_shape_fnames = ""
        for shape_name in shape_names:
            train_shape_fnames += f"{train_seq_fasta}.{shape_name} "
            test_shape_fnames += f"{test_seq_fasta}.{shape_name} "

        with open(train_seq_fasta, "w") as train_seq_f:
            train_seqs.write(train_seq_f)

        with open(train_score_fname, "w") as train_score_f:
            train_score_f.write("name\tscore")
            for name,yval in zip(train_seqs.names,train_scores):
                train_score_f.write(f"\n{name}\t{yval}")

        with open(test_seq_fasta, "w") as test_seq_f:
            test_seqs.write(test_seq_f)

        with open(test_score_fname, "w") as test_score_f:
            test_score_f.write("name\tscore")
            for name,yval in zip(test_seqs.names,test_scores):
                test_score_f.write(f"\n{name}\t{yval}")

        convert = f"Rscript {this_path}/utils/calc_shape.R {data_dir}/{train_seq_fasta}"
        #convert = shlex.quote(convert)
        convert_result = subprocess.run(
            convert,
            shell=True,
            capture_output=True,
            #check=True,
        )
        if convert_result.returncode != 0:
            logging.error(
                f"ERROR: running the following command:\n\n"\
                f"{convert}\n\n"\
                f"resulted in the following stderr:\n\n"\
                f"{convert_result.stderr.decode()}\n\n"
                f"and the following stdout:\n\n"\
                f"{convert_result.stdout.decode()}"
            )
            sys.exit(1)

        convert = f"Rscript {this_path}/utils/calc_shape.R {data_dir}/{test_seq_fasta}"
        #convert = shlex.quote(convert)
        convert_result = subprocess.run(
            convert,
            shell=True,
            capture_output=True,
            #check=True,
        )
        if convert_result.returncode != 0:
            logging.error(
                f"ERROR: running the following command:\n\n"\
                f"{convert}\n\n"\
                f"resulted in the following stderr:\n\n"\
                f"{convert_result.stderr.decode()}\n\n"
                f"and the following stdout:\n\n"\
                f"{convert_result.stdout.decode()}"
            )
            sys.exit(1)

        INFER_EXE = f"python {this_path}/infer_motifs.py "\
            f"--score_file fold_{k}_train.txt "\
            f"--shape_files {train_shape_fnames} "\
            f"--shape_names {' '.join(shape_names)} "\
            f"--out_prefix {outdir_pre} "\
            f"--data_dir {data_dir} "\
            f"--out_dir {out_dir} "\
            f"--kmer {args.kmer} " \
            f"--max_count {args.max_count} "\
            f"--threshold_sd {args.threshold_sd} "\
            f"--init_threshold_seed_num {args.init_threshold_seed_num} "\
            f"--init_threshold_recs_per_seed {args.init_threshold_recs_per_seed} "\
            f"--init_threshold_windows_per_record {args.init_threshold_windows_per_record} "\
            f"--max_batch_no_new_seed {args.max_batch_no_new_seed} "\
            f"--nprocs {args.nprocs} "\
            f"--threshold_constraints "\
                f"{args.threshold_constraints[0]} {args.threshold_constraints[1]} " \
            f"--shape_constraints " \
                f"{args.shape_constraints[0]} {args.shape_constraints[1]} " \
            f"--weights_constraints " \
                f"{args.weights_constraints[0]} {args.weights_constraints[1]} " \
            f"--temperature {args.temperature} " \
            f"--t_adj {args.t_adj} " \
            f"--stepsize {args.stepsize} " \
            f"--opt_niter {args.opt_niter} " \
            f"--alpha {args.alpha} " \
            f"--batch_size {args.batch_size} " \
            f"--shape_rust_file {args.shape_rust_file} " \
            f"--log_level {args.log_level}"

        if args.exhaustive:
            INFER_EXE += f" --exhaustive"
        if args.write_all_files:
            INFER_EXE += f" --write_all_files"
        if args.no_shape_motifs:
            INFER_EXE += f" --no_shape_motifs"

        EVAL_EXE = f"python {this_path}/evaluate_motifs.py "\
            f"--test_shape_files {test_shape_fnames} "\
            f"--train_shape_files {train_shape_fnames} "\
            f"--shape_names {shape_names} "\
            f"--data_dir {data_dir} "\
            f"--train_score_file {train_score_fname} "\
            f"--test_score_file {test_score_fname} "\
            f"--out_dir {out_dir} "\
            f"--nprocs {args.nprocs} "\
            f"--out_prefix {outdir_pre}"

        if args.continuous is not None:
            INFER_EXE += f" --continuous {args.continuous}"
            EVAL_EXE += f" --continuous {args.continuous}"

        if find_seq_motifs:

            INFER_EXE += f" --seq_fasta {train_seq_fasta} " \
                f"--seq_motif_positive_cats {args.seq_motif_positive_cats} " \
                f"--streme_thresh {args.streme_thresh} " \
                f"--find_seq_motifs {args.find_seq_motifs} "
            EVAL_EXE += f" --test_seq_fasta {test_seq_fasta} "\
                f"--train_seq_fasta {train_seq_fasta} "\
                f"--find_seq_motifs {args.find_seq_motifs} "

        logging.info(f"Inferring motifs for fold {k}...")
        # workaround for potential security vulnerability of shell=True
        INFER_CMD = shlex.quote(INFER_EXE)
        INFER_CMD = INFER_EXE
        infer_result = subprocess.run(
            INFER_CMD,
            shell=True,
            capture_output=True,
            #check=True,
        )
        if infer_result.returncode != 0:
            logging.error(
                f"ERROR: running the following command:\n\n"\
                f"{INFER_CMD}\n\n"\
                f"resulted in the following stderr:\n\n"\
                f"{infer_result.stderr.decode()}\n\n"
                f"and the following stdout:\n\n"\
                f"{infer_result.stdout.decode()}"
            )
            sys.exit(1)

        logging.info(f"Evaluating motifs identified for fold {k}...")
        # workaround for potential security vulnerability of shell=True
        EVAL_CMD = shlex.quote(EVAL_EXE)
        EVAL_CMD = EVAL_EXE
        eval_result = subprocess.run(
            EVAL_CMD,
            shell=True,
            #capture_output=True,
            check=True,
        )
        if eval_result.returncode != 0:
            logging.error(
                f"ERROR: running the following command:\n\n"\
                f"{EVAL_CMD}\n\n"\
                f"resulted in the following stderr:\n\n"\
                f"{eval_result.stderr.decode()}\n\n"
                f"and the following stdout:\n\n"\
                f"{eval_result.stdout.decode()}"
            )
            sys.exit(1)


    

if __name__ == '__main__':
    main()


