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
import shutil
import json
from jinja2 import Environment,FileSystemLoader
import seaborn as sns
import pandas as pd
import base64
from matplotlib import pyplot as plt

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)

from convert_narrowpeak_to_fire import make_kfold_datasets
import inout

jinja_env = Environment(loader=FileSystemLoader(os.path.join(this_path, "templates/")))


class Performance():

    def __init__(self, fold_direcs):
        self.gather_performance_metrics(fold_direcs)

    def gather_performance_metrics(self, fold_direcs):

        self.fold_count = len(fold_direcs)
        self.fold_auprs = {}
        self.random_aupr = {}
        self.aupr = {}
        self.cv_aupr = {}
        self.cv_aupr_sd = {}
        randoms = {}
        self.any_motif = False
        for k,fold_direc in enumerate(fold_direcs):
            motifs_file = os.path.join(fold_direc, "final_motifs.dsm")
            if os.path.isfile(motifs_file):
                any_motif = True
                # place fold motifs into all_motifs
                aupr_fname = os.path.join(fold_direc, "precision_recall.json")
                with open(aupr_fname, "r") as aupr_file:
                    fold_data = json.load(aupr_file)
                for category,aupr_data in fold_data.items():
                    if not category in self.fold_auprs:
                        self.fold_auprs[category] = []
                        randoms[category] = []
                    self.fold_auprs[category].append(aupr_data["auc"])
                    randoms[category].append(aupr_data["random_auc"])
            else:
                continue

            

        joint_aupr_fname = os.path.join(fold_direc, "../precision_recall.json")
        with open(joint_aupr_fname, "r") as aupr_file:
            aupr_data = json.load(aupr_file)

        for category in self.fold_auprs.keys():
            self.random_aupr[category] = np.mean(randoms[category])
            self.cv_aupr[category] = np.mean(self.fold_auprs[category])
            self.cv_aupr_sd[category] = np.std(self.fold_auprs[category])
            self.aupr[category] = aupr_data[category]["auc"]

        if any_motif:
            self.fold_count_with_motifs = len(randoms[category])
        else:
            self.fold_count_with_motifs = 0


    def __str__(self):
        string = f"\nJoint AUPR:\n{self.aupr}\n"\
            f"Cross-validated AUPR:\n{self.cv_aupr}\n"\
            f"Cross-validated random AUPR expected:\n{self.random_aupr}\n"\
            f"Of {self.fold_count} folds used for cross-validation, "\
            f"{self.fold_count_with_motifs} had motifs.\n"
        return string

    def plot_performance(self, plot_fname):
        
        folds = []
        fold_categories = []
        fold_auprs = []

        cv_categories = []
        cv_auprs = []

        std_categories = []
        ranges = []

        joint_categories = []
        joint_auprs = []

        rand_yvals = []
        randoms = []
        rand_categories = []
        
        for category in self.aupr.keys():
            #cat = int(category)
            for k in range(self.fold_count):
                try:
                    folds.append(k)
                    fold_categories.append(category)
                    fold_auprs.append(self.fold_auprs[category][k])
                except IndexError:
                    continue
 
            rand_yvals.append(float(category)-0.3)
            rand_yvals.append(float(category)+0.3)
            randoms.append(self.random_aupr[category])
            randoms.append(self.random_aupr[category])
            rand_categories.append(category)
            rand_categories.append(category)

            cv_aupr = self.cv_aupr[category]
            stddev = self.cv_aupr_sd[category]
            lower = cv_aupr - stddev
            upper = cv_aupr + stddev

            std_categories.append(category)
            std_categories.append(category)
            ranges.append(lower)
            ranges.append(upper)

            cv_categories.append(category)
            cv_auprs.append(cv_aupr)

            joint_categories.append(category)
            joint_auprs.append(self.aupr[category])

        fold_df = pd.DataFrame(data={"fold":folds, "aupr":fold_auprs, "cat":fold_categories})
        print(f"fold_df:\n{fold_df}")
        lims_df = pd.DataFrame(data={"xvals":ranges, "cat":std_categories})
        print(f"lims_df:\n{lims_df}")
        cv_df = pd.DataFrame(data={"aupr":cv_auprs, "cat":cv_categories})
        print(f"cv_df:\n{cv_df}")
        joint_df = pd.DataFrame(data={"aupr":joint_auprs, "cat":joint_categories})
        print(f"joint_df:\n{joint_df}")
        rand_df = pd.DataFrame(data={"randoms":randoms, "yvals":rand_yvals, "cat":rand_categories})
        print(f"rand_df:\n{rand_df}")

        fig,ax = plt.subplots(figsize=(5,1.5*len(self.aupr)))
        sns.scatterplot(
            data = rand_df,
            x = "randoms",
            y = "cat",
            #hue = "cat",
            s = 200,
            marker = "|",
            color = "red",
            legend = False,
            ax = ax,
        )
        sns.lineplot(
            data = lims_df,
            x = "xvals",
            y = "cat",
            hue = "cat",
            legend = False,
            ax = ax,
        )
        sns.stripplot(
            data = fold_df,
            x = "aupr",
            y = "cat",
            hue = "cat", 
            jitter = 0.05,
            dodge = True,
            marker = ".",
            legend = False,
            ax = ax,
        )
        sns.scatterplot(
            data = cv_df,
            x = "aupr",
            y = "cat",
            hue = "cat", 
            marker = "o",
            legend = False,
            ax = ax,
        )
        sns.scatterplot(
            data = joint_df,
            x = "aupr",
            y = "cat",
            hue = "cat", 
            marker = "$\circ$",
            ec = "face",
            s = 100,
            legend = False,
            ax = ax,
        )
        plt.xlabel("AUPR")
        plt.ylabel("Category")
        plt.xlim((-0.05,1.05))
        plt.tight_layout()
        plt.savefig(plot_fname)


def write_report(environ, temp_base, info, out_name):
    print("writing report")
    print(f"base template: {temp_base}")
    template = environ.get_template(temp_base)
    print(f"out_name: {out_name}")
    content = template.render(**info)
    with open(out_name, "w", encoding="utf-8") as report:
        report.write(content)

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

    subparsers = parser.add_subparsers(help="commands", dest="command")

    # prep_data
    prep_data = subparsers.add_parser("prep_data", help=f"prepares fasta and scores file "\
        f"given a reference genome fasta file and a narrowpeak file identifying "\
        f"the 'positive' set in which to call enriched motifs.")
    prep_data.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    prep_data.add_argument('--narrowpeak_file', action="store", required=True,
        help=f"Base name of narrowpeak file containing the 'positive' regions "\
            f"in which to call enrichment motifs. "\
            f"Must reside in the location given by `--data_dir`.")
    prep_data.add_argument('--fasta_file', action="store", required=True,
        help=f"Base name of fasta file from which to extract positive sequences "\
            f"present in `--narrowpeak` file, in addition to randomly-selected negative sequences.")
    prep_data.add_argument('--wsize', type=int, default=60,
                        help="total window size around peak center")
    prep_data.add_argument('--nrand', type=int, default=3, 
            help="multiplier for number of random seqs to include")
    prep_data.add_argument('--seed', type=int, default=1234, 
            help="random seed for reproducibility")
    prep_data.add_argument('--rmchr', action="store_true", default=False, 
            help="rm chr string from peak chromosomes")
    prep_data.add_argument('--continuous', default=False, action="store_true",
            help="Include at command line to keep value field continuous")
    prep_data.add_argument('--center_metric', type=str, 
            help="geom or height, geom gives geometric center of the peak (default). \
                    height gives narrowpeak defined peak summit.")

    # infer
    infer = subparsers.add_parser("infer", help="run motif inference using an existing fasta file and scores file")

    infer.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    infer.add_argument('--skip_inference', action="store_true", default=False,
        help=f"Include this flag at the command line to skip motif inference. "\
            f"This is useful if you've already run inference on all folds.")
    infer.add_argument('--skip_evaluation', action="store_true", default=False,
        help=f"Include this flag at the command line to skip evaluation of motifs.")
    infer.add_argument('--force', action="store_true", default=False,
        help=f"Forces each fold to run, clobbering any extant output directories.")
    infer.add_argument('--crossval_folds', action="store", type=int, required=True,
        help="Number of folds into which to split data for k-fold cross-validation",
        default=5)
    infer.add_argument('--score_file', action='store', type=str, required=True,
        help='input text file with names and scores for training data')
    infer.add_argument('--kmer', type=int,
        help='kmer size to search for shape motifs. Default=%(default)d', default=15)
    infer.add_argument('--max_count', type=int, default=1,
        help=f"Maximum number of times a motif can match "\
            f"each of the forward and reverse strands in a reference. "\
            f"Default: %(default)d")
    infer.add_argument('--continuous', type=int, default=None,
        help="number of bins to discretize continuous input data with")
    infer.add_argument('--threshold_sd', type=float, default=2.0, 
        help=f"std deviations below mean for seed finding. "\
            f"Only matters for greedy search. Default=%(default)f")
    infer.add_argument('--init_threshold_seed_num', type=int, default=500, 
        help=f"Number of randomly selected seeds to compare to records "\
            f"in the database during initial threshold setting. Default=%(default)d")
    infer.add_argument('--init_threshold_recs_per_seed', type=int, default=20, 
        help=f"Number of randomly selected records to compare to each seed "\
            f"during initial threshold setting. Default=%(default)d")
    infer.add_argument('--init_threshold_windows_per_record', type=int, default=2, 
        help=f"Number of randomly selected windows within a given record "\
            f"to compare to each seed during initial threshold setting. "\
            f"Default=%(default)d")
    infer.add_argument("--max_batch_no_new_seed", type=int, default=10,
        help=f"Sets the number of batches of seed evaluation with no new motifs "\
            f"added to the set of motifs to be optimized prior to truncating the "\
            f"initial search for motifs.")
    infer.add_argument('--nprocs', type=int, default=1,
        help="number of processors. Default: %(default)d")
    infer.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help=f"Sets the upper and lower limits on the match "\
            f"threshold during optimization. Defaults to 0 for the "\
            f"lower limit and 10 for the upper limit.")
    infer.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help=f"Sets the upper and lower limits on the shapes' z-scores "\
            f"during optimization. Defaults to -4 for the lower limit "\
            f"and 4 for the upper limit.")
    infer.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, "\
            f"pre-normalized weights during optimization. Defaults to -4 "\
            f"for the lower limit and 4 for the upper limit.")
    infer.add_argument('--temperature', type=float, default=0.4,
        help=f"Sets the temperature argument for simulated annealing. "\
            f"Default: %(default)f")
    infer.add_argument('--t_adj', type=float, default=0.001,
        help=f"Fraction by which temperature decreases each iteration of "\
            f"simulated annealing. Default: %(default)f")
    infer.add_argument('--stepsize', type=float, default=0.25,
        help=f"Sets the stepsize argument simulated annealing. This "\
            f"defines how far a given value can be modified for iteration i "\
            f"from its value at iteration i-1. A higher value will "\
            f"allow farther hops. Default: %(default)f")
    infer.add_argument('--opt_niter', type=int, default=10000,
        help=f"Sets the number of simulated annealing iterations to "\
            f"undergo during optimization. Default: %(default)d.")
    infer.add_argument('--alpha', type=float, default=0.0,
        help=f"Lower limit on transformed weight values prior to "\
            f"normalization to sum to 1. Default: %(default)f")
    infer.add_argument('--batch_size', type=int, default=2000,
        help=f"Number of records to process seeds from at a time. Set lower "\
            f"to avoid out-of-memory errors. Default: %(default)d")
    infer.add_argument('--find_seq_motifs', action="store_true",
        help=f"Add this flag to call sequence motifs using streme in addition "\
            f"to calling shape motifs.")
    infer.add_argument("--no_shape_motifs", action="store_true",
        help=f"Add this flag to turn off shape motif inference. "\
            f"This is useful if you basically want to use this script "\
            f"as a wrapper for streme to just find sequence motifs.")
    infer.add_argument("--seq_fasta", type=str, default=None,
        help=f"Name of fasta file (located within data_dir, do not include the "\
            f"directory, just the file name) containing sequences in which to "\
            f"search for motifs")
    infer.add_argument('--seq_motif_positive_cats', required=False, default="1",
        action="store", type=str,
        help=f"Denotes which categories in `--infile` (or after quantization "\
            f"for a continous signal in the number of bins denoted by the "\
            f"`--continuous` argument) to use as the positive "\
            f"set for sequence motif calling using streme. Example: "\
            f"\"4\" would use category 4 as the positive set, whereas "\
            f"\"3,4\" would use categories 3 and 4 as "\
            f"the positive set.")
    infer.add_argument('--streme_thresh', default = 0.05,
        help="Threshold for including motifs identified by streme. Default: %(default)f")
    infer.add_argument("--seq_meme_file", type=str, default=None,
        help=f"Name of meme-formatted file (file must be located in data_dir) "\
            f"to be used for searching for known sequence motifs of interest in "\
            f"seq_fasta")
    infer.add_argument("--write_all_files", action="store_true",
        help=f"Add this flag to write all motif meme files, regardless of whether "\
            f"the model with shape motifs, sequence motifs, or both types of "\
            f"motifs was most performant.")
    infer.add_argument("--exhaustive", action="store_true", default=False,
        help=f"Add this flag to perform and exhaustive initial search for seeds. "\
            f"This can take a very long time for datasets with more than a few-thousand "\
            f"binding sites. Setting this option will override the "\
            f"--max_rounds_no_new_seed option.")
    infer.add_argument("--max_n", type=int, action="store", default=np.Inf,
        help=f"Sets the maximum number of fasta records to use for motif inference. "\
            f"This is useful when runs are taking prohibitively long.")
    infer.add_argument("--log_level", type=str, default="INFO",
        help=f"Sets log level for logging module. Valid values are DEBUG, "\
                f"INFO, WARNING, ERROR, CRITICAL.")

    args = parser.parse_args()
    return args


def quantize_yvals(y, nbins):
    quants = np.arange(0, 100, 100.0/nbins)
    values = y
    bins = []
    for quant in quants:
        bins.append(np.percentile(values, quant))
    logging.warning("Quantizing on bins: {}".format(bins))
    # subtract 1 to ensure categories start with 0
    y = np.digitize(values, bins) - 1
    return y


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
            logging.error(
                f"You included --no_shape_motifs without including --find_seq_motifs. "\
                f"No motifs will be found. Exiting now."
            )
            sys.exit(1)
    return outdir_pre


def infer(args):
    data_dir = args.data_dir
    shape_names = ["EP", "HelT", "MGW", "ProT", "Roll"]
    seq_fasta = os.path.join(data_dir, args.seq_fasta)
    kfold = args.crossval_folds
    find_seq_motifs = args.find_seq_motifs
    no_shape_motifs = args.no_shape_motifs
    in_fname = os.path.join(data_dir, args.score_file)
    motifs_rust_file = os.path.join(data_dir, "fold_motifs.json")
    status_fname = os.path.join(data_dir, "job_status.json")
    max_n = args.max_n

    status = "Running"
    with open(status_fname, "w") as status_f:
        json.dump(status, status_f)

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
    if args.continuous is not None:
        yvals = quantize_yvals(yvals, args.continuous)

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

    #############################################################
    #############################################################
    ## currently I do a bunch of redundant conversions, one set for each fold.
    ## in future versions, use this conversion in fold splitting.
    #############################################################
    #############################################################
    convert = f"Rscript {this_path}/utils/calc_shape.R {seq_fasta}"
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
    else:
        logging.info("Converting training sequences to shapes ran without error")
        full_shape_fnames = ""
        for shape_name in shape_names:
            full_shape_fnames += f"{seq_fasta}.{shape_name} "

    tmpdir = os.path.join(data_dir, "tmp")
    # if the output directory does not exist, make it
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)

    INFER_EXE = f"python {this_path}/infer_motifs.py "\
        f"--score_file {in_fname} "\
        f"--shape_files {full_shape_fnames} "\
        f"--shape_names {' '.join(shape_names)} "\
        f"--out_prefix {outdir_pre} "\
        f"--data_dir {data_dir} "\
        f"--out_dir {data_dir} "\
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
        f"--tmpdir {tmpdir} " \
        f"--log_level {args.log_level} "\
        f"--no_report"

    if args.exhaustive:
        INFER_EXE += f" --exhaustive"
    if args.write_all_files:
        INFER_EXE += f" --write_all_files"
    if args.no_shape_motifs:
        INFER_EXE += f" --no_shape_motifs"
    if args.continuous is not None:
        INFER_EXE += f" --continuous {args.continuous}"

    if find_seq_motifs:

        INFER_EXE += f" --seq_fasta {seq_fasta} " \
            f"--seq_motif_positive_cats {args.seq_motif_positive_cats} " \
            f"--streme_thresh {args.streme_thresh} " \
            f"--find_seq_motifs "

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
    else:
        logging.info(
            f"Motif inference was performed by running the following command:\n\n"\
            f"{INFER_CMD}\n\n"\
            f"resulting in the following stderr:\n\n"\
            f"{infer_result.stderr.decode()}\n\n"
            f"and the following stdout:\n\n"\
            f"{infer_result.stdout.decode()}"
        )
        # if no motifs in this fold, move on to next one
        if "No shape or sequence motifs found" in infer_result.stdout.decode():
            status = "FinishedWithMotifs"
            with open(status_fname, "w") as status_f:
                json.dump(status, status_f)
            logging.info("ShapeME finished")

    MERGE_EVAL_EXE = f"python {this_path}/evaluate_motifs.py "\
        f"--test_shape_files {full_shape_fnames} "\
        f"--shape_names {' '.join(shape_names)} "\
        f"--data_dir {data_dir} "\
        f"--test_score_file {in_fname} "\
        f"--out_dir {data_dir} "\
        f"--nprocs {args.nprocs} "\
        f"--out_prefix {outdir_pre}"

    if args.continuous is not None:
        MERGE_EVAL_EXE += f" --continuous {args.continuous}"

    if find_seq_motifs:
        MERGE_EVAL_EXE += f" --test_seq_fasta {seq_fasta} "

    # workaround for potential security vulnerability of shell=True
    MERGE_EVAL_CMD = shlex.quote(MERGE_EVAL_EXE)
    MERGE_EVAL_CMD = MERGE_EVAL_EXE
    merge_eval_result = subprocess.run(
        MERGE_EVAL_CMD,
        shell=True,
        capture_output=True,
        #check=True,
    )
    if merge_eval_result.returncode != 0:
        logging.error(
            f"ERROR: running the following command:\n\n"\
            f"{MERGE_EVAL_CMD}\n\n"\
            f"resulted in the following stderr:\n\n"\
            f"{merge_eval_result.stderr.decode()}\n\n"
            f"and the following stdout:\n\n"\
            f"{merge_eval_result.stdout.decode()}"
        )
        sys.exit(1)
    else:
        logging.info(
            f"Motif evaluation was performed by running the following command:\n\n"\
            f"{MERGE_EVAL_CMD}\n\n"\
            f"resulting in the following stderr:\n\n"\
            f"{merge_eval_result.stderr.decode()}\n\n"
            f"and the following stdout:\n\n"\
            f"{merge_eval_result.stdout.decode()}"
        )

    # get list of ((train_seq,train_y),(test_seq,test_y)) tuples for each fold

    folds = seqs.split_kfold( kfold, yvals )
    fold_direcs = []

    # write the data to files for each fold, run motif inference and evaluation
    # on each fold
    for k,fold in enumerate(folds):

        out_dir = os.path.join(data_dir, f"{outdir_pre}_fold_{k}_output")
        tmpdir = os.path.join(out_dir, "tmp")
        fold_direcs.append(out_dir)
        # if the output directory does not exist, make it
        if not os.path.exists(out_dir):
            #os.makedirs(out_dir)
            os.makedirs(tmpdir)
        # if the output directory does exist, exit by default, but allow
        #  to clobber if user provides --force at CLI
        else:
            if not args.skip_inference:
                if args.force:
                    # remove the current directory and all its contents
                    shutil.rmtree(out_dir)
                    # create empty directory
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

        train_seq_fasta = f"{data_dir}/{train_base}.fa"
        test_seq_fasta = f"{data_dir}/{test_base}.fa"

        train_score_fname = f"{data_dir}/{train_base}.txt"
        test_score_fname = f"{data_dir}/{test_base}.txt"

        train_shape_fnames = ""
        test_shape_fnames = ""
        for shape_name in shape_names:
            train_shape_fnames += f"{train_seq_fasta}.{shape_name} "
            test_shape_fnames += f"{test_seq_fasta}.{shape_name} "

        if not args.skip_inference:
            print("========================================")
            print(f"Writing fasta sequences for k-fold crossvalidation on fold {k}")
            print(f"Writing {train_seq_fasta}")
            with open(train_seq_fasta, "w") as train_seq_f:
                train_seqs.write(train_seq_f)

            print(f"Writing {train_score_fname}")
            with open(train_score_fname, "w") as train_score_f:
                train_score_f.write("name\tscore")
                for name,yval in zip(train_seqs.names,train_scores):
                    train_score_f.write(f"\n{name}\t{yval}")

            print(f"Writing {test_seq_fasta}")
            with open(test_seq_fasta, "w") as test_seq_f:
                test_seqs.write(test_seq_f)

            print(f"Writing {test_score_fname}")
            with open(test_score_fname, "w") as test_score_f:
                test_score_f.write("name\tscore")
                for name,yval in zip(test_seqs.names,test_scores):
                    test_score_f.write(f"\n{name}\t{yval}")

            convert = f"Rscript {this_path}/utils/calc_shape.R {train_seq_fasta}"
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
            else:
                logging.info("Converting training sequences to shapes ran without error")

            convert = f"Rscript {this_path}/utils/calc_shape.R {test_seq_fasta}"
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
            else:
                logging.info("Converting testing sequences to shapes ran without error")

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
                f"--tmpdir {tmpdir} " \
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
            f"--shape_names {' '.join(shape_names)} "\
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
                f"--find_seq_motifs "
            EVAL_EXE += f" --test_seq_fasta {test_seq_fasta} "\
                f"--train_seq_fasta {train_seq_fasta} "
                #f"--find_seq_motifs "

        if not args.skip_inference:
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
            else:
                logging.info(
                    f"Motif inference was performed by running the following command:\n\n"\
                    f"{INFER_CMD}\n\n"\
                    f"resulting in the following stderr:\n\n"\
                    f"{infer_result.stderr.decode()}\n\n"
                    f"and the following stdout:\n\n"\
                    f"{infer_result.stdout.decode()}"
                )
                # if no motifs in this fold, move on to next one
                if "No shape or sequence motifs found" in infer_result.stdout.decode():
                    continue


        if not args.skip_evaluation:
            logging.info(f"Evaluating motifs identified for fold {k}...")
            # workaround for potential security vulnerability of shell=True
            EVAL_CMD = shlex.quote(EVAL_EXE)
            EVAL_CMD = EVAL_EXE
            eval_result = subprocess.run(
                EVAL_CMD,
                shell=True,
                capture_output=True,
                #check=True,
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
            else:
                logging.info(
                    f"Motif evaluation was performed by running the following command:\n\n"\
                    f"{EVAL_CMD}\n\n"\
                    f"resulting in the following stderr:\n\n"\
                    f"{eval_result.stderr.decode()}\n\n"
                    f"and the following stdout:\n\n"\
                    f"{eval_result.stdout.decode()}"
                )

            #print(eval_result.stderr.decode())
            #print()
            #print(eval_result.stdout.decode())
            #sys.exit()

#    folds_with_motifs = []
#    for k,fold_direc in enumerate(fold_direcs):
#        motifs_file = os.path.join(fold_direc, "final_motifs.dsm")
#        if os.path.isfile(motifs_file):
#            # place fold motifs into all_motifs
#            folds_with_motifs.append((k, motifs_file))
#        else:
#            continue
#
    #print(f"fold_with_motifs: {folds_with_motifs}")
    ## if not motifs, write status file saying so and exit normally
    #if len(folds_with_motifs) == 0:
    #    status = "FinishedNoMotif"
    #    with open(status_fname, "w") as status_f:
    #        json.dump(status, status_f)
    #    logging.info("No motifs found in any folds")
    #    sys.exit(0)

    ## if only one fold has a result, copy it to the main directory
    #elif len(folds_with_motifs) == 1:
    #    src_file = folds_with_motifs[0][1]
    #    dsm_file = os.path.join(data_dir, "final_motifs.dsm")
    #    shutil.copyfile(src_file, dsm_file)
    #    ###########################################################
    #    ###########################################################
    #    ## need to also copy files with prec-rec curve, heatmap, etc
    #    ###########################################################
    #    ###########################################################

    #else:
    #    # for each motif name, append fold_n after its name
    #    for (k,fname) in folds_with_motifs:
    #        cmd = r"sed -i 's/^\(MOTIF.*\)/\1_fold_{}/g' {}".format(k,fname)
    #        subprocess.run(
    #            cmd,
    #            shell=True,
    #        )
    #    # first_file will just be copied to the main job directory
    #    first_file = folds_with_motifs.pop(0)
    #    first_fold_direc = os.path.dirname(first_file[1])
    #    dsm_files = [ fold[1] for fold in folds_with_motifs ]
    #    dsm_file = os.path.join(data_dir, "fold_motifs.dsm")
    #    # copy first dsm file to main data directory
    #    shutil.copyfile(first_file[1], dsm_file)
    #    # copy a config file to main direc for initializing merge_folds.py
    #    cfg_basename = "template_config.json"
    #    cfg_file = os.path.join(data_dir, cfg_basename)
    #    shutil.copyfile(os.path.join(first_fold_direc, "config.json"), cfg_file)
    #    # append each following dms file's motifs to the main dms file
    #    # this file will be read and split between seq/shape by merge_folds.py
    #    for fname in dsm_files:
    #        subprocess.run(
    #            f"sed -n -e '/MOTIF/,$p' {fname} >> {dsm_file}",
    #            shell=True,
    #        )
    #    MERGE_CMD = f"python {this_path}/merge_folds.py "\
    #        f"--config_file {cfg_basename} "\
    #        f"--seq_fasta {seq_fasta} "\
    #        f"--shape_files {full_shape_fnames} "\
    #        f"--shape_names {' '.join(shape_names)} "\
    #        f"--motifs_file {dsm_file} "\
    #        f"--direc {data_dir} "\
    #        f"--score_file {in_fname} "\
    #        f"--nprocs {args.nprocs} "\
    #        f"--tmpdir {tmpdir} "\
    #        f"--out_prefix {outdir_pre}"

    #    merge_result = subprocess.run(
    #        MERGE_CMD,
    #        shell=True,
    #        capture_output=True,
    #        #check=True,
    #    )
    #    if merge_result.returncode != 0:
    #        logging.error(
    #            f"ERROR: running the following command:\n\n"\
    #            f"{MERGE_CMD}\n\n"\
    #            f"resulted in the following stderr:\n\n"\
    #            f"{merge_result.stderr.decode()}\n\n"
    #            f"and the following stdout:\n\n"\
    #            f"{merge_result.stdout.decode()}"
    #        )
    #        sys.exit(1)
    #    else:
    #        logging.info(
    #            f"Merging motifs across folds was performed by running "\
    #            f"the following command:\n\n"\
    #            f"{MERGE_CMD}\n\n"\
    #            f"resulting in the following stderr:\n\n"\
    #            f"{merge_result.stderr.decode()}\n\n"
    #            f"and the following stdout:\n\n"\
    #            f"{merge_result.stdout.decode()}"
    #        )


    #    with open(status_fname, "r") as status_f:
    #        status = json.load(status_f)
    #    if status == "FinishedNoMotif":
    #        logging.info("No motifs after running merge_folds.py. Exiting.")
    #        sys.exit(0)

    #    logging.info(f"Evaluating set of motifs merged from folds...")
    #    MERGE_EVAL_EXE = f"python {this_path}/evaluate_motifs.py "\
    #        f"--test_shape_files {full_shape_fnames} "\
    #        f"--shape_names {' '.join(shape_names)} "\
    #        f"--data_dir {data_dir} "\
    #        f"--test_score_file {in_fname} "\
    #        f"--out_dir {data_dir} "\
    #        f"--nprocs {args.nprocs} "\
    #        f"--out_prefix {outdir_pre}"

    #    if args.continuous is not None:
    #        MERGE_EVAL_EXE += f" --continuous {args.continuous}"

    #    if find_seq_motifs:
    #        MERGE_EVAL_EXE += f" --test_seq_fasta {seq_fasta} "

    #    # workaround for potential security vulnerability of shell=True
    #    MERGE_EVAL_CMD = shlex.quote(MERGE_EVAL_EXE)
    #    MERGE_EVAL_CMD = MERGE_EVAL_EXE
    #    merge_eval_result = subprocess.run(
    #        MERGE_EVAL_CMD,
    #        shell=True,
    #        capture_output=True,
    #        #check=True,
    #    )
    #    if merge_eval_result.returncode != 0:
    #        logging.error(
    #            f"ERROR: running the following command:\n\n"\
    #            f"{MERGE_EVAL_CMD}\n\n"\
    #            f"resulted in the following stderr:\n\n"\
    #            f"{merge_eval_result.stderr.decode()}\n\n"
    #            f"and the following stdout:\n\n"\
    #            f"{merge_eval_result.stdout.decode()}"
    #        )
    #        sys.exit(1)
    #    else:
    #        logging.info(
    #            f"Motif evaluation was performed by running the following command:\n\n"\
    #            f"{MERGE_EVAL_CMD}\n\n"\
    #            f"resulting in the following stderr:\n\n"\
    #            f"{merge_eval_result.stderr.decode()}\n\n"
    #            f"and the following stdout:\n\n"\
    #            f"{merge_eval_result.stdout.decode()}"
    #        )

    ########################################################
    ## still need to gather final AUPR metrics for each fold, final motif, mean across folds
    ########################################################

    aupr_plot_fname = os.path.join(data_dir, "cv_aupr.png")
    performance = Performance(fold_direcs)
    performance.plot_performance(aupr_plot_fname)

    with open(aupr_plot_fname, "rb") as image_file:
        performance_plot = base64.b64encode(image_file.read()).decode()
    performance_data = {
        "plot": performance_plot,
        "folds_with_motifs": f"{performance.fold_count_with_motifs}/{performance.fold_count}"
    }
    
    report_data_fname = os.path.join(data_dir, "report_data.pkl")
    with open(report_data_fname, "rb") as info_f:
        report_info = pickle.load(info_f)
    report_info["performance_data"] = performance_data

    out_page_name = os.path.join(data_dir, "report.html")
    write_report(
        environ = jinja_env,
        temp_base = "shapeme_report.html.temp",
        info = report_info,
        out_name = out_page_name,
    )

    status = "FinishedWithMotifs"
    with open(status_fname, "w") as status_f:
        json.dump(status, status_f)
    logging.info("ShapeME finished")

  


def prep_data(args):
    data_dir = args.data_dir
    np_basename = args.narrowpeak_file
    fa_basename = args.fasta_file
    wsize = args.wsize
    nrand = args.nrand
    seed = args.seed
    rmchr = args.rmchr
    continuous = args.continuous
    center_metric = args.center_metric

    np_fname = os.path.join(data_dir, np_basename)
    fa_fname = os.path.join(data_dir, fa_basename)

    PREP_EXE = f"python {this_path}/convert_narrowpeak_to_fire.py "\
        f"{np_fname} {fa_fname} seqs "\
        f"--wsize {wsize} --nrand {nrand} --center_metric {center_metric}"

    # workaround for potential security vulnerability of shell=True
    PREP_CMD = shlex.quote(PREP_EXE)
    prep_result = subprocess.run(
        PREP_EXE,
        shell=True,
        capture_output=True,
        #check=True,
    )
    if prep_result.returncode != 0:
        logging.error(
            f"ERROR: running the following command:\n\n"\
            f"{PREP_CMD}\n\n"\
            f"resulted in the following stderr:\n\n"\
            f"{prep_result.stderr.decode()}\n\n"
            f"and the following stdout:\n\n"\
            f"{prep_result.stdout.decode()}"
        )
        sys.exit(1)
    else:
        logging.info(
            f"Data were prepared by running the following command:\n\n"\
            f"{PREP_CMD}\n\n"\
            f"resulting in the following stderr:\n\n"\
            f"{prep_result.stderr.decode()}\n\n"
            f"and the following stdout:\n\n"\
            f"{prep_result.stdout.decode()}"
        )


def main():

    args = parse_args()
    if args.command == "infer":
        infer(args)
    elif args.command == "prep_data":
        prep_data(args)


if __name__ == '__main__':
    main()

