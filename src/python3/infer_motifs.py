import inout
import base64
import sys
import os
import logging
import argparse
import numpy as np
import shapemotifvis as smv
import json
import pickle
import time
import subprocess
import multiprocessing
import tempfile
from jinja2 import Environment,FileSystemLoader
from pathlib import Path

import evaluate_motifs as evm
import fimopytools as fimo

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)
infer_bin = os.path.join(this_path, '../rust_utils/target/release/infer_motifs')
supp_bin = os.path.join(this_path, '../rust_utils/target/release/get_robustness')
cmi_bin = os.path.join(this_path, '../rust_utils/target/release/filter_motifs')

jinja_env = Environment(loader=FileSystemLoader(os.path.join(this_path, "templates/")))

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# import R's "PRROC" package
prroc = importr('PRROC')
glmnet = importr('glmnet')
base = importr('base')

def write_report(environ, temp_base, info, out_name):
    print("writing report")
    print(f"base template: {temp_base}")
    template = environ.get_template(temp_base)
    print(f"out_name: {out_name}")
    content = template.render(**info)
    with open(out_name, "w", encoding="utf-8") as report:
        report.write(content)


def two_way_to_log_odds(two_way):
    """ Function to determine the log odds from a two way table

    Args:
        two_way (list)- a list of 0-cat1Truecat2True 1-cat1Falsecat2True 
                                  2-cat1Truecat2False 3-cat1Falsecat2False
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    num = np.array(two_way[0], dtype=float) / np.array(two_way[1],dtype=float)
    denom = np.array(two_way[2], dtype=float) / np.array(two_way[3],dtype=float)
    return np.log(num/denom)

def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--out_dir', type=str, required=True,
        help="Directory (within 'data_dir') into which output files will be written.")
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
    #parser.add_argument('--nonormalize', action="store_true",
    #    help='don\'t normalize the input data by robustZ')
    #parser.add_argument('--motif_perc', type=float, default=1,
    #    help="fraction of data to EVALUATE motifs on. Default=%(default)f")
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
            f"for a continuos signal in the number of bins denoted by the "\
            f"`--continuous` argument) to use as the positive "\
            f"set for sequence motif calling using streme. Example: "\
            f"\"4\" would use category 4 as the positive set, whereas "\
            f"\"3,4\" would use categories 3 and 4 as "\
            f"the positive set.")
    parser.add_argument('--streme_thresh', default = 0.05, type=float,
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
    parser.add_argument("--log_level", type=str, default="INFO",
        help=f"Sets log level for logging module. Valid values are DEBUG, "\
                f"INFO, WARNING, ERROR, CRITICAL.")
    parser.add_argument('--tmpdir', action='store', type=str, default=None,
        help=f"Sets the location into which to write temporary files. If ommitted, will "\
                f"use TMPDIR environment variable.")
    parser.add_argument('--no_report', action="store_true", default=False,
        help="Include at command line to suppress writing an html report of results. Note: a report will still be written in the case on an error.")
    args = parser.parse_args()
    return args


def main(args, status):

    my_env = os.environ.copy()
    my_env['RUST_BACKTRACE'] = "1"

    loglevel = args.log_level
    numeric_level = getattr(logging, loglevel.upper(), None)

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=numeric_level,
        stream=sys.stdout,
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.debug(f"Number of cores set by the --nprocs argument: {args.nprocs}")
    logging.debug(
        f"Number of cores available: {multiprocessing.cpu_count()}"
    )

    logging.info("Arguments:")
    print(str(args))

    out_pref = args.out_prefix
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)
    in_fname = args.score_file
    shape_names = args.shape_names
    shape_files = args.shape_files
    out_motif_basename = os.path.join(out_direc, "final_motifs")
    out_motif_fname = out_motif_basename + ".dsm"
    out_coefs_fname = out_motif_basename + "_coefficients.pkl"
    out_heatmap_fname = os.path.join(out_direc, "final_heatmap.png")
    out_page_name = os.path.join(out_direc, "report.html")
    status_fname = os.path.join(out_direc, "job_status.json")
    find_seq_motifs = args.find_seq_motifs
    seq_fasta = args.seq_fasta
    if seq_fasta is not None:
        seq_fasta = os.path.join(in_direc, seq_fasta)
    seq_meme_file = args.seq_meme_file
    if seq_meme_file is not None:
        seq_meme_file = os.path.join(in_direc, seq_meme_file)
    fimo_direc = f"{out_direc}/fimo_out"
    streme_direc = f"{out_direc}/streme_out"
    streme_thresh = float(args.streme_thresh)
    no_shape_motifs = args.no_shape_motifs
    tmpdir = args.tmpdir
    if tmpdir is None:
        tmpdir = "/tmp"

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    if tmpdir is not None:
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)

    if os.path.isfile(status_fname):
        with open(status_fname, "r") as status_f:
            status = json.load(status_f)
        
    status = "Running"

    with open(status_fname, "w") as status_f:
        json.dump(status, status_f)

    print()
    logging.info("Reading input data and shape info.")
    # read in shapes
    records = inout.construct_records(
        in_direc,
        shape_names,
        shape_files,
        in_fname,
    )
    logging.info("Finished reading input data and shape info.")

    assert len(records.y) == records.X.shape[0], "Number of y values does not equal number of shape records!!"
           
    # read in the values associated with each sequence and store them
    # in the sequence database
    ints = np.ones_like(records.y, dtype="bool")
    if args.continuous is not None:
        for i,val in enumerate(records.y):
            ints[i] = val.is_integer()
        all_int = np.all(ints)
        if all_int:
            logging.warning("WARNING: You have included the --continuous flag at the command line despite having input scores comprising entirely integers. Double-check whether this is really what you want before interpreting ShapeME resulgs.")
        records.quantize_quant(args.continuous)

    for category in np.unique(records.y):
        if not category.is_integer():
            sys.exit(f"ERROR: At least one category in your input data is not an integer. Non-integer inputs can be used only if you set the --continuous <int> flag, substituting the desired number of bins into which to discretize your input data for '<int>'. Did you intend to use the --continuous flag, or did you intend to discretize your data prior to using the scores and inputs? If so, go back and do so. As things stand currently, however, we cannot work with the input data as is. Exiting now with an error.")

    fam,num_cats = evm.set_family(records.y)
    records.set_category_lut()

    logging.info("Distribution of sequences per class:")
    seq_bin_str = records.seqs_per_bin()
    logging.info(seq_bin_str)

    logging.info("Normalizing parameters")
    #if args.nonormalize:
    #    records.determine_center_spread(method=inout.identity_csp)
    #else:
    records.determine_center_spread()
    records.normalize_shape_values()

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        logging.info(f"{name}: center={this_center:.2f}, spread={this_spread:.2f}")

    yval_fname = os.path.join(out_direc, 'y_vals.npy')
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, records.y.astype(np.int64))

    if find_seq_motifs:
        # if asked for seq motifs but didn't pass seq fa file, exception
        if seq_fasta is None:
            raise inout.NoSeqFaException()
        # if both seq_motifs and meme file were passed, raise exception
        if seq_meme_file is not None:
            raise inout.SeqMotifOptionException(seq_meme_file)

        seq_meme_file = f"{streme_direc}/streme.txt"
        streme_exec = os.path.join(this_path, "run_streme.py")

        # NA-containing records were removed, so use retained_records to get right
        # sequences if calling sequence motifs
        seqs = inout.FastaFile()
        with open(seq_fasta,"r") as seq_f:
            seqs.read_whole_file(seq_f)

        seqs = seqs[np.where(records.complete_records)[0]]
        tmp_dir = tempfile.TemporaryDirectory(dir=tmpdir)
        tmp_direc = tmp_dir.name
        tmp_seq_fname = os.path.join(tmp_direc,"tmp_seq.fa")
        #print(f"tmp_seq_fname: {tmp_seq_fname}")
        with open(tmp_seq_fname, "w") as tmp_f:
            seqs.write(tmp_f)

        #tmp_seq_fname = os.path.join(out_direc,"tmp_seq.fa")
        #with open(tmp_seq_fname, "w") as tmp_f:
        #    seqs.write(tmp_f)
        streme_log_fname = f"{streme_direc}/streme_run.log"
        streme_err_fname = f"{streme_direc}/streme_run.err"

        if not os.path.isdir(streme_direc):
            os.makedirs(streme_direc)

        STREME = f"python {streme_exec} "\
            f"--seq_fname {tmp_seq_fname} "\
            f"--yvals_fname {yval_fname} "\
            f"--pos_cats {args.seq_motif_positive_cats} "\
            f"--threshold {streme_thresh} "\
            f"--out_direc {streme_direc} "\
            f"--tmpdir {tmpdir} "\
            f"> {streme_log_fname} "\
            f"2> {streme_err_fname} "

        print()
        logging.info(
            f"Running STREME using the following command:\n"\
            f"{STREME}"
        )
        streme_result = subprocess.run(
            STREME,
            shell=True,
            capture_output=False,
        )
        if streme_result.returncode != 0:

            status = "FinishedError"
            with open(status_fname, "w") as status_f:
                json.dump(status, status_f)

            raise(Exception(
                f"run_streme.py returned non-zero exit status.\n"\
                f"Check files {streme_log_fname} and {streme_err_fname}."
            ))

        print()
        logging.info(
            f"STREME run finished: for details, see "\
            f"{streme_log_fname} and {streme_err_fname}"
        )

        #with open(streme_log_fname, "w") as streme_out:
        #    # streme log gets captured as stderr, so write stderr to file
        #    streme_out.write(streme_result.stdout.decode())
        #with open(streme_err_fname, "w") as streme_err:
        #    # streme log gets captured as stderr, so write stderr to file
        #    try:
        #        streme_err.write(streme_result.stderr.decode())
        #    except UnicodeDecodeError as e:
        #        logging.warning(f"Problem writing to {streme_err_fname}:\n{e}")
        #        #import ipdb; ipdb.set_trace()

        #        if not args.no_report:
        #            report_info = {"error": e}
        #            write_report(
        #                environ = jinja_env,
        #                temp_base = "streme_err_html.temp",
        #                info = report_info,
        #                out_name = out_page_name,
        #            )

        #        status = "FinishedError"
        #        with open(status_fname, "w") as status_f:
        #            json.dump(status, status_f)
        #        sys.exit(1)

    # if user has a meme file (could be from streme above, or from input arg), run fimo
    if seq_meme_file is not None:

        if seq_fasta is None:
            raise inout.NoSeqFaException()

        # NA-containing records were removed, so use retained_records to get right
        # sequences if calling sequence motifs
        seqs = inout.FastaFile()
        with open(seq_fasta,"r") as seq_f:
            seqs.read_whole_file(seq_f)

        seqs = seqs[np.where(records.complete_records)[0]]
        tmp_dir = tempfile.TemporaryDirectory(dir=tmpdir)
        tmp_direc = tmp_dir.name
        tmp_seq_fname = os.path.join(tmp_direc,"tmp_seq.fa")
        #print(f"tmp_seq_fname: {tmp_seq_fname}")
        with open(tmp_seq_fname, "w") as tmp_f:
            seqs.write(tmp_f)

        fimo_log_fname = f"{fimo_direc}/fimo_run.log"
        fimo_err_fname = f"{fimo_direc}/fimo_run.err"

        if not os.path.isdir(fimo_direc):
            os.makedirs(fimo_direc)

        fimo_exec = os.path.join(this_path, "run_fimo.py")
        FIMO = f"python {fimo_exec} "\
            f"--seq_fname {tmp_seq_fname} "\
            f"--meme_file {seq_meme_file} "\
            f"--out_direc {fimo_direc} "\
            f"> {fimo_log_fname} "\
            f"2> {fimo_err_fname} "

        print()
        logging.info(
            f"Running FIMO using the following command:\n"\
            f"{FIMO}"
        )
        fimo_result = subprocess.run(
            FIMO,
            shell=True,
            check=True,
            capture_output=False,
        )

        if fimo_result.returncode != 0:
            status = "FinishedError"
            with open(status_fname, "w") as status_f:
                json.dump(status, status_f)
            raise(Exception(
                f"run_fimo.py returned non-zero exit status.\n"\
                f"Check files {fimo_log_fname} and {fimo_err_fname}."
            ))

        print()
        logging.info(
            f"FIMO run finished: for details, see "\
            f"{fimo_log_fname} and {fimo_err_fname}"
        )
        #with open(fimo_log_fname, "w") as fimo_out:
        #    fimo_out.write(fimo_result.stdout.decode())
        #with open(fimo_err_fname, "w") as fimo_err:
        #    fimo_err.write(fimo_result.stderr.decode())

    alpha = args.alpha
    max_count = args.max_count

    temp = args.temperature
    step = args.stepsize
    
    mi_fname = os.path.join(
        out_direc,
        f'{out_pref}_initial_mutual_information_max_count_{max_count}.pkl'
    )

    shape_fname = os.path.join(out_direc, 'shapes.npy')
    config_fname = os.path.join(out_direc, 'config.json')
    rust_out_fname = os.path.join(out_direc, 'rust_results.json')
    shape_fit_fname = os.path.join(out_direc, 'shape_lasso_fit.pkl')
    seq_fit_fname = os.path.join(out_direc, 'seq_lasso_fit.pkl')
    shape_and_seq_fit_fname = os.path.join(out_direc, 'shape_and_seq_lasso_fit.pkl')

    # get the F-score for an intercept-only model, which will ultimately be compared
    # to the F-score we get from any other fit to choose whether there is a motif or not.
    intercept_X = np.ones((len(records), 1))
    intercept_fit = evm.train_sklearn_glm(
        intercept_X,
        records.y,
        family = fam,
        fit_intercept = False,
    )

    intercept_metric = evm.CV_F1(
        intercept_X,
        records.y,
        folds = 5,
        family = fam,
        fit_intercept = False, # intercept already in design mat
        cores = args.nprocs,
    )

    seq_motif_exists = False
    shape_motif_exists = False

    # if we found motifs using streme, we'll have a value for seq_meme_file.
    # also will work if we passed a known motif file instead of finding our own
    # sequence motif
    if seq_meme_file is not None:

        # This step will just get the motif names and sequences,
        # hits arrays and such will be supplemented later using fimo output
        seq_motifs = inout.Motifs(
            fname = seq_meme_file,
            motif_type = "sequence",
            evalue_thresh = streme_thresh,
        )

        if len(seq_motifs) == 0:
            print()
            logging.info(f"No sequence motifs passed e-value "\
                f"threshold of {streme_thresh}, setting find_seq_motifs "\
                f"back to False and moving on to shape motif inference.")
            # set find_seq_motifs back to False to disable seq stuff later on
            find_seq_motifs = False
        else:
            find_seq_motifs = True

    if find_seq_motifs:

        logging.info("\nPlacing each motif's robustness into motifs")
        # set_X is called with nosort=True here just to get the hits array required
        # for supplement_robustness and sorting later
        #import ipdb; ipdb.set_trace()
        seq_motifs.set_X(
            max_count = max_count,
            fimo_fname = f"{fimo_direc}/fimo.tsv",
            rec_db = records,
            pval_thresh = streme_thresh,
            nosort = True,
        )

        seq_motifs.supplement_robustness(
            records,
            supp_bin,
            my_env=my_env,
            tmpdir=tmpdir,
        )

        seq_motifs.cmi_filter(
            max_count = max_count,
            binary = cmi_bin,
            fimo_fname = f"{fimo_direc}/fimo.tsv",
            rec_db = records,
            pval_thresh = streme_thresh,
            my_env = my_env,
            tmpdir = tmpdir,
        )
        if len(seq_motifs) == 0:
            print()
            logging.info(
                f"No sequence motifs left after CMI filter.\n"\
                f"Therefore, no informatife sequence motif exists."
            )
            seq_motif_exists = False

        else:

            logging.info("\nSorting sequence motifs by z-score")
       
            logging.info("\nFitting regression model to sequence motifs")
            seq_motifs.set_X(
                max_count = max_count,
                fimo_fname = f"{fimo_direc}/fimo.tsv",
                rec_db = records,
                pval_thresh = streme_thresh,
            )
            print(f"X shape: {seq_motifs.X.shape}")
            print(f"distinct y vals: {np.unique(records.y)}")

            print("Done getting X for seq motifs")
            one_seq_motif = False

            if len(seq_motifs) == 1:

                print()
                logging.info(
                    f"Only one sequence motif present. "\
                    f"Performing model selection using CV-F1 to determine whether "\
                    f"the motif is informative over intercept alone."
                )
                # toggle one_seq_motif to True for later use in building combined
                # seq and shape motif design matrix
                ######################################################################
                one_seq_motif = True # comment for debugging to force seq inclusion
                #seq_motif_exists = True # uncomment for debugging to force seq inclusion

            else:

                #import ipdb; ipdb.set_trace()
                seq_fit = evm.train_glmnet(
                    seq_motifs.X,
                    records.y,
                    folds=10,
                    family = fam,
                    alpha=1,
                )

                #print(f"X shape: {seq_motifs.X.shape}")
                #print(f"distinct y vals: {np.unique(records.y)}")
                with open(seq_fit_fname, "wb") as f:
                    pickle.dump(seq_fit, f)

                seq_coefs = evm.fetch_coefficients(fam, seq_fit, num_cats)

                print()
                logging.info(f"Sequence motif coefficients:\n{seq_coefs}")
                logging.info(f"Sequence coefficient lookup table:\n{seq_motifs.var_lut}")

                filtered_seq_coefs = seq_motifs.filter_motifs(
                    seq_coefs,
                    fimo_fname = f"{fimo_direc}/fimo.tsv",
                    rec_db = records,
                    pval_thresh = streme_thresh,
                )

                print()
                logging.info(
                    f"Number of sequence motifs left after LASSO regression: "\
                    f"{len(seq_motifs)}"
                )

                if len(seq_motifs) == 0:
                    print()
                    logging.info(
                        f"Only intercept term left after LASSO regression.\n"\
                        f"Therefore, no informatife sequence motif exists."
                    )
                    seq_motif_exists = False

                elif len(seq_motifs) == 1:
                    print()
                    logging.info(
                        f"Only one sequence motif left after LASSO regression.\n"\
                        f"Performing model selection using CV-F1 to determine whether "\
                        f"the remaining motif is informative over intercept alone."
                    )

                    one_seq_motif = True

                # if more than one left after LASSO, seq seq_motif_exists to True
                else:
                    seq_motif_exists = True
 
            # supplement motifs object with CV-F1 score
            intercept_and_motif_X = np.append(intercept_X, seq_motifs.X, axis=1)

            motif_fit = evm.train_sklearn_glm(
                intercept_and_motif_X,
                records.y,
                family = fam,
                fit_intercept = False, # intercept already in design mat
            )

            seq_motifs.metric = evm.CV_F1(
                intercept_and_motif_X,
                records.y,
                folds = 5,
                family = fam,
                fit_intercept = False, # intercept already in design mat
                cores = args.nprocs,
            )

            # if there's only one covariate, compare CV-F1 from intercept+motif
            # and intercept only
            if one_seq_motif:
                metric_list = [ intercept_metric, seq_motifs.metric ]
                model_list = [ intercept_fit, motif_fit ]

                best_mod_idx = evm.choose_model(
                    metric_list,
                    model_list,
                    return_index = True,
                )

                if best_mod_idx == 0:
                    print()
                    logging.info(
                        f"Intercept-only model had better score than model fit using "\
                        f"intercept and one sequence motif.\nTherefore, there is no "\
                        f"informative "\
                        f"sequence motif. Not writing a sequence motif to output."
                    )
                    seq_motif_exists = False
                # if our one seq motif is better than intercept, set seq_motif_exits to True
                else:
                    print()
                    logging.info(
                        f"Sequence-motif-containing model performed better "\
                        f"than intercept-only model. Therefore, at least one "\
                        f"informative sequence motif exists."
                    )

                    filtered_seq_coefs = motif_fit.coef_
                    seq_coefs = motif_fit.coef_
                    print()
                    logging.info(f"Sequence motif coefficients:\n{filtered_seq_coefs}")
                    logging.info(f"Sequence coefficient lookup table:\n{seq_motifs.var_lut}")
                    seq_motif_exists = True

    good_motif_out_fname = os.path.join(
        out_direc,
        f"{out_pref}_post_opt_cmi_filtered_motifs_temp_{temp}_"\
        f"stepsize_{step}_alpha_{alpha}_max_count_{max_count}.pkl",
    )

    seq_motif_plot_fname = os.path.join(
        out_direc,
        f"seq_motif_logo.png"
    )
    shape_motif_plot_fname = os.path.join(
        out_direc,
        f"shape_motif_logo.png"
    )
    final_motif_plot_suffix = os.path.join(
        out_direc,
        "{}_final_motif.png"
    )

    logit_reg_fname = os.path.join(
        out_direc,
        f"{out_pref}_logistic_regression_result.pkl",
    )

    coef_per_class_fname = os.path.join(
        out_direc,
        f"{out_pref}_logistic_regression_coefs_per_class.txt",
    )

    #res_log_fname = os.path.join(out_direc, "infer_motifs_bin.log")
    #res_err_fname = os.path.join(out_direc, "infer_motifs_bin.err")
    #run_output = result.stdout.decode()
    #with open(res_log_fname, "w") as logf:
    #    logf.write(run_output)
    #run_err = result.stderr.decode()
    #with open(res_err_fname, "w") as errf:
    #    errf.write(run_err)
 
    FIND_CMD = f"{infer_bin} {config_fname} > {res_log_fname} 2> {res_err_fname}"

    max_batch = args.max_batch_no_new_seed
    if args.exhaustive:
        max_batch = 1_000_000_000

    find_args_dict = {
        'out_fname': rust_out_fname,
        'shape_fname': shape_fname,
        'yvals_fname': yval_fname,
        'alpha': args.alpha,
        'max_count': max_count,
        'kmer': args.kmer,
        'cores': args.nprocs,
        'seed_sample_size': args.init_threshold_seed_num,
        'records_per_seed': args.init_threshold_recs_per_seed,
        'windows_per_record': args.init_threshold_windows_per_record,
        'thresh_sd_from_mean': args.threshold_sd,
        'threshold_lb': args.threshold_constraints[0],
        'threshold_ub': args.threshold_constraints[1],
        'shape_lb': args.shape_constraints[0],
        'shape_ub': args.shape_constraints[1],
        'weight_lb': args.weights_constraints[0],
        'weight_ub': args.weights_constraints[1],
        'temperature': args.temperature,
        'stepsize': args.stepsize,
        'n_opt_iter': args.opt_niter,
        't_adjust': args.t_adj,
        'batch_size': args.batch_size,
        'max_batch_no_new': max_batch,
        'good_motif_out_fname': good_motif_out_fname, 
    }

    if args.continuous is not None:
        find_args_dict['y_cat_num'] = num_cats

    # supplement args info with shape center and spread from database
    find_args_dict['names'] = []
    find_args_dict['indices'] = []
    find_args_dict['centers'] = []
    find_args_dict['spreads'] = []

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        find_args_dict['names'].append(name)
        find_args_dict['indices'].append(shape_idx)
        find_args_dict['centers'].append(this_center)
        find_args_dict['spreads'].append(this_spread)
    
    # write cfg to file
    with open(config_fname, 'w') as f:
        json.dump(find_args_dict, f, indent=1)

    if not no_shape_motifs:
        # write shapes to npy file. Permute axes 1 and 2.
        with open(shape_fname, 'wb') as shape_f:
            np.save(shape_fname, records.X.transpose((0,2,1)))

        print()
        if args.shape_rust_file is None:
            logging.info("Running shape motif selection and optimization.")
            result = subprocess.run(
                FIND_CMD,
                shell=True,
                env=my_env,
                capture_output=True,
            )
            # here is a spot where I need to write the ouptut of the rust binary to logs
           
            if result.returncode != 0:
                raise inout.RustBinaryException(FIND_CMD)
            if "No shape motifs found by infer_motifs binary." in result.stdout.decode():
                no_shape_motifs = True
        else:
            logging.info(f"Reading prior shape motifs from {args.shape_rust_file}.")
            rust_out_fname = args.shape_rust_file

        if not os.path.isfile(rust_out_fname):
            print()
            logging.warning(
                f"No output json file containing motifs from rust binary. "\
                f"This usually means no motifs were identified, but you should "\
                f"carfully check your log and error messages to make sure that's "\
                f"really the case."
            )
            no_shape_motifs = True

    if not no_shape_motifs:

        shape_motifs = inout.Motifs(
            rust_out_fname,
            motif_type="shape",
            shape_lut = records.shape_name_lut,
            max_count = max_count,
        )

        # places design matrix and variable lookup table into shape_motifs
        shape_motifs.set_X(
            max_count = max_count,
            fimo_fname = f"{fimo_direc}/fimo.tsv",
            rec_db = records,
        )
        logging.info(f"Shape coefficient lookup table:\n{shape_motifs.var_lut}")

        # check whether there are more than one covariates, do LASSO regression
        if shape_motifs.X.shape[1] > 1:

            shape_fit = evm.train_glmnet(
                shape_motifs.X,
                records.y,
                folds = 10,
                family=fam,
                alpha=1,
            )

            with open(shape_fit_fname, "wb") as f:
                pickle.dump(shape_fit, f)

            coefs = evm.fetch_coefficients(fam, shape_fit, num_cats)

            print()
            logging.info(f"Shape motif coefficients:\n{coefs}")
            logging.info(f"Shape coefficient lookup table:\n{shape_motifs.var_lut}")

            # go through coefficients and weed out motifs for which all
            #   hits' coefficients are zero.
            filtered_shape_coefs = shape_motifs.filter_motifs(
                coefs,
                max_count = max_count,
                rec_db = records,
            )

            print()
            logging.info(
                f"Number of shape motifs left after LASSO regression: "\
                f"{len(shape_motifs)}"
            )
            logging.info(
                f"Shape coefficient lookup table after filter:\n"\
                f"{shape_motifs.var_lut}"
            )

        # supplement the shape motifs object with the CV-F1 from a model
        intercept_and_shape_X = np.append(intercept_X, shape_motifs.X, axis=1)

        motif_fit = evm.train_sklearn_glm(
            intercept_and_shape_X,
            records.y,
            family = fam,
            fit_intercept = False, # intercept already in design mat
        )

        if shape_motifs.X.shape[1] == 1:
            filtered_shape_coefs = motif_fit.coef_

        shape_motifs.metric = evm.CV_F1(
            intercept_and_shape_X,
            records.y,
            folds = 5,
            family = fam,
            fit_intercept = False, # intercept already in design mat
            cores = args.nprocs,
        )

        # check whether there's only one informative covariate
        if shape_motifs.X.shape[1] == 1:
            print()
            logging.info(
                f"Only one covariate for shape motifs was found to be "\
                f"informative using LASSO regression. Calculating the scoring metric "\
                f"for a model with only an intercept and this covariate to "\
                f"compare to a model fit using only an intercept."
            )

            metric_list = [ intercept_metric, shape_motifs.metric ]
            model_list = [ intercept_fit, motif_fit ]

            best_mod_idx = evm.choose_model(
                metric_list,
                model_list,
                return_index = True,
            )

            print()
            logging.info(
                f"Intercept-only metric: {intercept_metric}\n"\
                f"Intercept and one shape covariate metric: {shape_motifs.metric}"
            )

            if best_mod_idx == 0:
                print()
                logging.info(
                    f"Intercept-only model had better score than model fit using "\
                    f"intercept and one shape covariate. Therefore, there is no "\
                    f"informative shape motif. Not writing a shape motif to output."\
                    f"Exiting now."
                )

                if not args.no_report:
                    report_info = {}
                    write_report(
                        environ = jinja_env,
                        temp_base = "no_motifs.html.temp",
                        info = report_info,
                        out_name = out_page_name,
                    )

                status = "FinishedNoMotif"
                with open(status_fname, "w") as status_f:
                    json.dump(status, status_f)
                sys.exit()
            # if the shape performs better than intercept, set shape_motif_exists to True
            else:
                shape_motif_exists = True

        elif len(shape_motifs) == 0:
            print()
            logging.warning(
                f"There were no shape motifs left after LASSO regression. "\
                f"Exiting now."
            )
            no_shape_motifs = True

        # if more than one covariate left after LASSO, set shape_motif_exists to True
        else:
            shape_motif_exists = True

    if not no_shape_motifs:

        # if there were both shape and seq motifs, combine into one model
        if shape_motif_exists and seq_motif_exists:

            #if num_cats != 2:
            #    print(
            #        f"Combining shape and sequence motifs only supported "\
            #        f"for binary inputs. Skipping merged sequence and shape model "\
            #        f"steps."
            #    )

            #else:

            shape_and_seq_motifs = shape_motifs.new_with_motifs(
                seq_motifs,
                max_count = max_count,
                fimo_fname = f"{fimo_direc}/fimo.tsv",
                rec_db = records,
                pval_thresh = streme_thresh,
            )
            shape_and_seq_motifs.motif_type = "shape_and_seq"

            shape_and_seq_motifs.cmi_filter(
                max_count = max_count,
                binary = cmi_bin,
                fimo_fname = f"{fimo_direc}/fimo.tsv",
                rec_db = records,
                pval_thresh = streme_thresh,
                my_env = my_env,
                tmpdir = tmpdir,
            )
            # check whether all motifs of a specific type were removed using cmi
            motif_types = [motif.motif_type for motif in shape_and_seq_motifs]
            if not "sequence" in motif_types:
                logging.info(
                    f"Filtering shape and sequence motif model using CMI "\
                    f"removed all sequence motifs. Moving forward with only shape "\
                    f"motifs."
                )
                seq_motif_exists = False
            if not "shape" in motif_types:
                logging.info(
                    f"Filtering shape and sequence motif model using CMI "\
                    f"removed all shape motifs. Moving forward with only sequence "\
                    f"motifs."
                )
                shape_motif_exists = False

            # if we still see evidence that shape AND sequence are important,
            #  do the next stuff
            if shape_motif_exists and seq_motif_exists:

                shape_and_seq_motifs.set_X(
                    max_count = max_count,
                    fimo_fname = f"{fimo_direc}/fimo.tsv",
                    rec_db = records,
                    pval_thresh = streme_thresh,
                )
                print(f"shape_and_seq_var_lut: {shape_and_seq_motifs.var_lut}")
                print(f"seq_var_lut: {seq_motifs.var_lut}")
                print(f"shape_and_seq_var_lut: {shape_and_seq_motifs.var_lut}")

                shape_and_seq_fit = evm.train_glmnet(
                    shape_and_seq_motifs.X,
                    records.y,
                    folds=10,
                    family=fam,
                    alpha=1,
                )
                with open(shape_and_seq_fit_fname, "wb") as f:
                    pickle.dump(shape_and_seq_fit, f)

                #shape_and_seq_motifs = shape_motifs.copy()
                #shape_and_seq_motifs.extend(seq_motifs.copy())

                shape_and_seq_coefs = evm.fetch_coefficients(
                    fam,
                    shape_and_seq_fit,
                    num_cats,
                )

                print()
                logging.info(
                    f"Shape and sequence motif coefficients:\n"\
                    f"{shape_and_seq_coefs}"
                )
                logging.info(
                    f"Shape and sequence coefficient lookup table:\n"\
                    f"{shape_and_seq_motifs.var_lut}"
                )

                filtered_shape_and_seq_coefs = shape_and_seq_motifs.filter_motifs(
                    shape_and_seq_coefs,
                    max_count = max_count,
                    fimo_fname = f"{fimo_direc}/fimo.tsv",
                    rec_db = records,
                    pval_thresh = streme_thresh,
                )

                print()
                logging.info(f"Number of final motifs: {len(shape_and_seq_motifs)}")

                # supplement motifs object with CV-F1
                intercept_and_shape_and_seq_X = np.append(
                    intercept_X,
                    shape_and_seq_motifs.X,
                    axis=1,
                )

                int_and_shape_and_seq_fit = evm.train_sklearn_glm(
                    intercept_and_shape_and_seq_X,
                    records.y,
                    family = fam,
                    fit_intercept = False, # intercept already in design mat
                )

                shape_and_seq_motifs.metric = evm.CV_F1(
                    intercept_and_shape_and_seq_X,
                    records.y,
                    folds = 5,
                    family = fam,
                    fit_intercept = False, # intercept already in design mat
                    cores = args.nprocs,
                )

                if len(shape_and_seq_motifs) == 0:
                    print()
                    logging.info(
                        f"Only intercept term left after LASSO regression.\n"\
                        f"Therefore, no informative sequence or shape motif exists."\
                        f"Not writing a motif to output. Exiting now."
                    )

                    if not args.no_report:
                        report_info = {}
                        write_report(
                            environ = jinja_env,
                            temp_base = "no_motifs.html.temp",
                            info = report_info,
                            out_name = out_page_name,
                        )

                    status = "FinishedNoMotif"
                    with open(status_fname, "w") as status_f:
                        json.dump(status, status_f)
                    sys.exit()

                elif len(shape_and_seq_motifs) == 1:
                    print()
                    logging.info(
                        f"Only one motif left after LASSO regression. "\
                        f"Performing model selection using F1 to determine whether "\
                        f"the remaining motif is informative over intercept alone."
                    )
     
                    metric_list = [ intercept_metric, shape_and_seq_motifs.metric ]
                    model_list = [ intercept_fit, int_and_shape_and_seq_fit ]

                    best_mod_idx = evm.choose_model(
                        metric_list,
                        model_list,
                        return_index = True,
                    )

                    print()
                    logging.info(
                        f"Intercept-only F-score: {intercept_metric}\n"\
                        f"Intercept and one covariate F-score: {shape_and_seq_motifs.metric}"
                    )
                    if best_mod_idx == 0:
                        print()
                        logging.info(
                            f"Intercept-only model had better score than model fit using "\
                            f"intercept and one motif covariate.\nTherefore, there is no "\
                            f"informative motif. Not writing a motif to output. "\
                            f"Exiting now."
                        )

                        if not args.no_report:
                            report_info = {}
                            write_report(
                                environ = jinja_env,
                                temp_base = "no_motifs.html.temp",
                                info = report_info,
                                out_name = out_page_name,
                            )

                        status = "FinishedNoMotif"
                        with open(status_fname, "w") as status_f:
                            json.dump(status, status_f)
                        sys.exit()

    motifs_info = []
    if shape_motif_exists:
        motifs_info.append((shape_motifs, filtered_shape_coefs))
        if args.write_all_files:
            out_fname = out_motif_basename + "_shape_motifs.dsm"
            shape_motifs.write_file(out_fname, records)
    if seq_motif_exists:
        motifs_info.append((seq_motifs, filtered_seq_coefs))
        if args.write_all_files:
            out_fname = out_motif_basename + "_sequence_motifs.dsm"
            seq_motifs.write_file(out_fname, records)
    if shape_motif_exists and seq_motif_exists:
        #if num_cats != 2:
        motifs_info.append((shape_and_seq_motifs, filtered_shape_and_seq_coefs))
        if args.write_all_files:
            out_fname = out_motif_basename + "_shape_and_sequence_motifs.dsm"
            shape_and_seq_motifs.write_file(out_fname, records)

    if not np.any([seq_motif_exists, shape_motif_exists]):
        print()
        logging.info("No shape or sequence motifs found. Exiting now.")

        if not args.no_report:
            report_info = {}
            write_report(
                environ = jinja_env,
                temp_base = "no_motifs.html.temp",
                info = report_info,
                out_name = out_page_name,
            )

        status = "FinishedNoMotif"
        with open(status_fname, "w") as status_f:
            json.dump(status, status_f)
        sys.exit()

    # if there was more than one inout.Motifs object generated, choose best model here
    if len(motifs_info) > 1:

        motif_metrics = [x[0].metric for x in motifs_info]

        best_motifs,best_motif_coefs = evm.choose_model(
            motif_metrics,
            motifs_info,
            return_index=False,
        )
        print()
        logging.info(f"Best model, based on F-score, was {best_motifs.motif_type}.")

    # if only one, set the extant one to "best_motifs"
    else:
        best_motifs,best_motif_coefs = motifs_info[0]

    #######################################################################
    #best_motifs = motifs_info[-1] # uncomment for forcing a specific model for debug

    print(f"all motifs_info:\n{motifs_info}")
    print(f"Best motif coefficients:\n{best_motif_coefs}")
    print(f"Best motif info:\n{best_motifs}")

    # plot_fnames is a list of tuples with paired png file names and motif objects
    plot_fnames = smv.plot_logos(
        best_motifs,
        final_motif_plot_suffix,
        records.shape_name_lut,
        #top_n = np.Inf,
    )

    # modify each motif's enrichments attribute
    best_motifs.get_enrichments(records)

    logo_data = []
    job_id = in_direc.split("/")[-1]
    for plot_fname,motif in plot_fnames:
        with open(plot_fname, "rb") as image_file:
            logo_img = base64.b64encode(image_file.read()).decode()
        motif.set_evalue(len(best_motifs))
        plot_basename = os.path.basename(plot_fname)
        logo_data.append((
            logo_img,
            motif.alt_name,
            motif.identifier,
            np.round(motif.mi, 2),
            np.round(motif.zscore, 1),
            motif.robustness,
            np.round(motif.evalue, 2),
            f"{job_id}/{plot_basename}",
        ))

    # write motifs to meme-like file
    best_motifs.write_file(out_motif_fname, records)
    coef_dict = {
        "coefs": best_motif_coefs,
        "var_lut": best_motifs.var_lut
    }
    with open(out_coefs_fname, "wb") as out_coef_f:
        pickle.dump(coef_dict, out_coef_f)
    logging.info(f"Writing motif enrichment heatmap to {out_heatmap_fname}")

    smv.plot_motif_enrichment_seaborn(best_motifs, out_heatmap_fname, records)
    with open(out_heatmap_fname, "rb") as image_file:
        heatmap_data = base64.b64encode(image_file.read()).decode()
    logging.info(f"Finished motif inference. Final results are in {out_motif_fname}")

    report_info = {
        "logo_data": logo_data,
        "heatmap_data": heatmap_data,
        "heatmap_path": f"{job_id}/final_heatmap.png",
    }
    print(f"heatmap_path: {report_info['heatmap_path']}")

    if not args.no_report:
        print(f"report_info keys:\n{report_info.keys()}")
        write_report(
            environ = jinja_env,
            temp_base = "motifs.html.temp",
            info = report_info,
            out_name = out_page_name,
        )
        print("finished writing report")
    else:
        report_data_fname = os.path.join(out_direc, "report_data.pkl")
        with open(report_data_fname, "wb") as report_data_f:
            pickle.dump(report_info, report_data_f)

    status = "FinishedWithMotifs"
    with open(status_fname, "w") as status_f:
        json.dump(status, status_f)


if __name__ == "__main__":

    status = "Running"
    args = parse_args()

    try:
        main(args, status)
    except Exception as err:
        logging.error(f"\nError encountered in infer_motifs.py:\n{err}\n")
        status = "FinishedError"
        in_direc = args.data_dir
        out_direc = args.out_dir
        out_direc = os.path.join(in_direc, out_direc)

        if not os.path.isdir(out_direc):
            os.mkdir(out_direc)

        out_page_name = os.path.join(out_direc, "report.html")

        report_info = {
            "error": err,
        }
        write_report(
            environ = jinja_env,
            temp_base = "error.html.temp",
            info = report_info,
            out_name = out_page_name,
        )

        status_fname = os.path.join(out_direc, "job_status.json")
        with open(status_fname, "w") as status_f:
            json.dump(status, status_f)
        sys.exit(1)

