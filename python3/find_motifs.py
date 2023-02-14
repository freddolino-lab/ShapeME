import inout
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
from pathlib import Path

import evaluate_motifs as evm
import fimopytools as fimo

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)
find_bin = os.path.join(this_path, '../rust_utils/target/release/find_motifs')
supp_bin = os.path.join(this_path, '../rust_utils/target/release/get_robustness')

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# import R's "PRROC" package
prroc = importr('PRROC')
glmnet = importr('glmnet')
base = importr('base')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', action='store', type=str, required=True,
        help='input text file with names and scores for training data')
    parser.add_argument('--test_infile', action='store', type=str, default=None,
        help='input text file with sequence names and scores for held-out testing data.')
    parser.add_argument('--params', nargs="+", type=str, required=True,
        help='input files with shape scores')
    parser.add_argument('--test_params', nargs="+", type=str, default=None,
        help='input files with shape scores for held-out testing data.')
    parser.add_argument('--param_names', nargs="+", type=str,
        help='parameter names (MUST BE IN SAME ORDER AS CORRESPONDING PARAMETER FILES)')
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
        help=f"Sets the stepsize argument for scipy.optimize.basinhopping. "\
            f"Default: %(default)f")
    parser.add_argument('--opt_niter', type=int, default=100,
        help=f"Sets the number of simulated annealing iterations to "\
            f"undergo during optimization. Default: %(default)d.")
    parser.add_argument('--kmer', type=int,
        help='kmer size to search for. Default=%(default)d', default=15)
    parser.add_argument('--nonormalize', action="store_true",
        help='don\'t normalize the input data by robustZ')
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
    parser.add_argument('--motif_perc', type=float, default=1,
        help="fraction of data to EVALUATE motifs on. Default=%(default)f")
    parser.add_argument('--continuous', type=int, default=None,
        help="number of bins to discretize continuous input data with")
    parser.add_argument('--alpha', type=float, default=0.0,
        help=f"Lower limit on transformed weight values prior to "\
            f"normalization to sum to 1. Default: %(default)f")
    parser.add_argument('--max_count', type=int, default=1,
        help=f"Maximum number of times a motif can match "\
            f"each of the forward and reverse strands in a reference. "\
            f"Default: %(default)d")
    parser.add_argument('-o', type=str, required=True,
        help="Prefix to apply to output files.")
    parser.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    parser.add_argument('--out_dir', type=str, required=True,
        help="Directory (within 'data_dir') into which output files will be written.")
    parser.add_argument('-p', type=int, default=5,
        help="number of processors. Default: %(default)d")
    parser.add_argument('--batch_size', type=int, default=2000,
        help=f"Number of records to process seeds from at a time. Set lower "\
            f"to avoid out-of-memory errors. Default: %(default)d")
    parser.add_argument('--find_seq_motifs', action="store_true",
        help=f"Add this flag to call sequence motifs using streme in addition "\
            f"to calling shape motifs.")
    parser.add_argument('--streme_thresh', default = 0.05,
        help="Threshold for including motifs identified by streme. Default: %(default)f")
    parser.add_argument("--seq_fasta", type=str, default=None,
        help=f"Name of fasta file (located within in_direc, do not include the "\
            f"directory, just the file name) containing sequences in which to "\
            f"search for motifs")
    parser.add_argument("--seq_meme_file", type=str, default=None,
        help=f"Name of meme-formatted file (file must be located in data_dir) "\
            f"to be used for searching for known sequence motifs of interest in "\
            f"seq_fasta")
    parser.add_argument("--shape_rust_file", type=str, default=None,
        help=f"Name of json file containing output from rust binary")
    parser.add_argument("--no_shape_motifs", action="store_true",
        help=f"Add this flag to turn off shape motif inference. "\
            f"This is useful if you basically want to use this script "\
            f"as a wrapper for streme to just find sequence motifs.")
    parser.add_argument("--write_all_files", action="store_true",
        help=f"Add this flag to write all motif meme files, regardless of whether "\
            f"the model with shape motifs, sequence motifs, or both types of "\
            f"motifs was most performant.")
    parser.add_argument("--exhaustive", action="store_true", default=False,
        help=f"Add this flag to perform and exhaustive initial search for seeds. "\
            f"This can take a very long time for datasets with more than a few-thousand "\
            f"binding sites. Setting this option will ignore the --max-rounds-no-new-seed "\
            f"option.")
    parser.add_argument("--max-batch-no-new-seed", type=int, default=10,
        help=f"Sets the number of batches of seed evaluation with no new motifs "\
            f"added to the set of motifs to be optimized prior to truncating the "\
            f"initial search for motifs.")

    my_env = os.environ.copy()
    my_env['RUST_BACKTRACE'] = "1"

    level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=level,
        stream=sys.stdout,
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    args = parser.parse_args()

    logging.info("Arguments:")
    print(str(args))

    out_pref = args.o
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)
    in_fname = os.path.join(in_direc, args.infile)
    out_motif_basename = os.path.join(out_direc, "final_motifs")
    out_motif_fname = out_motif_basename + ".dsm"
    out_coefs_fname = out_motif_basename + "_coefficients.npy"
    out_heatmap_fname = os.path.join(out_direc, "final_heatmap.png")
    find_seq_motifs = args.find_seq_motifs
    seq_fasta = args.seq_fasta
    if seq_fasta is not None:
        seq_fasta = os.path.join(in_direc, seq_fasta)
    seq_meme_file = args.seq_meme_file
    if seq_meme_file is not None:
        seq_meme_file = os.path.join(in_direc, seq_meme_file)
    fimo_direc = f"{out_direc}/fimo_out"
    streme_direc = f"{out_direc}/streme_out"
    streme_thresh = args.streme_thresh
    no_shape_motifs = args.no_shape_motifs

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    if find_seq_motifs:
        # if asked for seq motifs but didn't pass seq fa file, exception
        if seq_fasta is None:
            raise inout.NoSeqFaException()
        # if both seq_motifs and meme file were passed, raise exception
        if seq_meme_file is not None:
            raise inout.SeqMotifOptionException(seq_meme_file)

        seq_meme_file = f"{streme_direc}/streme.txt"
        streme_exec = os.path.join(this_path, "run_streme.py")

        STREME = f"python {streme_exec} "\
            f"--seq_fname {seq_fasta} "\
            f"--yvals_fname {in_fname} "\
            f"--threshold {streme_thresh} "\
            f"--out_direc {streme_direc}"

        streme_result = subprocess.run(
            STREME,
            shell=True,
            capture_output=True,
        )
        if streme_result.returncode != 0:
            raise(Exception(
                f"run_streme.py returned non-zero exit status.\n"\
                f"Stderr: {streme_result.stderr.decode()}\n"\
                f"Stdout: {streme_result.stdout.decode()}"
            ))

        streme_log_fname = f"{streme_direc}/streme_run.log"
        streme_err_fname = f"{streme_direc}/streme_run.err"
        print()
        logging.info(
            f"Ran streme: for details, see "\
            f"{streme_log_fname} and {streme_err_fname}"
        )
        with open(streme_log_fname, "w") as streme_out:
            # streme log gets captured as stderr, so write stderr to file
            streme_out.write(streme_result.stdout.decode())
        with open(streme_err_fname, "w") as streme_err:
            # streme log gets captured as stderr, so write stderr to file
            try:
                streme_err.write(streme_result.stderr.decode())
            except UnicodeDecodeError as e:
                logging.warning("Problem writing to {streme_err_fname}:\n{e}")

    # if user has a meme file (could be from streme above, or from input arg), run fimo
    if seq_meme_file is not None:

        if seq_fasta is None:
            raise inout.NoSeqFaException()

        fimo_exec = os.path.join(this_path, "run_fimo.py")
        FIMO = f"python {fimo_exec} "\
            f"--seq_fname {seq_fasta} "\
            f"--meme_file {seq_meme_file} "\
            f"--out_direc {fimo_direc}"

        fimo_result = subprocess.run(
            FIMO,
            shell=True,
            check=True,
            capture_output=True,
        )
        fimo_log_fname = f"{fimo_direc}/fimo_run.log"
        fimo_err_fname = f"{fimo_direc}/fimo_run.err"
        print()
        logging.info(
            f"Ran fimo: for details, see "\
            f"{fimo_log_fname} and {fimo_err_fname}"
        )
        with open(fimo_log_fname, "w") as fimo_out:
            fimo_out.write(fimo_result.stdout.decode())
        with open(fimo_err_fname, "w") as fimo_err:
            fimo_err.write(fimo_result.stderr.decode())

    print()
    logging.info("Reading in shape files")
    # read in shapes
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(args.param_names, args.params)
    }
    logging.info("Reading input data and shape info.")
    records = inout.RecordDatabase(
        in_fname,
        shape_fname_dict,
        shift_params = ["Roll", "HelT"],
    )
    records = records.permute_records()
    assert len(records.y) == records.X.shape[0], "Number of y values does not equal number of shape records!!"
           
    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        records.quantize_quant(args.continuous)

    fam,num_cats = evm.set_family(records.y)

    logging.info("Distribution of sequences per class:")
    logging.info(inout.seqs_per_bin(records))

    logging.info("Normalizing parameters")
    if args.nonormalize:
        records.determine_center_spread(method=inout.identity_csp)
    else:
        records.determine_center_spread()
        records.normalize_shape_values()

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        logging.info(f"{name}: center={this_center:.2f}, spread={this_spread:.2f}")

    alpha = args.alpha
    max_count = args.max_count

    temp = args.temperature
    step = args.stepsize
    
    mi_fname = os.path.join(
        out_direc,
        f'{out_pref}_initial_mutual_information_max_count_{max_count}.pkl'
    )

    shape_fname = os.path.join(out_direc, 'shapes.npy')
    yval_fname = os.path.join(out_direc, 'y_vals.npy')
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
        cores = args.p,
    )

    #intercept_bic = evm.get_sklearn_bic(
    #    intercept_X,
    #    records.y,
    #    intercept_fit,
    #)

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

        logging.info("\nFitting regression model to sequence motifs")
        
        seq_motifs.get_X(
            fimo_fname = f"{fimo_direc}/fimo.tsv",
            rec_db = records,
        )
        seq_motifs.supplement_robustness(records, supp_bin, my_env=my_env)

        one_seq_motif = False

        if len(seq_motifs) == 1:

            print()
            logging.info(
                f"Only one sequence motif present. "\
                f"Performing model selection using BIC to determine whether "\
                f"the motif is informative over intercept alone."
            )
            # toggle one_seq_motif to True for later use in building combined
            # seq and shape motif design matrix
            ######################################################################
            one_seq_motif = True # comment for debugging to force seq inclusion
            #seq_motif_exists = True # uncomment for debugging to force seq inclusion

        else:

            seq_fit = evm.train_glmnet(
                seq_motifs.X,
                records.y,
                folds=10,
                family=fam,
                alpha=1,
            )

            with open(seq_fit_fname, "wb") as f:
                pickle.dump(seq_fit, f)

            seq_coefs = evm.fetch_coefficients(fam, seq_fit, num_cats)

            print()
            logging.info(f"Sequence motif coefficients:\n{seq_coefs}")
            logging.info(f"Sequence coefficient lookup table:\n{seq_motifs.var_lut}")

            filtered_seq_coefs = seq_motifs.filter_motifs(seq_coefs)

            print()
            logging.info(
                f"Number of sequence motifs left after LASSO regression: "\
                f"{len(seq_motifs)}"
            )

            if len(seq_motifs) == 1:
                print()
                logging.info(
                    f"Only one sequence motif left after LASSO regression.\n"\
                    f"Performing model selection using BIC to determine whether "\
                    f"the remaining motif is informative over intercept alone."
                )
                one_seq_motif = True

            # if more than one left after LASSO, seq seq_motif_exists to True
            else:
                seq_motif_exists = True
 
        # supplement motifs object with bic
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
            cores = args.p,
        )

        #seq_motifs.bic = evm.get_sklearn_bic(
        #    intercept_and_motif_X,
        #    records.y,
        #    motif_fit,
        #)

        # if there's only one covariate, compare bic from intercept+motif
        # and intercept only
        if one_seq_motif:
            #bic_list = [ intercept_bic, seq_motifs.bic ]
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
            # if our one seq motif is better than intercept, set seq_motif_exits to True
            else:
                seq_motif_exists = True

    good_motif_out_fname = os.path.join(
        out_direc,
        f"{out_pref}_post_opt_cmi_filtered_motifs_temp_{temp}_"\
        f"stepsize_{step}_alpha_{alpha}_max_count_{max_count}.pkl",
    )

    final_motif_plot_fname = os.path.join(
        out_direc,
        f"{out_pref}_post_opt_cmi_filtered_motifs_temp_{temp}_"\
        f"stepsize_{step}_alpha_{alpha}_max_count_{max_count}.png",
    )

    logit_reg_fname = os.path.join(
        out_direc,
        f"{out_pref}_logistic_regression_result.pkl",
    )

    coef_per_class_fname = os.path.join(
        out_direc,
        f"{out_pref}_logistic_regression_coefs_per_class.txt",
    )

    FIND_CMD = f"{find_bin} {config_fname}"

    max_batch = args.max_batch_no_new_seed
    if args.exhaustive:
        max_batch = 1_000_000_000

    find_args_dict = {
        'out_fname': rust_out_fname,
        'shape_fname': shape_fname,
        'yvals_fname': yval_fname,
        'alpha': args.alpha,
        'max_count': args.max_count,
        'kmer': args.kmer,
        'cores': args.p,
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
        find_args_dist['y_cat_num'] = num_cats

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
            np.save(shape_fname, records.X.transpose((0,2,1,3)))
        # write y-vals to npy file.
        with open(yval_fname, 'wb') as f:
            np.save(f, records.y.astype(np.int64))

        print()
        if args.shape_rust_file is None:
            logging.info("Running shape motif selection and optimization.")
            retcode = subprocess.call(FIND_CMD, shell=True, env=my_env)
            if retcode != 0:
                raise inout.RustBinaryException(Exception)
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
            max_count = args.max_count,
        )
        # places design matrix and variable lookup table into shape_motifs
        shape_motifs.get_X(max_count = args.max_count)

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
        filtered_shape_coefs = shape_motifs.filter_motifs(coefs)

        print()
        logging.info(
            f"Number of shape motifs left after LASSO regression: "\
            f"{len(shape_motifs)}"
        )
        logging.info(
            f"Shape coefficient lookup table after filter:\n"\
            f"{shape_motifs.var_lut}"
        )

        # supplement the shape motifs object with the bic from a model
        intercept_and_shape_X = np.append(intercept_X, shape_motifs.X, axis=1)

        motif_fit = evm.train_sklearn_glm(
            intercept_and_shape_X,
            records.y,
            family = fam,
            fit_intercept = False, # intercept already in design mat
        )

        shape_motifs.metric = evm.CV_F1(
            intercept_and_shape_X,
            records.y,
            folds = 5,
            family = fam,
            fit_intercept = False, # intercept already in design mat
            cores = args.p,
        )

        #shape_motifs.bic = evm.get_sklearn_bic(
        #    intercept_and_shape_X,
        #    records.y,
        #    motif_fit,
        #)
  
        # check whether there's only one informative covariate
        if shape_motifs.X.shape[1] == 1:
            print()
            logging.info(
                f"Only one covariate for shape motifs was found to be "\
                f"informative using LASSO regression. Calculating the scoring metric "\
                f"for a model with only an intercept and this covariate to "\
                f"compare to a model fit using only an intercept."
            )

            #bic_list = [ intercept_bic, shape_motifs.bic ]
            metric_list = [ intercept_metric, shape_motifs.metric ]
            model_list = [ intercept_fit, motif_fit ]

            best_mod_idx = evm.choose_model(
                bic_list,
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
                    f"informative shape motif. Not writing a shape motif to output. "\
                    f"Exiting now."
                )
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

        smv.plot_optim_shapes_and_weights(
            shape_motifs,
            final_motif_plot_fname,
            records,
        )

        # if there were both shape and seq motifs, combine into one model
        if shape_motif_exists and seq_motif_exists:

            shape_and_seq_motifs = shape_motifs.new_with_motifs(seq_motifs)
            shape_and_seq_motifs.motif_type = "shape_and_seq"

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
                shape_and_seq_coefs
            )

            print()
            logging.info(f"Number of final motifs: {len(shape_and_seq_motifs)}")

            # supplement motifs object with bic
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
                cores = args.p,
            )

            #shape_and_seq_motifs.bic = evm.get_sklearn_bic(
            #    intercept_and_shape_and_seq_X,
            #    records.y,
            #    int_and_shape_and_seq_fit,
            #)

            if len(shape_and_seq_motifs) == 1:
                print()
                logging.info(
                    f"Only one motif left after LASSO regression. "\
                    f"Performing model selection using F1 to determine whether "\
                    f"the remaining motif is informative over intercept alone."
                )
 
                #bic_list = [ intercept_bic, shape_and_seq_motifs.bic ]
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
        motifs_info.append((shape_and_seq_motifs, filtered_shape_and_seq_coefs))
        if args.write_all_files:
            out_fname = out_motif_basename + "_shape_and_sequence_motifs.dsm"
            shape_and_seq_motifs.write_file(out_fname, records)

    if not np.any([seq_motif_exists, shape_motif_exists]):
        print("No shape or sequence motifs found. Exiting now.")
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
        
    # place enrichment as key in each motif's dictionary
    best_motifs.get_enrichments(records)
    # write motifs to meme-like file
    best_motifs.write_file(out_motif_fname, records)
    with open(out_coefs_fname, "wb") as out_coef_f:
        np.save(out_coef_f, best_motif_coefs)
    logging.info(f"Writing motif enrichment heatmap to {out_heatmap_fname}")
    smv.plot_motif_enrichment(best_motifs, out_heatmap_fname, records)
    logging.info(f"Finished motif inference. Final results are in {out_motif_fname}")

