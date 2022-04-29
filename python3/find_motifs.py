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
rust_bin = os.path.join(this_path, '../rust_utils/target/release/find_motifs')

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


def info_zscore(vec1, vec2, n=10000):
    """ Similar to FIRE determine a Z score for vec1 based on MI scores
    from randomly shuffled vec2

    Args:
        vec1 - vector to not shuffle
        vec2 -vector to shuffle
        n - number of shuffles to do
    
    Returns:
        zscore - (MI_actual - mean MI_shuff)/std
        passed - was MI_actual greater than all shuffles?
    """
    # doing welford's again
    online_mean = welfords.Welford()
    passed = True
    actual = inout.mutual_information(vec1, vec2)
    for i in range(n):
        shuffle = np.random.permutation(len(vec2))
        newval = inout.mutual_information(vec1, vec2[shuffle])
        online_mean.update(newval)
        if newval >= actual:
            passed = False
    mean = online_mean.final_mean()
    stdev = online_mean.final_stdev()
    zscore = (actual-mean)/stdev
    return zscore, passed


def info_robustness(vec1, vec2, n=10000, r=10, holdout_frac=0.3):
    """ Similar to FIRE Robustness score, calculate Z score for
    jacknife replicates and report number that pass

    Args:
        vec1 - vector to not shuffle
        vec2 -vector to shuffle
        n - number of shuffles to do for zscore
        r - number of jackknife replicates
        holdout_frac - fraction of data to remove for jackknife
    
    Returns:
        num_passed - number of jacknife reps that passed
    """
    num_passed = 0
    num_to_use = int(np.floor((1-holdout_frac)*len(vec1)))
    for i in range(r):
        jk_selector = np.random.permutation(len(vec1))[0:num_to_use]
        zscore, passed = info_zscore(vec1[jk_selector], vec2[jk_selector], n=n)
        if passed:
            num_passed += 1
    return num_passed



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
    logging.info(str(args))

    out_pref = args.o
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)
    in_fname = os.path.join(in_direc, args.infile)
    find_seq_motifs = args.find_seq_motifs
    seq_fasta = args.seq_fasta
    if seq_fasta is not None:
        seq_fasta = os.path.join(in_direc, seq_fasta)
    known_motif_file = args.seq_meme_file
    if known_motif_file is not None:
        known_motif_file = os.path.join(in_direc, known_motif_file)
    fimo_direc = f"{out_direc}/fimo_out"
    streme_direc = f"{out_direc}/streme_out"
    streme_thresh = args.streme_thresh

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    if find_seq_motifs:
        # if asked for seq motifs but didn't pass seq fa file, exception
        if seq_fasta is None:
            raise inout.NoSeqFaException()
        # if both seq_motifs and meme file were passed, raise exception
        if known_motif_file is not None:
            raise inout.SeqMotifOptionException(known_motif_file)

        known_motif_file = f"{out_direc}/streme_out/streme.txt"

        STREME = f"run_streme.py "\
            f"--seq_fname {seq_fasta} "\
            f"--yvals_fname {in_fname} "\
            f"--threshold {streme_thresh} "\
            f"--out_direc {streme_direc}"

        streme_result = subprocess.run(
            STREME,
            shell=True,
            check=True,
            capture_output=True,
        )
        streme_log_fname = f"{streme_direc}/streme_run.log"
        streme_err_fname = f"{streme_direc}/streme_run.err"
        logging.info(
            f"Ran streme: for details, see "\
            f"{streme_log_fname} and {streme_err_fname}"
        )
        with open(streme_log_fname, "w") as streme_out:
            # streme log gets captured as stderr, so write stderr to file
            streme_out.write(streme_result.stdout)
        with open(streme_err_fname, "w") as streme_err:
            # streme log gets captured as stderr, so write stderr to file
            streme_err.write(streme_result.stderr)

    # if user has a meme file (could be from streme above, or from input arg), run fimo
    if known_motif_file is not None:

        if seq_fasta is None:
            raise inout.NoSeqFaException()

        FIMO = f"run_fimo.py "\
            f"--seq_fname {seq_fasta} "\
            f"--meme_file {known_motif_file} "\
            f"--out_direc {fimo_direc}"

        fimo_result = subprocess.run(
            FIMO,
            shell=True,
            check=True,
            capture_output=True,
        )
        fimo_log_fname = f"{fimo_direc}/fimo_run.log"
        fimo_err_fname = f"{fimo_direc}/fimo_run.err"
        logging.info(
            f"Ran fimo: for details, see "\
            f"{fimo_log_fname} and {fimo_err_fname}"
        )
        with open(fimo_log_fname, "w") as fimo_out:
            fimo_out.write(fimo_result.stdout)
        with open(fimo_err_fname, "w") as fimo_err:
            fimo_err.write(fimo_result.stderr)

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
    assert len(records.y) == records.X.shape[0], "Number of y values does not equal number of shape records!!"
           
    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        #records.read(args.infile, float)
        #logging.info("Discretizing data")
        #records.discretize_quant(args.continuous)
        #logging.info("Quantizing input data using k-means clustering")
        records.quantize_quant(args.continuous)

    fam = evm.set_family(records.y)
    ##############################################################################
    ##############################################################################
    ## incorporate test data if at CLI ###########################################
    ##############################################################################
    ##############################################################################

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
        logging.info(
            "{}: center={}, spread={}".format(
                name,
                this_center,
                this_spread
            )
        )

    alpha = args.alpha
    max_count = args.max_count

    temp = args.temperature
    step = args.stepsize
    
    mi_fname = os.path.join(
        out_direc,
        '{}_initial_mutual_information_max_count_{}.pkl'.format(
            out_pref,
            max_count,
        ),
    )

    shape_fname = os.path.join(out_direc, 'shapes.npy')
    yval_fname = os.path.join(out_direc, 'y_vals.npy')
    config_fname = os.path.join(out_direc, 'config.json')
    rust_out_fname = os.path.join(out_direc, 'rust_results.json')
    shape_fit_fname = os.path.join(out_direc, 'shape_lasso_fit.pkl')
    seq_fit_fname = os.path.join(out_direc, 'seq_lasso_fit.pkl')
    shape_and_seq_fit_fname = os.path.join(out_direc, 'shape_and_seq_lasso_fit.pkl')

    # get the BIC for an intercept-only model, which will ultimately be compared
    # to the BIC we get from any other fit to choose whether there is a motif or not.
    intercept_X = np.ones((len(records), 1))
    intercept_fit = evm.train_sklearn_glm(
        intercept_X,
        records.y,
        family = fam,
        fit_intercept = False,
    )

    intercept_bic = evm.get_sklearn_bic(
        intercept_X,
        records.y,
        intercept_fit,
    )

    if find_seq_motifs:

        print()
        logging.info("Fitting regression model to sequence motifs")
        seq_matches = fimo.FimoFile()
        seq_matches.parse(f"{fimo_direc}/fimo.tsv")
        seq_motifs = seq_matches.get_list()

        seq_X,seq_var_lut = seq_matches.get_design_matrix(
            records,
            # should I be filtering by p-value of fimo match here? I think not.
            #streme_thresh,
        )

        one_seq_motif = False
        
        if len(seq_motifs) == 0:
            logging.info(f"No motifs passed significance "\
                f"threshold of {streme_thresh}, setting find_seq_motifs "\
                f"back to False and moving on to shape motif inference.")
            find_seq_motifs = False
 
        elif len(seq_motifs) == 1:

            logging.info(
                f"Only one sequence motif present. "\
                f"Performing model selection using BIC to determine whether "\
                f"the motif is informative over intercept alone."
            )
            # toggle one_seq_motif to True for later use in building combined
            # seq and shape motif design matrix
            one_seq_motif = True

        else:

            seq_fit = evm.train_glmnet(
                seq_X,
                records.y,
                folds=10,
                family=fam,
                alpha=1,
            )

            seq_coefs = evm.fetch_coefficients(fam, seq_fit, args.continuous)

            print()
            logging.info(f"Sequence motif coefficients:\n{seq_coefs}")

            # clobber seq_X and seq_var_lut here
            seq_motifs,seq_X,seq_var_lut = evm.filter_motifs(
                seq_motifs,
                seq_X,
                seq_coefs,
                seq_var_lut,
            )

            logging.info(
                f"Number of shape motifs left after LASSO regression: "\
                f"{len(final_seq_motifs)}"
            )

            if len(final_seq_motifs) == 1:
                logging.info(
                    f"Only one sequence motif left after LASSO regression. "\
                    f"Performing model selection using BIC to determine whether "\
                    f"the remaining motif is informative over intercept alone."
                )
                one_seq_motif = True
 

        # if there's only one covariate, create intercept-only and intercept
        # plus motif design matrices
        if one_seq_motif:
            intercept_and_motif_X = np.append(intercept_X, seq_X, axis=1)

            motif_fit = evm.train_sklearn_glm(
                intercept_and_motif_X,
                records.y,
                family = fam,
                fit_intercept = False, # intercept already in design mat
            )

            int_and_motif_bic = evm.get_sklearn_bic(
                intercept_and_motif_X,
                records.y,
                motif_fit,
            )

            bic_list = [ intercept_bic, int_and_motif_bic ]
            model_list = [ intercept_fit, motif_fit ]

            best_mod_idx = evm.choose_model(
                bic_list,
                model_list,
                return_index = True,
            )

            if best_mod_idx == 0:
                logging.info(
                    f"Intercept-only model had lower BIC than model fit using "\
                    f"intercept and one sequence motif. Therefore, there is no "\
                    f"informative "\
                    f"sequence motif. Not writing a sequence motif to output."
                )
            
    
    good_motif_out_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_temp_{}_stepsize_{}_alpha_{}_max_count_{}.pkl".format(
            out_pref,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    final_motif_plot_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_temp_{}_stepsize_{}_alpha_{}_max_count_{}.png".format(
            out_pref,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    logit_reg_fname = os.path.join(
        out_direc,
        "{}_logistic_regression_result.pkl".format(out_pref),
    )

    coef_per_class_fname = os.path.join(
        out_direc,
        "{}_logistic_regression_coefs_per_class.txt".format(out_pref),
    )

    RUST = "{} {}".format(
        rust_bin,
        config_fname,
    )

    args_dict = {
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
        'good_motif_out_fname': good_motif_out_fname, 
    }

    if args.continuous is not None:
        args_dict['y_cat_num'] = args.continuous

    # supplement args info with shape center and spread from database
    args_dict['names'] = []
    args_dict['indices'] = []
    args_dict['centers'] = []
    args_dict['spreads'] = []

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        args_dict['names'].append(name)
        args_dict['indices'].append(shape_idx)
        args_dict['centers'].append(this_center)
        args_dict['spreads'].append(this_spread)
    
    # write shapes to npy file. Permute axes 1 and 2.
    with open(shape_fname, 'wb') as shape_f:
        np.save(shape_fname, records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, records.y.astype(np.int64))
    # write cfg to file
    with open(config_fname, 'w') as f:
        json.dump(args_dict, f, indent=1)

    logging.info("Running motif selection and optimization.")
    retcode = subprocess.call(RUST, shell=True, env=my_env)
    if retcode != 0:
        raise inout.RustBinaryException(Exception)

    if not os.path.isfile(rust_out_fname):
        info.warning(
            f"No output json file containing motifs from rust binary. "\
            f"This usually means no motifs were identified, but you should "\
            f"carfully check your log and error messages to make sure that's "\
            f"really the case."
        )
        sys.exit()

    good_motifs = inout.read_motifs_from_rust(rust_out_fname)

    shape_X,shape_var_lut = evm.get_X_from_motifs(
        rust_out_fname,
        args.max_count,
    )

    shape_fit = evm.train_glmnet(
        shape_X,
        records.y,
        folds = 10,
        family=fam,
        alpha=1,
    )

    with open(shape_fit_fname, "wb") as f:
        pickle.dump(shape_fit, f)

    coefs = evm.fetch_coefficients(fam, shape_fit, args.continuous)
    print()
    logging.info(f"Shape motif coefficients:\n{coefs}")

    # go through coefficients and weed out motifs for which all
    #   hits' coefficients are zero.
    final_motifs,final_X,final_var_lut = evm.filter_motifs(
        good_motifs,
        shape_X,
        coefs,
        shape_var_lut,
    )

    logging.info(
        f"Number of shape motifs left after LASSO regression: "\
        f"{len(final_motifs)}"
    )
   
    # check whether there's only one informative coefficient
    if final_X.shape[1] == 1:
        logging.info(
            f"Only one covariate for shape motifs was found to be "\
            f"informative using LASSO regression. Calculating the BIC "\
            f"for a model with only an intercept and this covariate to "\
            f"compare to a model fit using only an intercept."
        )

        intercept_and_shape_X = np.append(intercept_X, final_X, axis=1)

        motif_fit = evm.train_sklearn_glm(
            intercept_and_shape_X,
            records.y,
            family = fam,
            fit_intercept = False, # intercept already in design mat
        )

        int_and_motif_bic = evm.get_sklearn_bic(
            intercept_and_shape_X,
            records.y,
            motif_fit,
        )

        bic_list = [ intercept_bic, int_and_motif_bic ]
        model_list = [ intercept_fit, motif_fit ]

        best_mod_idx = evm.choose_model(
            bic_list,
            model_list,
            return_index = True,
        )

        if best_mod_idx == 0:
            logging.info(
                f"Intercept-only model had lower BIC than model fit using "\
                f"intercept and one shape covariate. Therefore, there is no "\
                f"informative shape motif. Not writing a shape motif to output. "\
                f"Exiting now."
            )
            sys.exit()

    if len(final_motifs) == 0:
        info.warning(
            f"There were no shape motifs left after LASSO regression. "\
            f"Exiting now."
        )
        sys.exit()

    smv.plot_optim_shapes_and_weights(
        final_motifs,
        final_motif_plot_fname,
        records,
    )

    if find_seq_motifs:
        # if there was only one seq motif, grab it from seq_X
        if one_seq_motif:
            seq_X = seq_X[:,0]
        shape_and_seq_X = np.append(shape_X, seq_X, axis=1)
        shape_and_seq_var_lut = shape_var_lut
        for seq_motif_key,seq_motif_info in seq_var_lut.items():
            new_key = seq_motif_key + shape_X.shape[1]
            seq_motif_info['motif_idx'] = new_key
            shape_and_seq_var_lut[new_key] = seq_motif_info

        shape_and_seq_fit = evm.train_glmnet(
            shape_and_seq_X,
            records.y,
            folds=10,
            family=fam,
            alpha=1,
        )
        with open(shape_and_seq_fit_fname, "wb") as f:
            pickle.dump(shape_and_seq_fit, f)

        shape_and_seq_motifs = good_motifs.copy()
        shape_and_seq_motifs.extend(seq_motifs.copy())

        shape_and_seq_coefs = evm.fetch_coefficients(
            fam,
            shape_and_seq_fit,
            args.continuous,
        )

        print()
        logging.info(f"Shape and sequence motif coefficients:\n{shape_and_seq_coefs}")

        (
            final_shape_and_seq_motifs,
            final_shape_and_seq_X,
            final_shape_and_seq_lut,
        ) = evm.filter_motifs(
            shape_and_seq_motifs,
            shape_and_seq_X,
            shape_and_seq_coefs,
            shape_and_seq_var_lut,
        )
        logging.info(f"Number of final motifs: {len(final_shape_and_seq_motifs)}")
 

    #good_motif_index = cvlogistic.choose_features(clf_f, tol=0)
    #if len(good_motif_index) < 1:
    #    logging.info("No motifs found")
    #    sys.exit()

    #cvlogistic.write_coef_per_class(clf_f, coef_per_class_fname)
    #final_good_motifs = [good_motifs[index] for index in good_motif_index]
    #logging.info("{} motifs survived".format(len(final_good_motifs)))

    #for motif in final_good_motifs:
    #    add_motif_metadata(this_records, motif) 
    #    logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
    #    logging.info("MI: {}".format(motif['mi']))
    #    logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
    #    logging.info("Category Entropy: {}".format(motif['category_entropy']))
    #    for key in sorted(motif['enrichment'].keys()):
    #        logging.info("Two way table for cat {} is {}".format(
    #            key,
    #            motif['enrichment'][key]
    #        ))
    #        logging.info("Enrichment for Cat {} is {}".format(
    #            key,
    #            two_way_to_log_odds(motif['enrichment'][key])
    #        ))
    #logging.info("Generating initial heatmap for passing motifs")
    #if len(final_good_motifs) > 25:
    #    logging.info("Only plotting first 25 motifs")
    #    enrich_hm = smv.EnrichmentHeatmap(final_good_motifs[:25])
    #else:
    #    enrich_hm = smv.EnrichmentHeatmap(final_good_motifs)

    #enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_before_hm.txt")
    #if not args.txt_only:
    #    enrich_hm.display_enrichment(outpre+"_enrichment_before_hm.pdf")
    #    enrich_hm.display_motifs(outpre+"motif_before_hm.pdf")
    #if args.optimize:
    #    logging.info("Optimizing motifs using {} processors".format(args.p))
    #    final_motifs = mp_optimize_motifs(
    #        final_good_motifs,
    #        other_records,
    #        args.optimize_perc,
    #        p=args.p,
    #    )
    #    if args.optimize_perc != 1:
    #        logging.info("Testing final optimized motifs on full database")
    #        for i,this_entry in enumerate(final_motifs):
    #            logging.info("Computing MI for motif {}".format(i))
    #            this_discrete = generate_peak_vector(
    #                other_records,
    #                this_entry['motif'],
    #                this_entry['threshold'],
    #                args.rc,
    #            )
    #            this_entry['mi'] = other_records.mutual_information(this_discrete)
    #            this_entry['motif_entropy'] = inout.entropy(this_discrete)
    #            this_entry['category_entropy'] = other_records.shannon_entropy()
    #            this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
    #            this_entry['discrete'] = this_discrete
    #else:
    #    if args.motif_perc != 1:
    #        logging.info("Testing final optimized motifs on held out database")
    #        for i,this_entry in enumerate(final_good_motifs):
    #            logging.info("Computing MI for motif {}".format(i))
    #            this_discrete = generate_peak_vector(
    #                other_records,
    #                this_entry['motif'],
    #                this_entry['threshold'],
    #                args.rc,
    #            )
    #            this_entry['mi'] = other_records.mutual_information(this_discrete)
    #            this_entry['motif_entropy'] = inout.entropy(this_discrete)
    #            this_entry['category_entropy'] = other_records.shannon_entropy()
    #            this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
    #            this_entry['discrete'] = this_discrete
    #    final_motifs = final_good_motifs

    #logging.info(
    #    "Filtering motifs by Conditional MI using {} as a cutoff".format(args.mi_perc)
    #)
    #novel_motifs = filter_motifs(
    #    final_motifs,
    #    other_records,
    #    args.mi_perc,
    #)

    #if args.debug:
    #    print_top_motifs(novel_motifs)
    #    print_top_motifs(novel_motifs, reverse=False)
    #logging.info("{} motifs survived".format(len(novel_motifs)))
    #for i, motif in enumerate(novel_motifs):
    #    logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
    #    logging.info("MI: {}".format(motif['mi']))
    #    if args.infoz > 0:
    #        logging.info("Calculating Z-score for motif {}".format(i))
    #        # calculate zscore
    #        zscore, passed = info_zscore(
    #            motif['discrete'],
    #            other_records.get_values(),
    #            args.infoz,
    #        )
    #        motif['zscore'] = zscore
    #        logging.info("Z-score: {}".format(motif['zscore']))
    #    if args.infoz > 0 and args.inforobust > 0:
    #        logging.info("Calculating Robustness for motif {}".format(i))
    #        num_passed = info_robustness(
    #            motif['discrete'],
    #            other_records.get_values(), 
    #            args.infoz,
    #            args.inforobust,
    #            args.fracjack,
    #        )
    #        motif['robustness'] = "{}/{}".format(num_passed,args.inforobust)
    #        logging.info("Robustness: {}".format(motif['robustness']))
    #    logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
    #    logging.info("Category Entropy: {}".format(motif['category_entropy']))
    #    for key in sorted(motif['enrichment'].keys()):
    #        logging.info("Two way table for cat {} is {}".format(
    #            key,
    #            motif['enrichment'][key]
    #        ))
    #        logging.info("Enrichment for Cat {} is {}".format(
    #            key,
    #            two_way_to_log_odds(motif['enrichment'][key])
    #        ))
    #    if args.optimize:
    #        logging.info("Optimize Success?: {}".format(motif['opt_success']))
    #        logging.info("Optimize Message: {}".format(motif['opt_message']))
    #        logging.info("Optimize Iterations: {}".format(motif['opt_iter']))
    #logging.info("Generating final heatmap for motifs")
    #enrich_hm = smv.EnrichmentHeatmap(novel_motifs)
    #enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_after_hm.txt")

    #if not args.txt_only:
    #    enrich_hm.display_enrichment(outpre+"_enrichment_after_hm.pdf")
    #    enrich_hm.display_motifs(outpre+"_motif_after_hm.pdf")
    #    if args.optimize:
    #        logging.info("Plotting optimization for final motifs")
    #        enrich_hm.plot_optimization(outpre+"_optimization.pdf")

    #logging.info("Writing final motifs")
    #outmotifs = inout.ShapeMotifFile()
    #outmotifs.add_motifs(novel_motifs)
    #outmotifs.write_file(outpre+"_called_motifs.dsp", records)
    #final = opt.minimize(lambda x: -optimize_mi(x, data=records, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=records), motif_to_optimize)
    #logging.info(final)
