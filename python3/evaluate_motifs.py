import inout
import fimopytools as fimo
import logging
import argparse
import sys
import os
import json
import pickle
import numpy as np
import subprocess
from pprint import pprint
from matplotlib import pyplot as plt

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# import R's "PRROC" package
prroc = importr('PRROC')
glmnet = importr('glmnet')

from pathlib import Path

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/evaluate_motifs')

def read_yvals(fname):
    yvals = []
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('name'):
                continue
            elements = line.strip().split('\t')
            yvals.append(int(elements[1]))
    return(np.asarray(yvals))

def save_prc_plot(precision, recall, plot_prefix, plot_label, no_skill=None):

    if no_skill is not None:
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    plt.plot(recall, precision, marker='.', label=plot_label)
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([-0.02,1.02])
    plt.savefig(plot_prefix + ".png")
    plt.savefig(plot_prefix + ".pdf")
    plt.close()

def save_combined_prc_plot(results, plot_prefix):

    for i,(data_type,type_results) in enumerate(results.items()):
        if i == 0:
            plt.plot(
                [0,1],
                [type_results['random_auc'],type_results['random_auc']],
                linestyle='--',
                label="Random",
            )
        plt.plot(
            type_results['recall'],
            type_results['precision'],
            marker='.',
            label=data_type,
        )
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([-0.02, 1.02])
    plt.xlim([-0.02, 1.02])
    plt.savefig(plot_prefix + ".png")
    plt.savefig(plot_prefix + ".pdf")
    plt.close()

def get_X_and_y_from_motifs(fname, max_count, rec_db):
    motifs = inout.read_motifs_from_rust(fname)
    X,var_lut = inout.prep_logit_reg_data(motifs, max_count)
    y = rec_db.y
    return(X,y,var_lut)

def train_glmnet(X,y,folds=10,family='binomial',alpha=1):
    X_r = ro.r.matrix(
        X,
        nrow=X.shape[0],
        ncol=X.shape[1],
    )
    y_r = ro.IntVector(y)

    # fit lasso regression, choosing best lambda using 10-fold CV
    fit = glmnet.cv_glmnet(
        X_r,
        y_r,
        family=family,
        alpha=alpha,
        folds=folds,
    )
    return fit

def evaluate_fit(fit, test_X, test_y, lambda_cut="lambda.1se"):

    test_X_r = ro.r.matrix(
        test_X,
        nrow=test_X.shape[0],
        ncol=test_X.shape[1],
    )

    yhat = glmnet.predict_cv_glmnet(
        fit,
        newx=test_X_r,
        s=lambda_cut,
    )  
    print("yhat: {}".format(yhat))
    print(yhat.shape)
    no_skill = len(test_y[test_y==1]) / len(test_y)

    yhat_peaks = yhat[test_y==1]
    yhat_nonpeaks = yhat[test_y!=1]

    r_yhat_peaks = ro.FloatVector(yhat_peaks)
    r_yhat_nonpeaks = ro.FloatVector(yhat_nonpeaks)

    print(yhat_peaks.shape)
    print(yhat_nonpeaks.shape)

    auc = prroc.pr_curve(
        scores_class0 = r_yhat_peaks,
        scores_class1 = r_yhat_nonpeaks,
        curve=True,
    )

    auprc = auc.rx2['auc.davis.goadrich'][0]
    prec = auc.rx2['curve'][:,1]
    recall = auc.rx2['curve'][:,0]
    thresholds = auc.rx2['curve'][:,2]

    output = {
        'precision': list(prec),
        'recall': list(recall),
        'logit_threshold': list(thresholds),
        'auc': auprc,
        'random_auc': no_skill,
    }

    return output

def read_records(args_dict, in_direc, infile, param_names, param_files, continuous=None, dset_type="training"):

    print("Infile: {}".format(infile))
    logging.info("Reading in files")
    # read in shapes
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(param_names, param_files)
    }
    logging.info("Reading input data and shape info.")
    records = inout.RecordDatabase(
        os.path.join(in_direc, infile),
        shape_fname_dict,
        shift_params = ["Roll", "HelT"],
    )
    records.shape_centers = np.asarray(args_dict['centers'])
    records.shape_spreads = np.asarray(args_dict['spreads'])
    logging.info("Number of records in {} data: {}.".format(dset_type, len(records)))

    logging.info("Normalizing parameters")
    records.normalize_shape_values()

    # read in the values associated with each sequence and store them
    # in the sequence database
    if continuous is not None:
        records.quantize_quant(continuous)

    return records



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--test_fimo_file', type=str, default=None,
            help="full path to tsv file containing fimo output for a sequence motif matched on held-out test data")
    parser.add_argument('--train_fimo_file', type=str, default=None,
            help="full path to tsv file containing fimo output for a sequence motif matched on training data")
    parser.add_argument('--test_params', nargs="+", type=str,
                         help='inputfiles with test shape scores')
    parser.add_argument('--train_params', nargs="+", type=str,
                         help='inputfiles with training shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--data_dir', type=str, help="Directory containing data")
    parser.add_argument('--train_infile', type=str, help="File with peak names and y-vals")
    parser.add_argument('--test_infile', type=str, help="File with peak names and y-vals")
    parser.add_argument('--out_dir', type=str, help="Directory to which to write outputs")
    parser.add_argument('-p', type=int, help="Number of cores to run in parallel")
    parser.add_argument('-o', type=str, help="Prefix to prepent to output files.")

    level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level, stream=sys.stdout) 
    logging.getLogger('matplotlib.font_manager').disabled = True

    my_env = os.environ.copy()
    my_env['RUST_BACKTRACE'] = "1"
    
    logging.warning("Reading in files")

    args = parser.parse_args()

    in_direc = args.data_dir

    logging.warning("Reading in files")

    args = parser.parse_args()

    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    shape_fname = os.path.join(out_direc, 'test_shapes.npy')
    yval_fname = os.path.join(out_direc, 'test_y_vals.npy')
    config_fname = os.path.join(out_direc, 'config.json')
    rust_motifs_fname = os.path.join(out_direc, 'rust_results.json')
    eval_out_fname = os.path.join(out_direc, 'shape_precision_recall.json')
    seq_eval_out_fname = os.path.join(out_direc, 'seq_precision_recall.json')
    seq_and_shape_eval_out_fname = os.path.join(out_direc, 'seq_and_shape_precision_recall.json')
    prc_prefix = os.path.join(out_direc, 'shape_precision_recall_curve')
    seq_prc_prefix = os.path.join(out_direc, 'seq_precision_recall_curve')
    seq_and_shape_prc_prefix = os.path.join(out_direc, 'seq_and_shape_precision_recall_curve')
    combined_plot_prefix = os.path.join(out_direc, 'combined_precision_recall_curve')
    out_pref = args.o
    
    logit_reg_str = "{}_{}_logistic_regression_result.pkl"
    shape_logit_reg_fname = os.path.join(
        out_direc,
        logit_reg_str.format(out_pref.replace('test','train'), "shape"),
    )

    with open(config_fname, 'r') as f:
        args_dict = json.load(f)

    # update cores so rust gets the right number for this job
    args_dict['cores'] = args.p
    args_dict['eval_shape_fname'] = shape_fname
    args_dict['eval_yvals_fname'] = yval_fname

    with open(config_fname, 'w') as f:
        json.dump(args_dict, f, indent=1)

    pprint(args_dict)

    test_records = read_records(
        args_dict,
        in_direc,
        args.test_infile,
        args.param_names,
        args.test_params,
        continuous=args.continuous,
        dset_type="test",
    )

    # write shapes to npy file. Permute axes 1 and 2.
    with open(shape_fname, 'wb') as shape_f:
        np.save(shape_fname, test_records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, test_records.y.astype(np.int64))

    logging.info("Distribution of sequences per class:")
    logging.info(inout.seqs_per_bin(test_records))

    logging.info("Getting distance between motifs and each record")

    train_records = read_records(
        args_dict,
        in_direc,
        args.train_infile,
        args.param_names,
        args.train_params,
        continuous=args.continuous,
        dset_type="training",
    )

    if os.path.isfile(rust_motifs_fname):
        RUST = "{} {}".format(
            rust_bin,
            config_fname,
        )

        retcode = subprocess.call(RUST, shell=True, env=my_env)

        if retcode != 0:
            sys.exit("Rust binary returned non-zero exit status")

        test_X,test_y,var_lut = get_X_and_y_from_motifs(
            os.path.join(out_direc, "evaluated_motifs.json"),
            args_dict['max_count'],
            test_records,
        )

        train_X,train_y,_ = get_X_and_y_from_motifs(
            os.path.join(out_direc, "rust_results.json"),
            args_dict['max_count'],
            train_records,
        )

        shape_fit = train_glmnet(
            train_X,
            train_y,
            folds=10,
            family='binomial',
            alpha=1,
        )

        # NOTE: TODO: go through coefficients and weed out motifs for which all "match"
        #   coefficients are zero.
        # predict on test data
        # NOTE: needs updated for multiclass
        shape_output = evaluate_fit(
            shape_fit,
            test_X,
            test_y,
            lambda_cut="lambda.1se",
        )

        with open(eval_out_fname, 'w') as f:
            json.dump(shape_output, f, indent=1)

        with open(shape_logit_reg_fname, 'wb') as f:
            pickle.dump(shape_fit, f)

        save_prc_plot(
            shape_output['precision'],
            shape_output['recall'],
            prc_prefix,
            "Shape motifs",
            shape_output['random_auc'],
        )

        logging.info("Done evaluating shape motifs on test data.")
        logging.info("==========================================")
        logging.info("Area under precision-recall curve: {}".format(shape_output['auc']))
        logging.info(
            "Expected area under precision-recall curve for random: {}".format(shape_output['random_auc'])
        )
        logging.info("Results of evaluation are in {}".format(eval_out_fname))
        logging.info(
            "Precision recall curve plotted. Saved as {} and {}".format(
                prc_prefix+".png",
                prc_prefix+".pdf",
            )
        )

    if args.test_fimo_file is not None:

        
        train_seq_matches = fimo.FimoFile()
        train_seq_matches.parse(args.train_fimo_file)
        train_seq_X = train_seq_matches.get_design_matrix(train_records)

        seq_fit = train_glmnet(
            train_seq_X,
            train_y,
            folds=10,
            family='binomial',
            alpha=1,
        )

        test_seq_matches = fimo.FimoFile()
        test_seq_matches.parse(args.test_fimo_file)
        test_seq_X = test_seq_matches.get_design_matrix(test_records)

        seq_output = evaluate_fit(
            seq_fit,
            test_seq_X,
            test_y,
            lambda_cut="lambda.1se",
        )
        seq_logit_reg_fname = os.path.join(
            out_direc,
            logit_reg_str.format(out_pref.replace('test','train'), "sequence"),
        )

        save_prc_plot(
            seq_output['precision'],
            seq_output['recall'],
            seq_prc_prefix,
            "Sequence motif",
            seq_output['random_auc'],
        )

        with open(seq_eval_out_fname, 'w') as f:
            json.dump(seq_output, f, indent=1)

        logging.info("Done evaluating sequence motifs on test data.")
        logging.info("==========================================")
        logging.info("Area under precision-recall curve: {}".format(seq_output['auc']))
        logging.info(
            "Expected area under precision-recall curve for random: {}".format(seq_output['random_auc'])
        )
        logging.info("Results of evaluation are in {}".format(seq_eval_out_fname))
        logging.info(
            "Precision recall curve plotted. Saved as {} and {}".format(
                seq_prc_prefix+".png",
                seq_prc_prefix+".pdf",
            )
        )

        if os.path.isfile(rust_motifs_fname):
            train_seq_and_shape_X = np.append(train_X, train_seq_X, axis=1)
            test_seq_and_shape_X = np.append(test_X, test_seq_X, axis=1)

            seq_and_shape_fit = train_glmnet(
                train_seq_and_shape_X,
                train_y,
                folds=10,
                family='binomial',
                alpha=1,
            )
            seq_and_shape_output = evaluate_fit(
                seq_and_shape_fit,
                test_seq_and_shape_X,
                test_y,
                lambda_cut="lambda.1se",
            )
            seq_shape_logit_reg_fname = os.path.join(
                out_direc,
                logit_reg_str.format(out_pref.replace('test','train'), "sequence_and_shape"),
            )

            save_prc_plot(
                seq_and_shape_output['precision'],
                seq_and_shape_output['recall'],
                seq_and_shape_prc_prefix,
                "Sequence and shape motifs",
                seq_and_shape_output['random_auc'],
            )

            combined_results = {
                'Shape':shape_output,
                'Sequence':seq_output,
                'Shape and sequence':seq_and_shape_output,
            }

            save_combined_prc_plot(
                combined_results,
                combined_plot_prefix,
            )

            with open(seq_and_shape_eval_out_fname, 'w') as f:
                json.dump(seq_and_shape_output, f, indent=1)

            logging.info("Done evaluating sequence and shape motifs together on test data.")
            logging.info("==========================================")
            logging.info("Area under precision-recall curve: {}".format(seq_and_shape_output['auc']))
            logging.info(
                "Expected area under precision-recall curve for random: {}".format(seq_and_shape_output['random_auc'])
            )
            logging.info("Results of evaluation are in {}".format(seq_and_shape_eval_out_fname))
            logging.info(
                "Precision recall curve plotted. Saved as {} and {}".format(
                    seq_and_shape_prc_prefix+".png",
                    seq_and_shape_prc_prefix+".pdf",
                )
            )
