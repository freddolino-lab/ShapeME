import inout
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

def save_prc_plot(precision, recall, no_skill, plot_prefix):

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    plt.plot(recall, prec, marker='.', label='Shape motifs')
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(plot_prefix + ".png")
    plt.savefig(plot_prefix + ".pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--training_yvals_file', type=str, required=True,
            help="file containing ground truth y-values used for training")
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--data_dir', type=str, help="Directory containing data")
    parser.add_argument('--infile', type=str, help="File with peak names and y-vals")
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
    rust_out_fname = os.path.join(out_direc, 'test_rust_motifs.json')
    eval_out_fname = os.path.join(out_direc, 'precision_recall.json')
    prc_prefix = os.path.join(out_direc, 'precision_recall_curve')
    out_pref = args.o
    
    logit_reg_fname = os.path.join(
        out_direc,
        "{}_logistic_regression_result.pkl".format(out_pref).replace('test','train'),
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

    logging.info("Reading in files")
    # read in shapes
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(args.param_names, args.params)
    }
    logging.info("Reading input data and shape info.")
    records = inout.RecordDatabase(
        os.path.join(in_direc, args.infile),
        shape_fname_dict,
        shift_params = ["Roll", "HelT"],
    )
    records.shape_centers = np.asarray(args_dict['centers'])
    records.shape_spreads = np.asarray(args_dict['spreads'])
    logging.info("Number of records in test data: {}.".format(len(records)))

    logging.info("Normalizing parameters")
    records.normalize_shape_values()

    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        #records.read(args.infile, float)
        #logging.info("Discretizing data")
        #records.discretize_quant(args.continuous)
        #logging.info("Quantizing input data using k-means clustering")
        records.quantize_quant(args.continuous)

    # write shapes to npy file. Permute axes 1 and 2.
    with open(shape_fname, 'wb') as shape_f:
        np.save(shape_fname, records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, records.y.astype(np.int64))

    logging.info("Distribution of sequences per class:")
    logging.info(inout.seqs_per_bin(records))

    logging.info("Getting distance between motifs and each record")

    RUST = "{} {}".format(
        rust_bin,
        config_fname,
    )

    retcode = subprocess.call(RUST, shell=True, env=my_env)

    if retcode != 0:
        sys.exit("Rust binary returned non-zero exit status")

    test_motifs = inout.read_motifs_from_rust(
        os.path.join(out_direc, "evaluated_motifs.json")
    )
    test_X,var_lut = inout.prep_logit_reg_data(test_motifs, args_dict['max_count'])
    test_y = records.y

    train_motifs = inout.read_motifs_from_rust(
        os.path.join(out_direc, "rust_results.json")
    )
    train_X,var_lut = inout.prep_logit_reg_data(train_motifs, args_dict['max_count'])
    train_y = read_yvals(os.path.join(in_direc, args.training_yvals_file))

    row_n,col_n = test_X.shape
    test_X_r = ro.r.matrix(
        test_X,
        nrow=test_X.shape[0],
        ncol=test_X.shape[1],
    )
    test_y_r = ro.IntVector(test_y)

    train_X_r = ro.r.matrix(
        train_X,
        nrow=train_X.shape[0],
        ncol=train_X.shape[1],
    )
    train_y_r = ro.IntVector(train_y)

    # fit lasso regression, choosing best lambda using 10-fold CV
    shape_fit = glmnet.cv_glmnet(
        train_X_r,
        train_y_r,
        family="binomial",
        alpha=1,
        folds=10,
    )
    # NOTE: TODO: go through coefficients and weed out motifs for which all "match"
    #   coefficients are zero.
    # predict on test data
    # NOTE: needs updated for multiclass
    yhat = glmnet.predict_cv_glmnet(
        shape_fit,
        newx=test_X_r,
        s="lambda.1se",
    )
    no_skill = len(test_y[test_y==1]) / len(test_y)

    yhat_peaks = yhat[test_y==1]
    yhat_nonpeaks = yhat[test_y!=1]

    r_yhat_peaks = ro.FloatVector(yhat_peaks)
    r_yhat_nonpeaks = ro.FloatVector(yhat_nonpeaks)

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
        'shape_fit': shape_fit,
    }
    with open(eval_out_fname, 'w') as f:
        json.dump(output, f, indent=1)

    save_prc_plot(
        prec,
        recall,
        no_skill,
        prc_prefix,
    )

    logging.info("Done evaluating motifs on test data.")
    logging.info("Area under precision-recall curve: {}".format(auprc))
    logging.info("Expected area under precision-recall curve for random: {}".format(no_skill))
    logging.info("Results of evaluation are in {}".format(eval_out_fname))
    logging.info("Precision recall curve plotted. Saved as {} and {}".format(prc_prefix+".png", prc_prefix+".pdf"))

