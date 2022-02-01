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

from pathlib import Path

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/evaluate_motifs')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
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

    good_motifs = inout.read_motifs_from_rust(
        os.path.join(out_direc, "evaluated_motifs.json")
    )
    X,var_lut = inout.prep_logit_reg_data(good_motifs, args_dict['max_count'])
    y = records.y
    ##################################################
    ##################################################
    ## I need to figure out how to appropriately handle
    ##    strandedness and max count greater than 1
    ##################################################
    ##################################################
    logging.info(
        "Predicting classes from distance from motifs to records"
    )

    with open(logit_reg_fname, 'rb') as f:
        logit_reg_info = pickle.load(f)

    clf_f = logit_reg_info['model']
    var_lut = logit_reg_info['var_lut']

    yhat = clf_f.predict_proba(X)
    prec, recall, auc, no_skill = inout.get_precision_recall(
        yhat,
        y,
        plot_prefix=prc_prefix,
    )
    output = {
        'precision': list(prec),
        'recall': list(recall),
        'auc': auc,
        'random_auc': no_skill,
    }
    with open(eval_out_fname, 'w') as f:
        json.dump(output, f, indent=1)
    logging.info("Done evaluating motifs on test data")

