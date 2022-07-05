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
import operator
from pprint import pprint
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from statsmodels.stats import rates
from scipy.stats import contingency

import seaborn as sns
import pandas as pd

from rpy2.robjects.functions import SignatureTranslatedFunction as STM
#STM = SignatureTranslatedFunction

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
numpy2ri.activate()
# import R's "PRROC" package
prroc = importr('PRROC')
glmnet = importr('glmnet')
base = importr('base',  robject_translations={'as.data.frame': 'as_data_frame'})
#brms = importr('brms')
#stats = importr('stats', robject_translations={'as.formula': 'as_formula'})

glmnet.glmnet = STM(glmnet.glmnet, init_prm_translate = {"lam": "lambda"})
#brms.set_prior = STM(brms.set_prior, init_prm_translate = {"cl": "class"})

from pathlib import Path

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/evaluate_motifs')

#EPSILON = sys.float_info.epsilon



#def get_log_enrichments(motif_list):
#    for motif in motif_list:
#        motif["fitted_log_enrichment"] = get_log_enrichment(
#            motif["contingency_fit"],
#            motif["null_fit"],
#        )
#
#
#def get_log_enrichment(contingency_fit, null_fit):
#
#    contingency_sample_fitted = stats.fitted(contingency_fit, summary=False)
#    null_sample_fitted = stats.fitted(null_fit, summary=False)
#    rel_enrichment = contingency_sample_fitted / null_sample_fitted
#    return np.log2(rel_enrichment)
#
#
#def get_evid_ratio(arr, oper, compare_val=0):
#    num_pass = len(np.where(oper(arr, compare_val)[0]))
#    num_fail = len(arr) - num_pass
#    pass


def fetch_coefficients(family, fit, continuous):
    if family == "multinomial":
        coefs = fetch_coefficients_multinomial(fit, continuous)
    else:
        coefs = fetch_coefficients_binomial(fit)
    return coefs


def read_yvals(fname):
    yvals = []
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('name'):
                continue
            elements = line.strip().split('\t')
            yvals.append(int(elements[1]))
    return(np.asarray(yvals))


def get_y_axis_limits(prc_dict):

    max_val = 0.0
    for class_name,class_info in prc_dict.items():
        this_max = np.max(class_info['precision'])
        if this_max > max_val:
            max_val = this_max
    expand = max_val * 0.02
    min_val = -expand
    max_val += expand
    return (min_val, max_val)

def save_prc_plot(prc_dict, plot_prefix, plot_label_prefix):

    label_str = plot_label_prefix + "_class: {}"

    ylims = get_y_axis_limits(prc_dict)

    for class_name,class_info in prc_dict.items():

        #no_skill = class_info['random_auc']
        recall = class_info['recall']
        precision = class_info['precision']
        
        #if no_skill is not None:
        #    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
        plt.plot(recall, precision, marker='.', label=label_str.format(class_name))

    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(ylims)
    plt.savefig(plot_prefix + ".png")
    plt.savefig(plot_prefix + ".pdf")
    plt.close()

def save_combined_prc_plot(results, plot_prefix):

    ylim_vals = []
    for i,(data_type,type_results) in enumerate(results.items()):
        ylim_vals.append(get_y_axis_limits(type_results))
        #if i == 0:
        #    plt.plot(
        #        [0,1],
        #        [type_results['random_auc'],type_results['random_auc']],
        #        linestyle='--',
        #        label="Random",
        #    )
        for class_name,class_info in type_results.items():
            plt.plot(
                type_results['recall'],
                type_results['precision'],
                marker='.',
                label=data_type + "_class: {}".format(class_name),
            )
    max_val = np.max([_[1] for _ in ylim_vals])
    min_val = np.min([_[0] for _ in ylim_vals])
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([min_val, max_val])
    plt.xlim([-0.02, 1.02])
    plt.savefig(plot_prefix + ".png")
    plt.savefig(plot_prefix + ".pdf")
    plt.close()


def get_X_from_lut(fname, lut):
    motifs = inout.read_motifs_from_rust(fname)
    X = np.zeros((motifs[0]['hits'].shape[0], len(lut)), dtype='uint8')
    for col_idx,col_info in lut.items():
        motif = motifs[col_info['motif_idx']]
        hit_cat = col_info['hits']
        rows = np.all(motif['hits'] == hit_cat, axis=1)
        X[rows,col_idx] = 1

    return X


def log_likelihood(truth, preds):
    return -metrics.log_loss(truth, preds)


def calculate_bic(n, loglik, num_params):
    return num_params * np.log(n) - 2 * loglik


def get_sklearn_bic(X,y,model):
    n = X.shape[0]
    n_params = X.shape[1]
    class_probs = model.predict_proba(X)
    pred_probs = model.predict_proba(X)
    loglik = log_likelihood(y, pred_probs)
    bic = calculate_bic(n, loglik, n_params)
    return bic


def get_glmnet_bic(X,y,fit,n_params):

    X_r = convert_to_r_mat(X)
    y_r = ro.IntVector(y)

    assess_res = glmnet.assess_glmnet(
        fit,
        newx = X_r,
        newy = y_r,
    )

    mse = assess_res.rx2["mse"]
    n = X.shape[0]
    return calculate_bic(n, mse, n_params)


def choose_model(bic_list, model_list, return_index):
    '''Returns model with lowest BIC.

    Args:
    -----
    bic_list : list
        bic for each model in model_list
    model_list : list
        List sklearn logistic regression fits
    return_index : bool
        If true, only return the index of the model
        with the lowest BIC. If false, return the best model.
    '''
    # initialize array of infinities to store BIC values
    if return_index:
        return np.argmin(bic_list)
    else:
        best_model = model_list[np.argmin(bic_list)]
        return best_model


def train_sklearn_glm(X,y,fit_intercept=False,family='binomial'):

    if family == "multinomial":
        model = linear_model.LogisticRegression(
            penalty = "none",
            multi_class = "multinomial",
            fit_intercept = fit_intercept,
        )
    else:
        model = linear_model.LogisticRegression(
            penalty="none",
            multi_class = "ovr",
            fit_intercept = fit_intercept,
        )

    model.fit(X,y)

    return model


def train_glmnet(X,y,folds=10,family='binomial',alpha=1):
    X_r = convert_to_r_mat(X)
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


def calc_prec_recall(yhat_background, yhat_foreground):
    r_yhat_fg = ro.FloatVector(yhat_foreground)
    r_yhat_bg = ro.FloatVector(yhat_background)

    #print(yhat_peaks.shape)
    #print(yhat_nonpeaks.shape)

    auc = prroc.pr_curve(
        scores_class0 = r_yhat_fg, # positive class scores
        scores_class1 = r_yhat_bg, # negative class scores
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
    }
    return output


def prec_recall(yhat, target_y, family='binomial', prefix=None, plot=False):
    if family == 'multinomial':
        return multinomial_prec_recall(yhat, target_y, plot, prefix)
    elif family == 'binomial':
        return binomial_prec_recall(yhat, target_y)


def multinomial_prec_recall(yhat, target_y, plot=False, prefix=None):

    n_classes = yhat.shape[1]
    pr_rec_dict = {}

    for this_class in range(n_classes):

        no_skill = len(target_y[target_y==this_class]) / len(target_y)

        this_class_yhat = yhat[:,this_class,0].copy()
        
        # plotting the distribution of yhat vals and, more importantly, using the
        # PPROC pr.curve function, requires a bit of difference between values
        # in the yhat vector. I there's only one distinct value, add a tiny bit
        # of noise to this class' yhat values just to make the funcitons work.
        if len(np.unique(this_class_yhat)) == 1:
            this_class_yhat += np.random.normal(0.0, 0.001, len(this_class_yhat))

        yhat_peaks = this_class_yhat[target_y==this_class]
        yhat_nonpeaks = this_class_yhat[target_y!=this_class]

        #inclass = np.zeros(len(target_y))
        #inclass[target_y == this_class] = 1

        if plot:
            df = pd.DataFrame(
                {'yhat': this_class_yhat, 'inclass': target_y == this_class}
            )
            sns.displot(df, x='yhat', hue='inclass', kde=True)
            plt.savefig('{}_class_{}'.format(prefix, this_class))
            plt.close()

        pr_rec = calc_prec_recall(yhat_nonpeaks, yhat_peaks)
        pr_rec['random_auc'] = no_skill
        pr_rec_dict[this_class] = pr_rec

    return pr_rec_dict


def binomial_prec_recall(yhat, target_y):

    no_skill = len(target_y[target_y==1]) / len(target_y)

    yhat_peaks = yhat[target_y==1]
    yhat_nonpeaks = yhat[target_y!=1]

    pr_rec = calc_prec_recall(yhat_nonpeaks, yhat_peaks)
    pr_rec['random_auc'] = no_skill
    # make key 1 and val a dict of prec/recall metrics
    # this makes the binary case sympatico with the multinomial case
    return {1: pr_rec}


def convert_to_r_mat(mat):
    mat_r = ro.r.matrix(
        mat,
        nrow=mat.shape[0],
        ncol=mat.shape[1],
    )
    return mat_r


def evaluate_fit(fit, test_X, test_y, family,
        lambda_cut="lambda.1se", prefix=None, plot=False):

    test_X_r = convert_to_r_mat(test_X)
    test_y_r = ro.IntVector(test_y)

    assess_res = glmnet.assess_glmnet(
        fit,
        newx = test_X_r,
        newy = test_y_r,
    )
    print("glmnet fit deviance:")
    print(assess_res.rx2["deviance"])
    print("misclassification error:")
    print(assess_res.rx2["class"])
    print("mse:")
    print(assess_res.rx2["mse"])
    print("mae:")
    print(assess_res.rx2["mae"])

    conf_mat = glmnet.confusion_glmnet(
        fit,
        newx = test_X_r,
        newy = test_y_r,
    )
    print("Confusion matrix on test data")
    print(conf_mat)

    yhat = glmnet.predict_cv_glmnet(
        fit,
        newx=test_X_r,
        s=lambda_cut,
    )

    output = prec_recall(yhat, test_y, family, prefix, plot)

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
    bins = []
    orig_y = records.y
    if continuous is not None:
        bins = records.quantize_quant(continuous)

    return records,bins,orig_y

def fetch_coefficients_binomial(fit):
    '''Returns a vector of coefficents.
    '''
    if "glmnet.fit" in fit.names:
        ncoefs = fit.rx2["glmnet.fit"].rx2["dim"][0] + 1
        coefs = glmnet.coef_cv_glmnet(fit, s="lambda.1se")
    else:
        ncoefs = 1
    coefs_arr = np.zeros((1, ncoefs))
    coefs_arr[0,:] = base.as_vector(coefs)
        
    return coefs_arr


def fetch_coefficients_multinomial(fit, n_classes):
    '''Yields an n_coefficients-by-n_classes array of fitted coefficients
    The first row is intercept.
    '''
    ncoefs = fit.rx2["glmnet.fit"].rx2["dim"][0] + 1
    coefs = glmnet.coef_cv_glmnet(fit, s="lambda.1se")
    coefs_arr = np.zeros((n_classes, ncoefs))
    for i in range(n_classes):
        coefs_arr[i,:] = base.as_matrix(coefs.rx2[str(i)])[:,0]
        
    return coefs_arr


def set_family(yvals):
    distinct_cats = np.unique(yvals)
    if len(distinct_cats) == 2:
        fam = 'binomial'
    else:
        fam = 'multinomial'
    return fam


#def filter_motifs(motif_list, motif_X, coefs, var_lut):
#    '''Determines which coeficients were shrunk to zero during LASSO regression
#    and removes motifs for which all covariates in motif_X were zero. Returns
#    a filtered set of motifs, a new array of X values (motif hits covariates),
#    and a new var_lut to map columns of the new X array to motif information.
#    '''
#
#    # keys are X arr indices, vals are dict
#    # of {'motif_idx': motif index in list of motifs,
#    #     'hits': the class of hit this covariate represents, i.e., [0,1], [1,1], etc.}
#    new_lut = {}
#    # construct lookup table where motif index is key, and value is list
#    #  of column indices for that motif in coefs
#    motif_lut = {}
#    
#    #print("------------------------------")
#    #print(var_lut)
#    #print("------------------------------")
#    for k,coef in var_lut.items():
#        # if this motif index isn't yet in the lut, place it in and give it a list
#        if not coef['motif_idx'] in motif_lut:
#            # k+1 here, since coefs will have the intercept at index 0
#            # and k is the index in the covariates array
#            motif_lut[coef['motif_idx']] = [k+1]
#        # if this motif idx is already present, append col to list
#        else:
#            motif_lut[coef['motif_idx']].append(k+1)
#
#    retain = []
#    # make nrow-by-zero array to start appending covariates from coeficiens with 
#    # predictive value
#    retained_X = np.zeros((motif_X.shape[0],0))
#    #print(retained_X.shape)
#    # now go through coefs columns to see whether any motif has all zeros
#    #print("------------------------------")
#    #print(motif_lut)
#    #print("------------------------------")
#    for motif_idx,motif_coef_inds in motif_lut.items():
#        # instantiate a list to carry bools
#        motif_any_nonzero = []
#        for coef_idx in motif_coef_inds:
#            # are any of these values non-zero?
#            #print(f"motif_idx: {motif_idx}")
#            #print(f"coef_idx: {coef_idx}")
#            #print(f"coefs: {coefs[:,coef_idx]}")
#            has_non_zero = np.any(coefs[:,coef_idx] != 0)
#            #print(f"has_non_zero: {has_non_zero}")
#            if has_non_zero:
#                retained_X = np.append(
#                    retained_X,
#                    motif_X[:,coef_idx-1][:,None],
#                    axis=1,
#                )
#                this_col_idx = retained_X.shape[1] - 1
#                new_lut[this_col_idx] = {
#                    # don't add one to len(retain) here, since we need the index
#                    # in the filtered list corresponding to this motif
#                    'motif_idx': len(retain),
#                    'hits': var_lut[coef_idx-1]['hits'],
#                }
#            motif_any_nonzero.append(has_non_zero)
#        
#        # if any column for this motif contained any non-zero values, retain the motif
#        #print(f"motif_any_nonzero: {motif_any_nonzero}")
#        if np.any(motif_any_nonzero):
#            retain.append(True)
#        else:
#            retain.append(False)
#            print(
#                f"WARNING: all regression coefficients for motif at "\
#                f"index {motif_idx} were shrunken to 0 during LASSO regression. "\
#                f"The motif has been removed from further consideration."
#            )
#        
#    #print(f"retain: {retain}")
#    # keep motifs for which at least one coefficient was non-zero
#    retained_motifs = [motif_list[i] for i,_ in enumerate(retain) if _]
#    return (retained_motifs, retained_X, new_lut)


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

    args = parser.parse_args()

    logging.info("Arguments")
    logging.info(str(args))
    logging.info("Reading in files")

    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    shape_fname = os.path.join(out_direc, 'test_shapes.npy')
    yval_fname = os.path.join(out_direc, 'test_y_vals.npy')
    config_fname = os.path.join(out_direc, 'config.json')
    rust_motifs_fname = os.path.join(out_direc, 'rust_results.json')
    lasso_fit_fname = os.path.join(out_direc, 'lasso_fit.pkl')
    eval_out_fname = os.path.join(out_direc, 'shape_precision_recall.json')
    seq_eval_out_fname = os.path.join(out_direc, 'seq_precision_recall.json')
    seq_and_shape_eval_out_fname = os.path.join(out_direc, 'seq_and_shape_precision_recall.json')
    prc_prefix = os.path.join(out_direc, 'shape_precision_recall_curve')
    seq_prc_prefix = os.path.join(out_direc, 'seq_precision_recall_curve')
    seq_and_shape_prc_prefix = os.path.join(out_direc, 'seq_and_shape_precision_recall_curve')
    combined_plot_prefix = os.path.join(out_direc, 'combined_precision_recall_curve')
    eval_dist_plot_prefix = os.path.join(out_direc, 'class_yhat_distribution')
    out_pref = args.o
    
    with open(config_fname, 'r') as f:
        args_dict = json.load(f)

    train_records,train_bins,train_orig_y = read_records(
        args_dict,
        in_direc,
        args.train_infile,
        args.param_names,
        args.train_params,
        continuous=args.continuous,
        dset_type="training",
    )

    logit_reg_str = "{}_{}_logistic_regression_result.pkl"
    shape_logit_reg_fname = os.path.join(
        out_direc,
        logit_reg_str.format(out_pref.replace('test','train'), "shape"),
    )

    # update cores so rust gets the right number for this job
    args_dict['cores'] = args.p
    args_dict['eval_shape_fname'] = shape_fname
    args_dict['eval_yvals_fname'] = yval_fname

    with open(config_fname, 'w') as f:
        json.dump(args_dict, f, indent=1)

    test_records,test_bins,test_orig_y = read_records(
        args_dict,
        in_direc,
        args.test_infile,
        args.param_names,
        args.test_params,
        continuous=args.continuous,
        #quantize_bins = bins,
        dset_type = "test",
    )

    # write shapes to npy file. Permute axes 1 and 2.
    with open(shape_fname, 'wb') as shape_f:
        np.save(shape_fname, test_records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(yval_fname, 'wb') as f:
        np.save(f, test_records.y.astype(np.int64))

    logging.info("Distribution of testing set sequences per class:")
    logging.info(inout.seqs_per_bin(test_records))

    logging.info("Getting distance between motifs and each record")

    fam = set_family(train_records.y)

    if os.path.isfile(rust_motifs_fname):
        RUST = "{} {}".format(
            rust_bin,
            config_fname,
        )

        retcode = subprocess.call(RUST, shell=True, env=my_env)

        if retcode != 0:
            sys.exit("Rust binary returned non-zero exit status")

        test_y = test_records.y
        test_X,var_lut = get_X_from_motifs(
            os.path.join(out_direc, "evaluated_motifs.json"),
            args_dict['max_count'],
        )

        train_y = train_records.y
        train_X = get_X_from_lut(
            os.path.join(out_direc, "rust_results.json"),
            var_lut,
        )

        # categories need to start with 0, so subtract one from
        #  each value until at least one of the y-val vectors has 0 as
        #  its minimum
        while (np.min(train_y) != 0) and (np.min(test_y) != 0):
            train_y -= 1
            test_y -= 1

        #shape_fit = train_glmnet(
        #    train_X,
        #    train_y,
        #    folds=10,
        #    family=fam,
        #    alpha=1,
        #)
        with open(lasso_fit_fname, 'rb') as f:
            shape_fit = pickle.load(f)

        coefs = fetch_coefficients(fam, shape_fit, args.continuous)

        # NOTE: TODO: go through coefficients and weed out motifs for which all
        #   coefficients are zero.
        # predict on test data
        shape_output = evaluate_fit(
            shape_fit,
            test_X,
            test_y,
            fam,
            lambda_cut="lambda.1se",
            prefix=eval_dist_plot_prefix + "_shape_only",
            plot=True,
        )

        with open(eval_out_fname, 'w') as f:
            json.dump(shape_output, f, indent=1)

        with open(shape_logit_reg_fname, 'wb') as f:
            pickle.dump(shape_fit, f)

        save_prc_plot(
            shape_output,
            prc_prefix,
            "Shape motifs",
        )

        logging.info("Done evaluating shape motifs on test data.")
        logging.info("==========================================")

        for class_name,class_info in shape_output.items():
            logging.info("Area under precision-recall curve for class {}: {}".format(class_name, class_info['auc']))
            logging.info(
                "Expected area under precision-recall curve for random performance on class {}: {}".format(class_name, class_info['random_auc'])
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
            family=fam,
            alpha=1,
        )

        test_seq_matches = fimo.FimoFile()
        test_seq_matches.parse(args.test_fimo_file)
        test_seq_X = test_seq_matches.get_design_matrix(test_records)

        seq_output = evaluate_fit(
            seq_fit,
            test_seq_X,
            test_y,
            fam,
            lambda_cut="lambda.1se",
            prefix=eval_dist_plot_prefix + "_seq_only",
            plot=True,
        )
        seq_logit_reg_fname = os.path.join(
            out_direc,
            logit_reg_str.format(out_pref.replace('test','train'), "sequence"),
        )

        save_prc_plot(
            seq_output,
            seq_prc_prefix,
            "Sequence motif",
        )

        with open(seq_eval_out_fname, 'w') as f:
            json.dump(seq_output, f, indent=1)

        logging.info("Done evaluating sequence motifs on test data.")
        logging.info("==========================================")

        for class_name,class_info in seq_output.items():
            logging.info("Area under precision-recall curve for class {}: {}".format(class_name, class_info['auc']))
            logging.info(
                "Expected area under precision-recall curve for random performance on class {}: {}".format(class_name, class_info['random_auc'])
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
                family=fam,
                alpha=1,
            )
            seq_and_shape_output = evaluate_fit(
                seq_and_shape_fit,
                test_seq_and_shape_X,
                test_y,
                fam,
                prefix=eval_dist_plot_prefix + "_seq_and_shape",
                plot=True,
                lambda_cut="lambda.1se",
            )
            seq_shape_logit_reg_fname = os.path.join(
                out_direc,
                logit_reg_str.format(out_pref.replace('test','train'), "sequence_and_shape"),
            )

            save_prc_plot(
                seq_and_shape_output,
                seq_and_shape_prc_prefix,
                "Sequence and shape motifs",
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

            for class_name,class_info in seq_and_shape_output.items():
                logging.info("Area under precision-recall curve for class {}: {}".format(class_name, class_info['auc']))
                logging.info(
                    "Expected area under precision-recall curve for random performance on class {}: {}".format(class_name, class_info['random_auc'])
                )

            logging.info("Results of evaluation are in {}".format(seq_and_shape_eval_out_fname))
            logging.info(
                "Precision recall curve plotted. Saved as {} and {}".format(
                    seq_and_shape_prc_prefix+".png",
                    seq_and_shape_prc_prefix+".pdf",
                )
            )

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

