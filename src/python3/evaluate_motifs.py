import inout
import shelve
import copy
import glob
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
from sklearn.model_selection import KFold,cross_val_score
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

def shape_run(
        shape_motifs,
        rust_motifs_fname,
        shape_fname,
        yval_fname,
        args_dict,
        config_fname,
        rust_bin,
        out_direc,
        recs,
):

    shape_motifs.write_shape_motifs_as_rust_output(rust_motifs_fname)

    new_motfs = copy.deepcopy(shape_motifs)

    # get motif evaluations on test data
    args_dict['eval_shape_fname'] = shape_fname
    args_dict['eval_yvals_fname'] = yval_fname
    args_dict['eval_rust_fname'] = rust_motifs_fname

    with open(config_fname, 'w') as f:
        json.dump(args_dict, f, indent=1)

    RUST = f"{rust_bin} {config_fname}"

    my_env = os.environ.copy()
    my_env['RUST_BACKTRACE'] = "1"
    retcode = subprocess.call(RUST, shell=True, env=my_env)

    if retcode != 0:
        print("Rust binary returned non-zero exit status")
        sys.exit(1)

    new_motifs = inout.Motifs(
        os.path.join(out_direc, "evaluated_motifs.json"),
        motif_type="shape",
        shape_lut = recs.shape_name_lut,
        max_count = args_dict["max_count"],
    )
    new_motifs.get_X(max_count = args_dict["max_count"])

    return new_motifs


def fimo_run(seq_motifs, seq_fasta, seq_meme_fname, fimo_direc, this_path, recs):

    seq_motifs.write_file(seq_meme_fname, recs)

    fimo_exec = os.path.join(this_path, "run_fimo.py")
    FIMO = f"python {fimo_exec} "\
        f"--seq_fname {seq_fasta} "\
        f"--meme_file {seq_meme_fname} "\
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

    new_seq_motifs = copy.deepcopy(seq_motifs)

    new_seq_motifs.get_X(
        fimo_fname = f"{fimo_direc}/fimo.tsv",
        rec_db = recs,
    )

    return new_seq_motifs


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


def calculate_F1():
    pass

def CV_F1(X, y, folds=5, family="binomial", fit_intercept=False, cores=None):

    mc = "multinomial"
    scoring = "f1_micro"
    if family == "binomial":
        mc = "ovr"
        scoring = "f1"

    estimator = linear_model.LogisticRegression(
        penalty = None,
        multi_class = mc,
        fit_intercept = fit_intercept,
        max_iter = 200,
    )

    F_scores = cross_val_score(
        estimator,
        X,
        y,
        scoring = scoring,
        cv = folds,
        n_jobs = cores,
    )

    return F_scores.mean()


def get_glmnet_bic(X,y,fit,n_params):
    pass


def get_sklearn_bic(X,y,model):
    n = X.shape[0]
    n_params = X.shape[1]
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


def choose_model(metric_list, model_list, return_index):
    '''Returns model with lowest BIC.

    Args:
    -----
    metric_list : list
        metric for each model in model_list
    model_list : list
        List of 2-tuples of (motifs_object, list_of_coefficients)
    return_index : bool
        If true, only return the index of the model
        with the lowest BIC. If false, return the best model.
    '''
    # initialize array of infinities to store BIC values
    if return_index:
        return np.argmax(metric_list)
    else:
        best_model = model_list[np.argmax(metric_list)]
        return best_model

def train_sklearn_glm(X,y,fit_intercept=False,family='binomial'):

    if family == "multinomial":
        model = linear_model.LogisticRegression(
            penalty = None,
            multi_class = "multinomial",
            fit_intercept = fit_intercept,
            max_iter = 200,
        )
    else:
        model = linear_model.LogisticRegression(
            penalty=None,
            multi_class = "ovr",
            fit_intercept = fit_intercept,
            max_iter = 200,
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


#def prec_recall(yhat, target_y, family='binomial', prefix=None, plot=False):
#    ######################################################################
#    ######################################################################
#    ## modify this to just use a more general fxn that will look a lot like the multinomial here. Since I'm returning a binary yhat array of shape (n_rec, n_cat, n_thresh) anyway
#    ######################################################################
#    ######################################################################
#    if family == 'multinomial':
#        return multinomial_prec_recall(yhat, target_y, plot, prefix)
#    elif family == 'binomial':
#        return binomial_prec_recall(yhat, target_y)


def prec_recall(yhat, target_y, plot=False, prefix=None):

    n_classes = yhat.shape[1]
    pr_rec_dict = {}

    for this_class in range(n_classes):

        # slice the correct index from yhat array
        this_class_yhat = yhat[:,this_class].copy()
        
        # if doing binary classification (logit regression) switch this_class to 1
        if n_classes == 1:
            this_class = 1

        no_skill = len(target_y[target_y==this_class]) / len(target_y)

        # plotting the distribution of yhat vals and, more importantly, using the
        # PPROC pr.curve function, requires a bit of difference between values
        # in the yhat vector. If there's only one distinct value, add a tiny bit
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
            print("***************")
            print(f"df shape: {df.shape}")
            print("***************")
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

def softmax(x):
    e_x = np.exp(x)
    return(e_x/e_x.sum())

def inv_logit(x):
    e_x = np.exp(x)
    return(e_x / (1+e_x))

def evaluate_fit2(
        coefs,
        test_X,
        test_y,
        prefix=None,
        plot=False,
    ):

    # X is shape (num_seqs, num_hit_cats*motif_num)
    num_seqs = test_X.shape[0]
    X = np.concatenate(
        (np.ones((num_seqs,1)), test_X),
        axis = 1,
    )

    num_cats = coefs.shape[0]
    #print("---------------------------------------------")
    #print(f"coefs shape: {coefs.shape}")
    #print("---------------------------------------------")
    yhat = np.zeros((num_seqs, num_cats))
    # get the logit-scale preditions for each category
    for col_i in range(num_cats):
        yhat[:,col_i] = np.dot(X, coefs[col_i,:])

    # check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if num_cats > 1:
        for row_i in range(num_seqs):
            yhat[row_i,:] = softmax(yhat[row_i,:])

    else:
        yhat = inv_logit(yhat)
        
    #print("************************************")
    #print(f"yhat shape: {yhat.shape}")
    #print(f"yhat: {yhat}")
    #print("************************************")

    #classes = np.zeros((num_seqs, num_cats, thresh_num))
    #for (i,threshold) in enumerate(np.linspace(
    #    start = 0.0,
    #    stop = 1.0,
    #    num = thresh_num,
    #)):
    #    for j in range(num_cats):
    #        classes[:, j, i] = y_hat[:,j] > threshold
        
    output = prec_recall(yhat, test_y, plot, prefix)

    return(output)


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


def read_records(
        args_dict,
        in_direc,
        infile,
        param_names,
        param_files,
        continuous=None,
        dset_type="training"
):

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
    with open("debug.pkl", "wb") as f:
        info = {'fit': fit, 'nclass': n_classes}
        pickle.dump(info, f)
    for i in range(n_classes):
        coefs_arr[i,:] = base.as_matrix(coefs.rx2[str(i)])[:,0]
        
    return coefs_arr


def set_family(yvals):
    distinct_cats = np.unique(yvals)
    num_cats = len(distinct_cats)
    if num_cats == 2:
        fam = 'binomial'
    else:
        fam = 'multinomial'
    return (fam,num_cats)


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
    parser.add_argument('--test_seq_fasta', type=str, help="basename of sequence fasta, must be within data_dir")
    parser.add_argument('--train_seq_fasta', type=str, help="basename of sequence fasta, must be within data_dir", default=None)
    parser.add_argument('--test_shape_files', nargs="+", type=str,
                         help='inputfiles with test shape scores')
    parser.add_argument('--train_shape_files', nargs="+", type=str,
                         help='inputfiles with training shape scores', default=None)
    parser.add_argument('--shape_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--data_dir', type=str, help="Directory containing data")
    parser.add_argument('--train_score_file', type=str, help="File with peak names and y-vals", default=None)
    parser.add_argument('--test_score_file', type=str, help="File with peak names and y-vals")
    parser.add_argument('--out_dir', type=str, help="Directory to which to write outputs")
    parser.add_argument('--nprocs', type=int, help="Number of cores to run in parallel")
    parser.add_argument('--out_prefix', type=str, help="Prefix to prepend to output files.")
    parser.add_argument('--config_file', type=str, help="Basename of configuration file.", default="config.json")

    level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level, stream=sys.stdout) 
    logging.getLogger('matplotlib.font_manager').disabled = True

    args = parser.parse_args()

    logging.info("Arguments")
    logging.info(str(args))
    logging.info("Reading in files")

    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    fimo_direc = f"{out_direc}/fimo_out"
    motif_fname = os.path.join(out_direc, 'final_motifs.dsm')
    test_shape_fname = os.path.join(out_direc, 'test_shapes.npy')
    train_shape_fname = os.path.join(out_direc, 'train_shapes.npy')
    test_yval_fname = os.path.join(out_direc, 'test_y_vals.npy')
    train_yval_fname = os.path.join(out_direc, 'train_y_vals.npy')
    config_fname = os.path.join(out_direc, args.config_file)
    # temp file just for running fimo
    seq_meme_fname = os.path.join(out_direc, 'seq_motifs.meme')
    rust_motifs_fname = os.path.join(out_direc, 'eval_rust_results.json')
    #fit_search = os.path.join(out_direc, '*lasso_fit.pkl')

    out_motif_basename = os.path.join(out_direc, "final_motifs")
    out_coefs_fname = out_motif_basename + "_coefficients.npy"

    with open(out_coefs_fname, "rb") as f:
        motif_coefs = np.load(f)

    #lasso_fit_fname = glob.glob(fit_search)[0]
    eval_out_fname = os.path.join(out_direc, 'precision_recall.json')
    prc_prefix = os.path.join(out_direc, 'precision_recall_curve')
    eval_dist_plot_prefix = os.path.join(out_direc, 'class_yhat_distribution')
    out_pref = args.out_prefix
    
    with open(config_fname, 'r') as f:
        args_dict = json.load(f)

    if (args.train_score_file is not None) and (args.train_shape_files is not None):
        train_records,train_bins,train_orig_y = read_records(
            args_dict,
            in_direc,
            args.train_score_file,
            args.shape_names,
            args.train_shape_files,
            continuous=args.continuous,
            dset_type="training",
        )

        # write shapes to npy file. Permute axes 1 and 2.
        with open(train_shape_fname, 'wb') as shape_f:
            np.save(shape_f, train_records.X.transpose((0,2,1,3)))
        # write y-vals to npy file.
        with open(train_yval_fname, 'wb') as f:
            np.save(f, train_records.y.astype(np.int64))

    logit_reg_str = "eval_logistic_regression_result.pkl"
    logit_reg_fname = os.path.join(
        out_direc,
        logit_reg_str,
    )

    # update cores so rust gets the right number for this job
    args_dict['cores'] = args.nprocs

    test_records,test_bins,test_orig_y = read_records(
        args_dict,
        in_direc,
        args.test_score_file,
        args.shape_names,
        args.test_shape_files,
        continuous=args.continuous,
        #quantize_bins = bins,
        dset_type = "test",
    )
    test_records.set_category_lut()

    # write shapes to npy file. Permute axes 1 and 2.
    with open(test_shape_fname, 'wb') as shape_f:
        np.save(shape_f, test_records.X.transpose((0,2,1,3)))
    # write y-vals to npy file.
    with open(test_yval_fname, 'wb') as f:
        np.save(f, test_records.y.astype(np.int64))

    logging.info("Distribution of testing set sequences per class:")
    logging.info(test_records.seqs_per_bin())

    logging.info("Getting distance between motifs and each record")

    if (args.train_score_file is not None) and (args.train_shape_files is not None):
        train_y = train_records.y
    test_y = test_records.y
    fam = set_family(test_y)

    if os.path.isfile(motif_fname):

        motifs = inout.Motifs()
        motifs.read_file( motif_fname )
        seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()

        if len(shape_motifs) > 0:

            test_shape_motifs = shape_run(
                shape_motifs,
                rust_motifs_fname,
                test_shape_fname,
                test_yval_fname,
                args_dict,
                config_fname,
                rust_bin,
                out_direc,
                test_records,
            )

            if (args.train_score_file is not None) and (args.train_shape_files is not None):
                train_shape_motifs = shape_run(
                    shape_motifs,
                    rust_motifs_fname,
                    train_shape_fname,
                    train_yval_fname,
                    args_dict,
                    config_fname,
                    rust_bin,
                    out_direc,
                    train_records,
                )

                all_train_motifs = train_shape_motifs
            all_test_motifs = test_shape_motifs

        if len(seq_motifs) > 0:

            if args.test_seq_fasta is None:
                raise inout.NoSeqFaException()

            if (args.train_score_file is not None) and (args.train_shape_files is not None):
                if args.train_seq_fasta is None:
                    raise inout.NoSeqFaException()

            test_seq_fasta = os.path.join(in_direc, args.test_seq_fasta)
            if (args.train_score_file is not None) and (args.train_shape_files is not None):
                train_seq_fasta = os.path.join(in_direc, args.train_seq_fasta)

                train_seq_motifs = fimo_run(
                    seq_motifs,
                    train_seq_fasta,
                    seq_meme_fname,
                    fimo_direc,
                    this_path,
                    train_records,
                )
            test_seq_motifs = fimo_run(
                seq_motifs,
                test_seq_fasta,
                seq_meme_fname,
                fimo_direc,
                this_path,
                test_records,
            )

            if len(shape_motifs) > 0:
                all_test_motifs = test_shape_motifs.new_with_motifs(test_seq_motifs)
                if (args.train_score_file is not None) and (args.train_shape_files is not None):
                    all_train_motifs = train_shape_motifs.new_with_motifs(train_seq_motifs)
            else:
                all_test_motifs = test_seq_motifs
                if (args.train_score_file is not None) and (args.train_shape_files is not None):
                    all_train_motifs = train_seq_motifs

        # categories need to start with 0, so subtract one from
        #  each value until at least one of the y-val vectors has 0 as
        #  its minimum
        if (args.train_score_file is not None) and (args.train_shape_files is not None):
            while (np.min(train_y) != 0) and (np.min(test_y) != 0):
                train_y -= 1
                test_y -= 1
        else:
            while np.min(test_y) != 0:
                test_y -= 1

##############################################################################
##############################################################################
##############################################################################
## check this as source of issue with seq and shape motif performance evaluation
##############################################################################
##############################################################################
##############################################################################

        #fit = train_glmnet(
        #    all_train_motifs.X,
        #    train_y,
        #    folds = 10,
        #    family=fam,
        #    alpha=1,
        #)

        #coefs = fetch_coefficients(fam, fit, args.continuous)
        fit_eval = evaluate_fit2(
            motif_coefs,
            all_test_motifs.X,
            test_y,
            prefix=eval_dist_plot_prefix,
            plot=False,
        )

        # predict on test data
        #fit_eval = evaluate_fit(
        #    fit,
        #    all_test_motifs.X,
        #    test_y,
        #    fam,
        #    lambda_cut="lambda.1se",
        #    prefix=eval_dist_plot_prefix,
        #    plot=True,
        #)

        with open(eval_out_fname, 'w') as f:
            json.dump(fit_eval, f, indent=1)

        #with open(logit_reg_fname, 'wb') as f:
        #    pickle.dump(fit, f)

        save_prc_plot(
            fit_eval,
            prc_prefix,
            "Motifs",
        )

        logging.info("Done evaluating motifs on test data.")
        logging.info("==========================================")

        for class_name,class_info in fit_eval.items():
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
        #os.remove(seq_meme_fname)

