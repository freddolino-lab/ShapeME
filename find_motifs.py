import inout
import sys
import os
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
from scipy.optimize import LinearConstraint
import shapemotifvis as smv
import multiprocessing as mp
import itertools
import welfords
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import cvlogistic
import numba
from numba import jit,prange
import pickle


class BasinHoppingBounds:
    '''class to use as accept_test argument to basinhopping
    minimizer.
    '''
    def __init__(self, xmax, xmin ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def retrieve_vals_from_target_vec(threshold, weights, shapes, targets_order,
                                  target_breaks, targets, L, S):

    vals_dict = {}
    if threshold is not None:
        vals_dict['threshold'] = threshold
    if weights is not None:
        vals_dict['weights'] = weights.reshape((L,S))
    if shapes is not None:
        vals_dict['shapes'] = shapes.reshape((L,S))

    for i,target in enumerate(targets_order):
        if i == 0:
            left_break = 0
        right_break = target_breaks[i]
        if target in ['weights', 'shapes']:
            vals_dict[target] = targets[left_break:right_break].reshape((L,S))
        else:
            vals_dict[target] = targets[left_break:right_break]
        left_break = right_break
    

    return vals_dict

def make_linear_constraint(target,S,L):
    """Sets up a LinearConstraint object to constrain the sum
    of all shape weights to one. 
    """

    # Here we make a 1-by-L*S+1 matrix to get dot product of beta
    #   and weights and threshold
    beta = np.zeros((1,L*S+1))
    # lower and upper bounds on sum of weights are 1
    lower_bound = np.array([S*L])
    upper_bound = np.array([S*L])

    # set appropriate values in beta to 1, leave the -1 index as 0, since
    #   we're not constraining the threshold
    beta[0,:-1] = 1.0

    const = LinearConstraint(
        beta,
        lower_bound,
        upper_bound,
    )
    return const

def brent_optimize_helper(r_idx, w_idx, dist, db):

    motif_weights = db.weights[r_idx,:,:,w_idx]
    L,S = motif_weights.shape
    motif_shapes = db.windows[r_idx,:,:,w_idx]
    motif_thresh = db.thresholds[r_idx,w_idx]

    # will compare motif_shapes to all reference shapes
    ref_shapes = db.windows
    y_vals = db.y

    final_motifs_dict = {}
    func_info = {"NFeval":0, "eval":[], "value":[], "threshold":[]}

    args = (
        motif_shapes,
        motif_weights,
        ref_shapes,
        y_vals,
        dist,
        func_info,
    )

    thresh_opt = opt.brent(
        brent_optimize_worker,
        args = args,
    )

    return(thresh_opt)
 

def brent_optimize_threshold(record_db, dist, window_inds=None):
    """Optmize only the threshold values for seeds.

    Modifies:
    ---------
    record_db
    """
    results = []
    R,L,S,W = record_db.windows.shape

    if window_inds is not None:
        for i in range(len(window_inds[0])):
            r_idx = window_inds[0][i]
            w_idx = window_inds[1][i]
            final_threshold = brent_optimize_helper(
                r_idx,
                w_idx,
                dist,
                record_db,
            )
            record_db.thresholds[r_idx,w_idx] = final_threshold
    else:
        for r_idx in range(R):
            for w_idx in range(W):
                final_threshold = brent_optimize_helper(
                    r_idx,
                    w_idx,
                    dist,
                    record_db,
                )
                record_db.thresholds[r_idx,w_idx] = final_threshold


def stochastic_optimize(out_fname, record_db, dist, temp=1.0, stepsize=0.5,
                        fatol=0.0001, adapt=False,
                        maxfev=None, opt_params=['weights'],
                        window_inds=None, method = 'nelder-mead',
                        niter=100, niter_success = 20, constraints_dict = {},
                        max_count = 4, alpha = 0.1):
    """Perform motif optimization stochastically
    
    Args:
    
    Returns:
    """

    R,L,S,W = record_db.windows.shape 

    if window_inds is not None:
        for i in range(len(window_inds)):
            r_idx,w_idx = window_inds[i]
            final = stochastic_opt_helper(
                r_idx,
                w_idx,
                dist = dist,
                db = record_db,
                temperature = temp,
                stepsize = stepsize,
                fatol = fatol,
                adapt = adapt,
                maxfev = maxfev,
                max_count = max_count,
                alpha = alpha,
                targets = opt_params,
                method = method,
                niter_success = niter_success,
                constraints_dict = constraints_dict,
                niter = niter,
            )
            
            if i > 0:

                with open(out_fname, 'rb') as outf:
                    results = pickle.load(outf)

                results.append((final,r_idx,w_idx))

                with open(out_fname, 'wb') as outf:
                    pickle.dump(results, outf)

            else:
                results = [(final,r_idx,w_idx)]
                with open(out_fname, 'wb') as outf:
                    pickle.dump(results, outf)

    else:
        for r_idx in range(R):
            for w_idx in range(W):
                final = stochastic_opt_helper(
                    r_idx,
                    w_idx,
                    dist = dist,
                    db = record_db,
                    temperature = temp,
                    stepsize = stepsize,
                    fatol = fatol,
                    adapt = adapt,
                    maxfev = maxfev,
                    max_count = max_count,
                    alpha = alpha,
                    targets = opt_params,
                    method = method,
                    niter_success = niter_success,
                    constraints_dict = constraints_dict,
                    niter = niter,
                )
                if os.path.isfile(out_fname):

                    with open(out_fname, 'rb') as outf:
                        results = pickle.load(outf)

                    results.append((final,r_idx,w_idx))

                    with open(out_fname, 'wb') as outf:
                        pickle.dump(results, outf)

                else:
                    results = [(final,r_idx,w_idx)]
                    with open(out_fname, 'wb') as outf:
                        pickle.dump(results, outf)



def mp_optimize(record_db, dist, fatol=0.0001, opt_params=['weights'],
                        adapt=False, window_inds=None, p=1, maxfev=None):
    """Perform motif optimization in a multiprocessed manner
    
    Args:
    
    Returns:
    """

    results = []
    R,L,S,W = record_db.windows.shape 

    if window_inds is not None:
        for i in range(len(window_inds[0])):
            r_idx = window_inds[0][i]
            w_idx = window_inds[1][i]
            final_weights = mp_optimize_helper(
                r_idx,
                w_idx,
                dist,
                record_db,
                fatol,
                adapt,
                maxfev,
                opt_params,
            )
            results.append(final_weights)

    else:
        for r_idx in range(R):
            for w_idx in range(W):
                final_weights = mp_optimize_helper(
                    r_idx,
                    w_idx,
                    dist,
                    record_db,
                    fatol,
                    adapt,
                    maxfev,
                    opt_params,
                )
                results.append(final_weights)

    return results

def stochastic_opt_helper(r_idx, w_idx, dist, db,
                          temperature, stepsize, fatol, adapt,
                          maxfev, max_count=4, alpha=0.1,
                          targets = ['weights'],
                          method='nelder-mead',
                          niter_success = 100, niter=100, constraints_dict={}):
    """Helper function to allow weight optimization to be multiprocessed
    
    Args:
    -----
    r_idx : int
        Record index of this motif within the db
    w_idx : int
        Window index of this motif within its record in the db
    dist : func
        Function defining the distance metric used for determining matches
    db : inout.RecordDatabse
    temperature : float
        Sets the T parameter to scipy.optimize.basinhopping
    stepsize : float
        Sets the stepsize parameter to scipy.optimize.basinhopping
    fatol : float
        MI tolerance used to set convergence criterion
    adapt : bool
        Set nelder-mead to 'adaptive' if True.
    maxfev : int
        Maximum number of function evaluations to perform.
    max_count : int
        Sets the maximum number of hits to be counted for each strand.
        Default is 4.
    alpha : float
        Between 0 and 1, sets the minimum value weights will acheive
        after inv-logit and prior to normalization to sum to one.
    targets : list
        Contains only 'weights' by default. Add 'shapes' and/or 'threshold'
        to also optimize those.
    method : str
        Sets the optimization method to scipy.optimize.minimize
    niter_success : int
        Stop the run if the global minimum candidate remains the same
        for this number of iterations.
    niter : int
        Total number of basin hops that will be run.
    constraints_dict : dict
        Keys are target types ("threshold", "weights", and "shapes")
        and values are tuples of (lower_bound, upper_bound)

    Returns:
    --------
        final_motif_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """

    motif_weights = db.weights[r_idx,:,:,w_idx]
    L,S = motif_weights.shape
    motif_shapes = db.windows[r_idx,:,:,w_idx]
    motif_thresh = db.thresholds[r_idx,w_idx]

    motif_dict = {
        'weights': motif_weights.flatten(),
        'threshold': motif_thresh,
        'shapes': motif_shapes.flatten(),
    }

    # scipy.optimize.minimize requires the target to optimize be a 1d array
    #target = np.append(motif_weights, motif_thresh)
    idx_breaks = []
    bounds = []
    for i,thing in enumerate(targets):
        this_bound = constraints_dict[thing]
        print("{} bounds are: ".format(thing))
        print(this_bound)
        if i == 0:
            target = motif_dict[thing]
            idx_breaks.append(len(target))
        else:
            target = np.append(target, motif_dict[thing])
            idx_breaks.append(len(target))
        this_bound = [this_bound] * motif_dict[thing].size
        bounds.extend(this_bound)

    # will compare motif_shapes to all reference shapes
    ref_shapes = db.windows
    y_vals = db.y

    final_motifs_dict = {}
    func_info = {"NFeval":0, "eval":[], "value":[], "threshold":[]}

    if 'threshold' in targets:
        thresh_arg = None
    else: thresh_arg = motif_dict['threshold']

    if 'weights' in targets:
        weights_arg = None
    else: weights_arg = motif_dict['weights']

    if 'shapes' in targets:
        shapes_arg = None
    else: shapes_arg = motif_dict['shapes']

    min_kwargs = {
        'args': (
            ref_shapes,
            y_vals,
            dist,
            func_info,
            idx_breaks,
            targets,
            max_count,
            thresh_arg,
            shapes_arg,
            weights_arg,
            alpha,
        ), 
        'method': method,
        'options' : {
            'fatol': fatol,
            'adaptive': adapt,
            'maxfev': maxfev,
        },
        'bounds': bounds,
    }

    min_bounds,max_bounds = list(zip(*bounds))
    bounds = BasinHoppingBounds(xmin=min_bounds, xmax=max_bounds)

    final_opt = opt.basinhopping(
        optimize_worker,
        target, 
        T = temperature,
        stepsize = stepsize,
        minimizer_kwargs = min_kwargs,
        niter_success = niter_success,
        niter = niter,
        accept_test = bounds,
    )

    final = final_opt['x']

    vals_dict = retrieve_vals_from_target_vec(
        thresh_arg,
        weights_arg,
        shapes_arg,
        targets,
        idx_breaks,
        final,
        L,
        S,
    )
    threshold_final = vals_dict['threshold']
    weights_final = vals_dict['weights']
    shapes_final = vals_dict['shapes']

    R,L,S,W = ref_shapes.shape
    mi_opt,hits = inout.run_query_over_ref(
        y_vals,
        shapes_final,
        weights_final,
        threshold_final,
        ref_shapes, 
        R,
        W,
        dist,
        max_count,
        alpha,
    )

    final_motifs_dict['hits'] = hits
    final_motifs_dict['mi'] = mi_opt
    final_motifs_dict['weights'] = weights_final
    final_motifs_dict['threshold'] = threshold_final
    final_motifs_dict['motif'] = shapes_final

    mi_orig,_ = inout.run_query_over_ref(
        y_vals,
        motif_shapes,
        motif_weights,
        motif_thresh,
        ref_shapes, 
        R,
        W,
        dist,
        max_count,
        alpha,
    )

    final_motifs_dict['mi_orig'] = mi_orig
    final_motifs_dict['orig_weights'] = motif_weights
    final_motifs_dict['orig_threshold'] = motif_thresh
    final_motifs_dict['orig_shapes'] = motif_shapes
    final_motifs_dict['r_idx'] = r_idx
    final_motifs_dict['w_idx'] = w_idx
    #final_motifs_dict['opt_success'] = final_opt['success']
    #final_motifs_dict['opt_message'] = final_opt['message']
    final_motifs_dict['opt_info'] = func_info

    return final_motifs_dict


def mp_optimize_helper(r_idx, w_idx, dist, db, fatol, adapt,
                              maxfev, targets = ['weights']):
    """Helper function to allow weight optimization to be multiprocessed
    
    Args:
    -----
    r_idx : int
        Record index of this motif within the db
    w_idx : int
        Window index of this motif within its record in the db
    dist : func
        Function defining the distance metric used for determining matches
    db : inout.RecordDatabse
    fatol : float
        MI tolerance used to set convergence criterion
    adapt : bool
        Set nelder-mead to 'adaptive' if True.
    maxfev : int
        Maximum number of function evaluations to perform.
    targets : list
        Contains only 'weights' by default. Add 'shapes' and/or 'threshold'
        to also optimize those.
    
    Returns:
    --------
        final_motif_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """

    motif_weights = db.weights[r_idx,:,:,w_idx]
    L,S = motif_weights.shape
    motif_shapes = db.windows[r_idx,:,:,w_idx]
    motif_thresh = db.thresholds[r_idx,w_idx]

    motif_dict = {
        'weights': motif_weights.flatten(),
        'threshold': motif_thresh,
        'shapes': motif_shapes.flatten(),
    }

    # scipy.optimize.minimize requires the target to optimize be a 1d array
    #target = np.append(motif_weights, motif_thresh)
    idx_breaks = []
    for i,thing in enumerate(targets):
        if i == 0:
            target = motif_dict[thing]
            idx_breaks.append(len(target))
        else:
            target = np.append(target, motif_dict[thing])
            idx_breaks.append(len(target))

    # will compare motif_shapes to all reference shapes
    ref_shapes = db.windows
    y_vals = db.y

    final_motifs_dict = {}
    func_info = {"NFeval":0, "eval":[], "value":[], "threshold":[]}

    if 'threshold' in targets:
        thresh_arg = None
    else: thresh_arg = motif_dict['threshold']

    if 'weights' in targets:
        weights_arg = None
    else: weights_arg = motif_dict['weights']

    if 'shapes' in targets:
        shapes_arg = None
    else: shapes_arg = motif_dict['shapes']

    final_opt = opt.minimize(
        optimize_worker,
        target, 
        args = (
            ref_shapes,
            y_vals,
            dist,
            func_info,
            idx_breaks,
            targets,
            thresh_arg,
            shapes_arg,
            weights_arg,
        ), 
        method = "nelder-mead",
        options = {
            'fatol': fatol,
            'adaptive': adapt,
            'maxfev': maxfev,
        }
    )

    final = final_opt['x']

    vals_dict = retrieve_vals_from_target_vec(
        thresh_arg,
        weights_arg,
        shapes_arg,
        targets,
        idx_breaks,
        final,
        L,
        S,
    )
    threshold_final = vals_dict['threshold']
    weights_final = vals_dict['weights']
    shapes_final = vals_dict['shapes']

    R,L,S,W = ref_shapes.shape
    mi_opt,hits = inout.run_query_over_ref(
        y_vals,
        shape_final,
        weights_final,
        threshold_final,
        ref_shapes, 
        R,
        W,
        dist,
        max_count,
        alpha,
    )

    final_motifs_dict['hits'] = hits
    final_motifs_dict['mi'] = mi_opt
    final_motifs_dict['weights'] = weights_final
    final_motifs_dict['threshold'] = threshold_final
    final_motifs_dict['motif'] = shapes_final

    mi_orig,_ = inout.run_query_over_ref(
        y_vals,
        motif_shapes,
        motif_weights,
        motif_thresh,
        ref_shapes, 
        R,
        W,
        dist,
        max_count,
        alpha,
    )

    final_motifs_dict['mi_orig'] = mi_orig
    final_motifs_dict['orig_weights'] = motif_weights
    final_motifs_dict['orig_threshold'] = motif_thresh
    final_motifs_dict['orig_shapes'] = motif_shapes
    final_motifs_dict['r_idx'] = r_idx
    final_motifs_dict['w_idx'] = w_idx
    final_motifs_dict['opt_success'] = final_opt['success']
    final_motifs_dict['opt_message'] = final_opt['message']
    final_motifs_dict['opt_info'] = func_info

    return final_motifs_dict


def brent_optimize_worker(threshold, shapes, weights, ref_shapes,
                          y, dist_func, max_count, alpha, info):
    """Function to optimize a particular motif's weights
    for distance calculation.

    Args:
    -----
        threshold : float
            threshold to be optimized by brent optimizer
        shapes : np.array
            Array of shape (L,S), where L is the length of each window
            and S is the number of shape parameters. Present only if
            we're not optimizing shape values. If we are optimizing shape
            values, then the values are wrapped into targets and this argument
            is None.
        weights : np.array
            Array of shape (L,S), where L is the length of each window
            and S is the number of shape parameters. Present only if
            we're not optimizing weights values. If we are optimizing weights
            values, then the values are wrapped into targets and this argument
            is None.
        ref_shapes : np.array
            Array of shape (R, L, S, W), where R is the number of records,
            L and S are described in shapes, and W is the number
            of windows per record/shape.
        y : np.array
            1D numpy array of length R containing the ground truth y values.
        dist_func : function
            Function to use in distance calculation
        max_count : int
            Sets the maximum number of hits to count for each strand.
        alpha : float
            Between 0 and 1, sets the minimum value weights will acheive
            after inv-logit and prior to normalization to sum to one.
        info : dict
            Store number of function evals and value associated with it.
            keys must include NFeval: int, value: list, eval: list

    Returns:
    --------
        MI for the weighted matches to the records
    """

    R,L,S,W = ref_shapes.shape

    this_mi,hits = inout.run_query_over_ref(
        y,
        shapes,
        weights,
        threshold,
        ref_shapes, 
        R,
        W,
        dist_func,
        max_count,
        alpha,
    )

    if info["NFeval"] % 10 == 0:
        info["value"].append(this_mi)
        info["eval"].append(info["NFeval"])
        info["threshold"].append(threshold)
    info["NFeval"] += 1

    return -this_mi


def optimize_worker(targets, all_shapes, y, dist_func, info,
                    target_breaks, targets_order, max_count=4,
                    threshold=None, shapes=None, weights=None,
                    alpha=0.1,
                    constraints_dict={}):
    """Function to optimize a particular motif's weights
    for distance calculation.

    Args:
    -----
        targets : np.array
            Targets to optimize. 1D array, the shape of which will vary
            depending on whether we're optimizing weights, shapes, thresholds,
            or some combination of those parameters.
        shapes : np.array
            Array of shape (L,S), where L is the length of each window
            and S is the number of shape parameters. Present only if
            we're not optimizing shape values. If we are optimizing shape
            values, then the values are wrapped into targets and this argument
            is None.
        weights : np.array
            Array of shape (L,S), where L is the length of each window
            and S is the number of shape parameters. Present only if
            we're not optimizing weights values. If we are optimizing weights
            values, then the values are wrapped into targets and this argument
            is None.
        threshold : float
            Maximum distance to reference to call a comparison a match.
            If we're optimizing the threshold, then this argument is None.
        all_shapes : np.array
            Array of shape (R, L, S, W), where R is the number of records,
            L and S are described in shapes, and W is the number
            of windows per record/shape.
        y : np.array
            1D numpy array of length R containing the ground truth y values.
        dist_func : function
            Function to use in distance calculation
        info : dict
            Store number of function evals and value associated with it.
            keys must include NFeval: int, value: list, eval: list
        target_breaks : list
            A list of break-points in targets to be able to slice appropriate
            values from targets vector.
        targets_order : list
            List of target types represented by the values in targets.
        max_count : int
            Default is 4. Sets the maximum number of hits to count
            for each strand.
        alpha : float
            Between 0 and 1, sets the minimum value weights will acheive
            after inv-logit and prior to normalization to sum to one.
        constraints_dict : dict
            Defines the constraints for each target. Has the keys 'threshold',
            'weights', and 'shapes'. Each key's value is a tupple of (lower, upper)
            limits on that target.

    Returns:
    --------
        MI for the weighted matches to the records
    """

    R,L,S,W = all_shapes.shape

    # vals_dict has 'threshold', 'weights', and 'shapes' keys.
    vals_dict = retrieve_vals_from_target_vec(
        threshold,
        weights,
        shapes,
        targets_order,
        target_breaks,
        targets,
        L,
        S,
    )

    # I commented this out on 2021-09-28 because I found that I should be passing an accept_test argument directly to basinhopping.
    # insert logic here to handle constraints
    #for target_name,target_constraints in constraints_dict.items():
    #    target_vals = vals_dict[target_name]

    #    below = False
    #    above = False

    #    # Which are below lower bound?
    #    below_sane = target_vals < target_constraints[0]
    #    # if a value is below the lower bound, calculate how far out
    #    #  to determine what to set garbage MI to.
    #    if np.any(below_sane):
    #        below = True
    #        distance_below = np.max(
    #            np.abs(
    #                target_vals[below_sane] - target_constraints[0]
    #            )
    #        )
    #        print("{} value of {} was {} far below the constraint ({}).".format(target_name, np.min(target_vals[below_sane]), distance_below, target_constraints[0]))
    #        garbage_val = 50 + 10_000 * distance_below
    #        
    #    # Which are above upper bound?
    #    above_sane = target_vals > target_constraints[1]
    #    # If any values were above the upper bound, calculate how far,
    #    #  then calculate the garbage MI, comparing the the one
    #    #  calculated for the values violating the lower constraint,
    #    #  if they exist, and finally, return the garbage MI
    #    if np.any(above_sane):
    #        above = True
    #        distance_above = np.max(
    #            np.abs(
    #                target_vals[above_sane] - target_constraints[1]
    #            )
    #        )
    #        print("{} value of {} was {} far above the constraint ({}).".format(target_name, np.max(target_vals[above_sane]), distance_above, target_constraints[1]))
    #        if below:
    #            above_garbage = 50 + 10_000 * distance_above
    #            garbage_val = np.max([garbage_mi, above_garbage])
    #        else:
    #            garbage_val = 50 + 10_000 * distance_above

    #    if above or below:
    #        return garbage_val

    this_mi,hits = inout.run_query_over_ref(
        y,
        vals_dict['shapes'],
        vals_dict['weights'],
        vals_dict['threshold'],
        all_shapes, 
        R,
        W,
        dist_func,
        max_count,
        alpha,
    )

    if info["NFeval"] % 10 == 0:
        info["value"].append(this_mi)
        info["eval"].append(info["NFeval"])
        info["threshold"].append(vals_dict['threshold'])
    info["NFeval"] += 1

    return -this_mi

#@jit(nopython=True, parallel=False)
#def optim_generate_peak_array(ref, query, weights, threshold,
#                              results, R, W, dist):
#    """Does same thing as generate_peak_vector, but hopefully faster
#    
#    Args:
#    -----
#    ref : np.array
#        The windows attribute of an inout.RecordDatabase object. Will be an
#        array of shape (R,L,S,W), where R is the number of records,
#        L is the window size, S is the number of shape parameters, and
#        W is the number of windows for each record.
#    query : np.array
#        A slice of the first and final indices of the windows attribute of
#        an inout.RecordDatabase object to check for matches in ref.
#        Should be an array of shape (L,S).
#    weights : np.array
#        A slice of the first and final indices of the weights attribute of
#        and inout.RecordDatabase object. Will be applied to the distance
#        calculation. Should be an array of shape (L,S).
#    threshold : np.array
#        Minimum distance to consider a match.
#    results : 1d np.array
#        Array of shape (R), where R is the number of records in ref.
#        This array should be populated with zeros, and will be filled
#        with 1's where matches are found.
#    R : int
#        Number of records
#    W : int
#        Number of windows for each record
#    dist : function
#        The distance function to use for distance calculation.
#    """
#    
#    for r in range(R):
#        for w in range(W):
#            
#            ref_seq = ref[r,:,:,w]
#            distance = dist(query, ref_seq, weights)
#            
#            if distance < threshold:
#                # if a window has a distance low enough,
#                #   set this record's result to 1
#                results[r] = 1
#                break

def make_initial_seeds(records, wsize,wstart,wend):
    """ Function to make all possible seeds, superceded by the precompute
    all windows method in seq_database
    """
    seeds = []
    for param in records:
        for window in param.sliding_windows(wsize, start=wstart, end=wend):
            seeds.append(window)
    return seeds

def optimize_mi(param_vec, data, sample_perc, info):
    """ Function to optimize a particular motif

    Args:
        param_vec (list) - values to optimize. Last value is threshold
        data (SeqDatabase) - full seq database
        sample_perc(float) - percentage of database to calculate MI over
        info (dict) - store number of function evals and value associated with it
                      keys must include NFeval: int, value: list, eval: list
    Returns:
        MI for the parameters over a random sample of the seq database
        as determined by input data
    """
    threshold = param_vec[-1]
    this_data = data
    this_discrete = generate_peak_vector(
        this_data,
        param_vec[:-1],
        threshold,
        args.rc,
    )
    this_mi = this_data.mutual_information(this_discrete)
    if info["NFeval"]%10 == 0:
        info["value"].append(this_mi)
        info["eval"].append(info["NFeval"])
    info["NFeval"] += 1
    return -this_mi

def generate_peak_vector(data, motif, threshold, rc=False):
    """ Function to calculate the sequences that have a match to a motif
    Args:
        data (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
        motif_vec(np.array) - numpy array containing motif vector
        threshold(float) - a distance threshold to be considered a match
    Returns:
        discrete (np.array) - a numpy array of 1 or 0 for each sequence where
                              1 is a match and 0 is not.
    """
    this_discrete = []
    motif_vec = motif.as_vector(cache=True)
    if rc:
        motif.rev_comp()
        motif_vec_rc = motif.as_vector()
        motif.rev_comp()

    for this_seq in data.iterate_through_precompute():
        seq_pass = False
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            if distance < threshold:
                seq_pass = True
                break
        if rc:
            for this_motif in this_seq:
                distance = this_motif.distance(motif_vec_rc, vec=True, cache=True)
                if distance < threshold:
                    seq_pass = True
                    break

        this_discrete.append(seq_pass)
    return np.array(this_discrete)

@jit(nopython=True, parallel=False)
def fast_generate_peak_array(data, threshold, results, N, W, dist):
    """Does same thing as generate_peak_vector, but hopefully faster
    
    Args:
    -----
    data : np.array
        Array of shape (N,L*P,W), where N is the number of records,
        L*P is the window size times the number of parameters, and
        W is the number of windows for each record.
    threshold : float
        Minimum distance to consider a match.
    results : 2d np.array
        Array of shape (N*W,N), where N*W is the number of records times the number
        of windows for each record and N is the number of records.
        This array should be populated with zeros.
    N : int
        Number of records
    W : int
        Number of windows for each record
    dist : function
    """
    
    for n in range(N):
        for w in range(W):
            
            row_idx = n*W + w
            q_seq = data[n,:,w]
            
            for r_n in range(N):
                for r_w in range(W):
                    ref_seq = data[r_n,:,r_w]
                    
                    distance = dist(q_seq, ref_seq)
                    if distance < threshold:
                        # if distance is low enough,
                        #   set the index for this seed/ref combo to 1
                        results[row_idx, r_n] = 1
                        break

def generate_dist_vector(data, motif, rc=False):
    """ Function to calculate the best possible match value for each seq
    Args:
        data (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
        motif (np.array) - numpy array containing motif vector
        rc (bool) - check the reverse complement matches as well
    Returns:
        discrete (np.array) - a numpy array of minimum match value for each seq
    """
    all_matches = []
    motif_vec = motif.as_vector(cache=True)
    if rc:
        motif.rev_comp()
        motif_vec_rc = motif.as_vector()
        motif.rev_comp()
    this_seq_matches = []
    for this_seq in data.iterate_through_precompute():
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            this_seq_matches.append(distance)
        if rc:
            for this_motif in this_seq:
                distance = this_motif.distance(motif_vec_rc, vec=True, cache=True)
                this_seq_matches.append(distance)

        all_matches.append(np.min(this_seq_matches))
        this_seq_matches = []
    return np.array(all_matches)

def find_initial_threshold(records, seeds_per_seq=1, max_seeds = 10000):
    """ Function to determine a reasonable starting threshold given a sample
    of the data

    Args:
        records (SeqDatabase) - database to calculate over, must already have
                             motifs pre_computed
    Returns:
        threshold (float) - a threshold that is the
                            mean(distance)-2*stdev(distance))
    """

    # calculate stdev and mean using welford's algorithm
    online_mean = welfords.Welford()
    records_shuffled = records.shuffle()
    total_seeds = []
    seed_counter = 0
    # get a set of seeds to run against each other
    for i, seq in enumerate(records_shuffled.iterate_through_precompute()):

        # sample in a random order from the sequence
        rand_order = np.random.permutation(len(seq))
        curr_seeds_per_seq = 0
        for index in rand_order:
            motif = seq[index]
            if curr_seeds_per_seq >= seeds_per_seq:
                break
            total_seeds.append(motif)
            curr_seeds_per_seq += 1
            seed_counter += 1
        if seed_counter >= max_seeds:
            break

    logging.info(
        "Using {} random seeds to determine threshold from pairwise distances".format(
            len(total_seeds)
        )
    )
    for i, seedi in enumerate(total_seeds):
        for j, seedj in enumerate(total_seeds):
            newval = seedi.distance(
                seedj.as_vector(cache=True),
                vec=True,
                cache=True,
            )
            if i >= j:
                continue
            else:
                newval = seedi.distance(
                    seedj.as_vector(cache=True),
                    vec=True,
                    cache=True,
                )
                online_mean.update(newval)

    mean = online_mean.final_mean()
    stdev = online_mean.final_stdev()
    
    logging.info("Threshold mean: %s and stdev %s"%(mean, stdev))
    return mean, stdev

def seqs_per_bin(records):
    """ Function to determine how many sequences are in each category

    Args:
        records (SeqDatabase) - database to calculate over
    Returns:
        outstring - a string enumerating the number of seqs in each category
    """
    string = ""
    for value in np.unique(records.y):
        string += "\nCat {}: {}".format(
            value, np.sum(records.y ==  value)
        )
    return string

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

def read_parameter_file(infile):
    """ Wrapper to read a single parameter file

    Args:
        infile (str) - input file name
    Returns:
        inout.FastaFile object containing data
    """
    fastadata = inout.FastaFile()
    with open(infile) as f:
        fastadata.read_whole_datafile(f)
    return fastadata

class MotifMatch(Exception):
    """ Exception class used for greedy search. To be raised when a motif
    match is found
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return "Distance is {}".format(self.value)

def greedy_search(records, threshold = 10, number=1000):
    """ Function to find initial seeds by a greedy search

    Prints the number of seeds per class to the logger

    Args:
        records (inout.SeqDatabase) - input data, must have motifs already 
                                   precomputed
        threshold (float) - threshold for considering a motif a match
        number (int) - number of seeds to stop near 
    Returns:
        seeds (list) - a list of dsp.ShapeParamSeq objects to be considered
    """
    seeds = []
    values = []
    records_shuffled = records.shuffle()
    for i,seq in enumerate(records_shuffled.iterate_through_precompute()):
        if(len(seeds) >= number):
            break
        for motif in seq:
            try:
                for motif2 in seeds:
                    distance = motif2.distance(
                        motif.as_vector(),
                        vec=True,
                        cache=True,
                    )
                    if distance < threshold:
                        raise MotifMatch(distance)
                seeds.append(motif)
                values.append(records_shuffled.values[i])
            except MotifMatch as e:
                continue
    values = np.array(values)
    for value in np.unique(values):
        logging.info("Seeds in Cat {}: {}".format(value, np.sum(values == value)))
    return seeds

def greedy_search2(records, threshold = 10, number=1000, seeds_per_seq = 1, rc=False, prev_seeds=None):
    """ Function to find initial seeds by a greedy search

    Prints the number of seeds per class to the logger

    Args:
        records (inout.SeqDatabase) - input data, must have motifs already 
                                   precomputed
        threshold (float) - threshold for considering a motif a match
        number (int) - number of seeds to stop near 
    Returns:
        seeds (list) - a list of dsp.ShapeParamSeq objects to be considered
    """
    if prev_seeds:
        seeds = prev_seeds
    else:
        seeds = []
    values = []
    records_shuffled = records.shuffle()
    for i,seq in enumerate(records_shuffled.iterate_through_precompute()):
        if i % 500 == 0:
            logging.info("Greedy search on seq {}".format(i))
        if(len(seeds) >= number):
            break
        # sample in a random order from the sequence
        rand_order = np.random.permutation(len(seq))
        curr_seeds_per_seq=0
        for index in rand_order:
            motif = seq[index]
            if curr_seeds_per_seq >= seeds_per_seq:
                break
            try:
                for j,motif2 in enumerate(seeds):
                    distance = motif2.distance(
                        motif.as_vector(),
                        vec=True,
                        cache=True,
                    )
                    if distance < threshold:
                        coin_flip = np.random.randint(0,2)
                        if coin_flip:
                            seeds[j] = motif
                        raise MotifMatch(distance)
                seeds.append(motif)
                curr_seeds_per_seq += 1
                values.append(records_shuffled.values[i])
            except MotifMatch as e:
                continue
    values = np.array(values)
    for value in np.unique(values):
        logging.info("Seeds in Cat {}: {}".format(value, np.sum(values == value)))
    return seeds

def evaluate_seeds(records, motifs, threshold_match, rc):
    """ Function to evaluate a set of seeds and return the results in a list

    Args:
        records - full sequence database. This is read only
        motifs - set of motifs, again read only
        threshold_match - distance threshold to be considered for a match
        rc - test the reverse complement of the motif or not
    """
    seeds = []
    for motif in motifs:
        this_entry = {}
        this_discrete = generate_peak_vector(records, motif, threshold_match, rc=rc)
        this_entry['mi'] = records.mutual_information(this_discrete)
        this_entry['motif'] = motif
        this_entry['discrete'] = this_discrete
        this_entry['threshold'] = threshold_match
        seeds.append(this_entry)
    return seeds

def fast_evaluate_seeds(recs, possible_motifs, threshold, rc, dist):
    """Gets hits for each potential motif to the input data and adds
    mutual information and other helpful tidbits.
    
    Args:
    -----
    recs : inout.SeqDatabase object
    possible_motifs : list
    threshold : float
    rc : bool
    dist : function
    """
    
    recs.shape_vectors_to_3d_array()

    if rc:
        window_arr = recs.flat_windows[:,::-1,:]
    else:
        window_arr = recs.flat_windows[...]

    rec_num,_,win_num = window_arr.shape
    seed_num = rec_num*win_num
    hits = np.zeros((seed_num, rec_num))
    
    # hits is modified in place
    fast_generate_peak_array(
        window_arr,
        threshold,
        hits,
        rec_num,
        win_num,
        dist,
    )
    
    seeds = []
    for i in range(seed_num):
        seed_info = {}
        these_hits = hits[i,:]
        seed_info['mi'] = recs.mutual_information(these_hits)
        seed_info['motif'] = possible_motifs[i]
        seed_info['discrete'] = these_hits
        seed_info['threshold'] = threshold
        seeds.append(seed_info)
        
    return(seeds)

def add_seed_metadata(records, seed):
    seed['motif_entropy'] = inout.entropy(seed['discrete'])
    seed['category_entropy'] = records.shannon_entropy()
    seed['enrichment'] = records.calculate_enrichment(seed['discrete'])

def print_top_motifs(motifs, n= 5, reverse=True):
    """
    Function to print the top motifs sorted by MI

    Args
        motifs (list) - motifs to sort
        n (int) - number of motifs to print
        reverse (bool) - sort in reverse
    Modifys
        stdout through the logging function
    """

    sorted_motifs = sorted(
        motifs,
        key=lambda x: x['mi'],
        reverse=reverse,
    )
    if reverse:
        logging.debug("Printing top {} motifs.".format(n))
    else:
        logging.debug("Printing bottom {} motifs.".format(n))

    for motif in sorted_motifs[0:n]:
        logging.debug("Motif MI: {}\n Motif Mem: {}\n{}".format(
            motif['mi'],
            motif['motif'],
            motif['motif'].as_vector(),
        ))

def save_mis(motifs, out_pre):
    """
    Function to save the MIs for each motif in a txt file
    """
    all_mis = []
    for motif in motifs:
        all_mis.append(motif['mi'])
    np.savetxt(out_pre+".txt", all_mis)

def mp_optimize_motifs_helper(args):
    """ Helper function to allow motif optimization to be multiprocessed
    
    Args:
        args (list) - list containing starting motif dict, category, and sample
                      percentage
    
    Returns:
        final_motif_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """
    motif, data, sample_perc = args
    #this_data = data.random_subset_by_class(sample_perc)
    final_motif_dict = {}
    func_info = {"NFeval":0, "eval":[], "value":[]}
    motif_to_optimize = list(motif['motif'].as_vector(cache=True))
    motif_to_optimize.append(motif['threshold'])
    final_opt = opt.minimize(optimize_mi,motif_to_optimize, 
                             args=(data, sample_perc, func_info), 
                             method="nelder-mead")
    final = final_opt['x']
    threshold = final[-1]
    final_motif = dsp.ShapeParams()
    final_motif.from_vector(motif['motif'].names, final[:-1])
    final_motif_dict['motif'] = final_motif
    discrete = generate_peak_vector(data, final_motif.as_vector(cache=True), 
                                    threshold = threshold, rc=args.rc)
    final_motif_dict['threshold'] = threshold
    final_motif_dict['enrichment'] = data.calculate_enrichment(discrete)
    final_motif_dict['motif_entropy'] = inout.entropy(discrete)
    final_motif_dict['category_entropy'] = data.shannon_entropy()
    final_motif_dict['mi'] = data.mutual_information(discrete)
    final_motif_dict['discrete'] = discrete
    final_motif_dict['opt_success'] = final_opt['success']
    final_motif_dict['opt_message'] = final_opt['message']
    final_motif_dict['opt_iter'] = final_opt['nit']
    final_motif_dict['opt_func'] = final_opt['nfev']
    final_motif_dict['opt_info'] = func_info

    return final_motif_dict

def mp_evaluate_motifs(data, motifs, threshold_match, rc, p=1):
    """Perform motif evaluation in a multiprocessed manner

    Args:
        motifs (list) - motifs to evaluate
        data (cat) - A sequence category object
        p (int) - number of processors

    Returns:
        list of evaluated motif dictionaries
    """
    # determine how to chunk the motifs:
    motifs_chunked = [motifs[i::p] for i in range(0,p)]
    pool = mp.Pool(processes=p)
    out_motifs = pool.map(mp_evaluate_motifs_helper, 
            ((data, motifs, threshold_match, rc) for motifs in motifs_chunked))
    pool.close()
    pool.join()
    # flatten list
    final_motifs = []
    for processor in out_motifs:
        for motif in processor:
            final_motifs.append(motif)
    return final_motifs

def mp_evaluate_motifs_helper(these_args):
    """ Helper function for doing evaluation in a multiprocessed way
    
    Args:
        these_args (list) - list of args i.e. database, list of motifs,
                            match cutoff, reverse complement

    Returns:
        list of evaluated motif dictionaries
    """
    data, motifs, threshold_match, rc = these_args
    these_motifs = evaluate_motifs(data, motifs, threshold_match, rc)
    return these_motifs

        
def mp_optimize_motifs(motifs, data, sample_perc, p=1):
    """Perform motif optimization in a multiprocessed manner
    
    Args:
        motifs (list) - List containing motif dictionaries
        data (SeqDatabase) - Full dataset to train on
        sample_perc (float) - percentage of data to sample for optimization
    
    Returns:
        final_motif_dict (dict) - A dictionary containing final motif and
                                 additional values that go with it
    """
    pool = mp.Pool(processes=p)
    this_data = data.random_subset_by_class(sample_perc)
    final_motifs = pool.map(mp_optimize_motifs_helper, 
                           ((motif, this_data, sample_perc) for motif in motifs))
    pool.close()
    pool.join()
    return final_motifs

def filter_motifs(motifs, records, mi_threshold):
    """ Select initial motifs through conditional mutual information

    Args:
        motifs (list of dicts) - list of motif dictionaries
        records (SeqDatabase Class) - sequences motifs are compared against
        mi_threshold (float) - percentage of total entropy CMI must be >
    
    Returns:
        final_motifs (list of dicts) - list of passing motif dictionaries
    """
    these_motifs = sorted(motifs, key=lambda x: x['mi'], reverse=True)
    top_motifs = [these_motifs[0]]
    for cand_motif in these_motifs[1:]:
        motif_pass = True
        for good_motif in top_motifs:
            cmi = inout.conditional_mutual_information(records.get_values(), 
                                                 cand_motif['discrete'], 
                                                 good_motif['discrete'])

            mi_btwn_motifs = inout.mutual_information(cand_motif['discrete'], 
                    good_motif['discrete'])
            if mi_btwn_motifs == 0:
                ratio = 0
            else:
                ratio = cmi/mi_btwn_motifs
            if ratio < mi_threshold:
                motif_pass = False
                break
        if motif_pass:
            top_motifs.append(cand_motif)
    return top_motifs

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

@jit(nopython=True)
def calc_aic(delta_k, rec_num, mi):
    # Peter came up with this version of aic
    #  Look up wiki definition for hypergeometric case
    aic = 2*delta_k - 2*rec_num*mi
    return aic

def aic_motifs2(records):
    """Select final motifs through AIC

    Args:
        records (RecordDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_motifs (list of dicts) - list of passing motif indices in records
    """

    # get number of parameters based on window length * shape number * 2 + 1
    #  We multiply by 2 because for each shape value we have a weight,
    #  and we add 1 because the motif's threshold was a parameter.
    rec_num,win_len,shape_num,win_num = records.windows.shape
    delta_k = win_len * shape_num * 2 + 1

    # get sorted order of mi values
    # returns a tuple of arrays corresponding to (row_idx, col_idx)
    sort_r_inds,sort_c_inds = np.unravel_index(
        # slice the sorted inds in reverse to get descending order
        np.argsort(records.mi, axis=None)[::-1],
        records.mi.shape,
    )

    top_motif_inds = []
    # Make sure first motif passes AIC
    top_mi = records.mi[sort_r_inds[0],sort_c_inds[0]]
    top_aic = calc_aic(delta_k, rec_num, top_mi)
    if top_aic < 0:
        top_motif_inds = [(sort_r_inds[0],sort_c_inds[0])]
    else:
        return []

    # loop through candidate motifs
    for cand_idx in range(1,len(sort_r_inds)):

        if cand_idx % 10 == 0:
            print("Working on candidate index: {}".format(cand_idx))

        cand_r_idx = sort_r_inds[cand_idx]
        cand_c_idx = sort_c_inds[cand_idx]
        cand_mi = records.mi[cand_r_idx, cand_c_idx]
        cand_hits = records.hits[cand_r_idx, cand_c_idx, :, :]

        motif_pass = True

        # if the total MI for this motif doesn't pass AIC skip it
        cand_aic = calc_aic(delta_k, rec_num, cand_mi)
        print("Candidate {} AIC: {}".format(cand_idx,cand_aic))
        if cand_aic > 0:
            continue

        for good_motif_r,good_motif_c in top_motif_inds:

            good_motif_hits = records.hits[good_motif_r, good_motif_c, :, :]

            # check the conditional mutual information for this motif with
            # each of the chosen motifs
            distinct_good_motif_hits = np.unique(good_motif_hits, axis=0)
            distinct_y = np.unique(records.y)
            distinct_cand_hits = np.unique(cand_hits, axis=0)
            this_cmi = inout.conditional_mutual_information(
                records.y, 
                distinct_y,
                cand_hits, 
                distinct_cand_hits,
                good_motif_hits,
                distinct_good_motif_hits,
            )
            this_aic = calc_aic(delta_k, rec_num, this_cmi)

            # if candidate motif doesn't improve model as added to each of the
            # chosen motifs, skip it
            if this_aic > 0:
                motif_pass = False
                break

        if motif_pass:
            top_motif_inds.append((cand_r_idx,cand_c_idx))

    return top_motif_inds

def aic_motifs(motifs, records):
    """Select final motifs through AIC

    Args:
        motifs (list of dicts) - list of final motif dictionaries
        records (SeqDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_motifs (list of dicts) - list of passing motif dictionaries
    """

    # get number of parameters based on window length * shape number * 2 + 1
    #  We multiply by 2 because for each shape value we have a weight,
    #  and we add 1 because the motif's threshold was a parameter.
    rec_num,win_len,shape_num,win_num = records.windows.shape
    delta_k = win_len * shape_num * 2 + 1

    # sort motifs by mutual information
    these_motifs = sorted(
        motifs,
        key=lambda x: x['mi'],
        reverse=True,
    )

    # Make sure first motif passes AIC
    if calc_aic(delta_k, rec_num, these_motifs[0]['mi']) < 0:
        top_motifs = [these_motifs[0]]
    else:
        return []

    # loop through candidate motifs
    for cand_motif in these_motifs[1:]:
        motif_pass = True

        # if the total MI for this motif doesn't pass AIC skip it
        if calc_aic(delta_k, rec_num, cand_motif['mi']) > 0:
            continue

        for good_motif in top_motifs:
            # check the conditional mutual information for this motif with
            # each of the chosen motifs
            ###########################################################
            ###########################################################
            ### I need to speed this step up!! ########################
            ###########################################################
            ###########################################################
            this_mi = inout.conditional_mutual_information(
                records.y, 
                cand_motif['hits'], 
                good_motif['hits'],
            )

            # if candidate motif doesn't improve model as added to each of the
            # chosen motifs, skip it
            if calc_aic(delta_k, rec_num, this_mi) > 0:
                motif_pass = False
                break

        if motif_pass:
            top_motifs.append(cand_motif)

    return top_motifs

def bic_motifs(motifs, records):
    """ Select final motifs through BIC

    Args:
        motifs (list of dicts) - list of final motif dictionaries
        records (SeqDatabase Class) - sequences motifs are compared against
    
    Returns:
        final_motifs (list of dicts) - list of passing motif dictionaries
    """
    delta_k = len(motifs[0]['motif'].as_vector(cache=True))
    n = len(records)
    these_motifs = sorted(motifs, key=lambda x: x['mi'], reverse=True)
    if 2*delta_k*np.log2(n) - 2*n*these_motifs[0]['mi'] < 0:
        top_motifs = [these_motifs[0]]
    else:
        return []
    for cand_motif in these_motifs[1:]:
        motif_pass = True
        if 2*delta_k*np.log2(n) - 2*n*cand_motif['mi'] > 0:
            continue
        for good_motif in top_motifs:
            this_mi = inout.conditional_mutual_information(
                records.get_values(), 
                cand_motif['discrete'], 
                good_motif['discrete'],
            )
            if 2*delta_k*np.log2(n) - 2*n*this_mi > 0:
                motif_pass = False
                break
        if motif_pass:
            top_motifs.append(cand_motif)
    return top_motifs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', action='store', type=str,
                         help='input text file with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--kmer', type=int,
                         help='kmer size to search for. Default=15', default=15)
    parser.add_argument('--ignorestart', type=int,
                         help='# bp to ignore at start of each sequence. Default=2',
                         default=2)
    parser.add_argument('--ignoreend', type=int,
                         help='# bp to ignore at end of each sequence. Default=2',
                         default=2)
    parser.add_argument('--search_method', type=str, default="greedy",
                        help="search method for initial seeds. Options: greedy, brute. Default=greedy")
    parser.add_argument('--num_seeds', type=int,
                         help='cutoff for number of seeds to test. Default=1000. Only matters for greedy search',
                        default=1000)
    parser.add_argument('--seeds_per_seq', type=int,
                         help='max number of seeds to come from a single sequence. Default=1. Only matters for greedy search.',
                        default=1)
    parser.add_argument('--seeds_per_seq_thresh', type=int,
                         help='max number of seeds to come from a single sequence. Default=1',
                        default=1)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--threshold_perc', type=float, default=0.05,
            help="fraction of data to determine threshold on. Default=0.05")
    parser.add_argument('--threshold_seeds', type=float, default=2.0, 
            help="std deviations below mean for seed finding. Only matters for greedy search. Default=2.0")
    parser.add_argument('--threshold_match', type=float, default=2.0, 
            help="std deviations below mean for match threshold. Default=2.0")
    parser.add_argument('--optimize_perc', type=float, default=0.1, 
            help="fraction of data to optimize on. Default=0.1")
    parser.add_argument('--motif_perc', type=float, default=1,
            help="fraction of data to EVALUATE motifs on. Default=1")
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--optimize', action="store_true",
            help="optimize motifs with Nelder Mead?")
    parser.add_argument('--mi_perc', type=float, default=5,
            help="ratio of CMI/MI to include an additional motif. Default=5")
    parser.add_argument('--infoz', type=int, default=2000, 
            help="Calculate Z-score for final motif MI with n data permutations. default=2000. Turn off by setting to 0")
    parser.add_argument('--inforobust', type=int, default=10, 
            help="Calculate robustness of final motif with x jacknifes. Default=10. Requires infoz to be > 0.")
    parser.add_argument('--fracjack', type=int, default=0.3, 
            help="Fraction of data to hold out in jacknifes. Default=0.3.")
    parser.add_argument('--distance_metric', type=str, default="constrained_manhattan",
            help="distance metric to use, manhattan using constrained weights is the only supported one for now")
    parser.add_argument('--seed', type=int, default=None,
            help="set the random seed, default=None, based on system time")
    parser.add_argument('--rc', action="store_true",
            help="search the reverse complement with each motif as well?")
    parser.add_argument('-o', type=str, default="motif_out_")
    parser.add_argument('-p', type=int, default=1, help="number of processors. Default=1")
    parser.add_argument("--debug", action="store_true",
            help="print debugging information to stderr. Write extra txt files.")
    parser.add_argument('--txt_only', action='store_true', help="output only txt files?")
    parser.add_argument('--save_opt', action='store_true', help="write motifs to pickle file after initial weights optimization step?")

    numba.set_num_threads(p)
    
    args = parser.parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level) 
    logging.getLogger('matplotlib.font_manager').disabled = True
    outpre = args.o
    # choose a random seed
    if args.seed:
        np.random.seed(args.seed)
    
    logging.info("Reading in files")
    # read in the fasta files containing parameter information
    # returns an inout.FastaFile obj for each param
    # all_params = [read_parameter_file(x) for x in args.params]
    # possible distance metrics that could be used
    dist_met = {"constrained_manhattan": inout.constrained_manhattan_distance,
                "manhattan": inout.manhattan_distance, 
                "hamming": inout.hamming_distance,
                "euclidean": inout.euclidean_distance}
    # store the distance metric chosen
    this_dist = dist_met[args.distance_metric]
    shape_fname_dict = {n:fn for n,fn in zip(args.param_names, args.params)}
    logging.info("Reading input data and shape info.")
    records = inout.RecordDatabase(args.infile, shape_fname_dict)

    # create an empty sequence database to store the sequences in
    #records = inout.SeqDatabase()
    
    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        #records.read(args.infile, float)
        logging.info("Discretizing data")
        records.discretize_quant(args.continuous)
    #else:
    #    records.read(args.infile, int)
    logging.info("Distribution of sequences per class:")
    logging.info(seqs_per_bin(records))

    # add parameter values for each sequence
    #for name, record in zip(records.names, records):
    #    for param, param_name in zip(all_params, args.param_names):
    #        record.add_shape_param(
    #            dsp.ShapeParamSeq(
    #                param_name,
    #                param.pull_entry(name).seq,
    #            ),
    #        )
    #        record.metric = this_dist

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

    logging.info("Computing all windows and initializing weights array for distance calculation.")
    records.compute_windows(wsize = args.kmer)
    records.initialize_weights()

    logging.info("Determining initial threshold")
    if args.distance_metric == "hamming":
        threshold_match = 4
        logging.info(
            "Using {} as an initial match threshold".format(threshold_match)
        )
    else:
        records.set_initial_thresholds(
            dist = dist_met,
            threshold_sd_from_mean = args.threshold_seeds
        )
        #mean,stdev = find_initial_threshold(
        #    records.random_subset_by_class(args.threshold_perc),
        #    args.seeds_per_seq_thresh,
        #)
        #threshold_seeds = max(mean - args.threshold_seeds*stdev, 0)
        #threshold_match = max(mean - args.threshold_match*stdev, 0)
        logging.info("Using {} as an initial match threshold".format(threshold_match))
    # generate initial MI score for the given shapes, weights, and threshold
    records.compute_mi(dist_met)
    with open('initial_mutual_information.pkl','wb') as f:
        pickle.dump(records.mi, f)

    raise()
    #if args.search_method == "greedy":
    #    logging.info(
    #        "Greedy search for possible motifs with threshold {}".format(threshold_seeds)
    #    )
    #    possible_motifs = greedy_search2(
    #        records,
    #        threshold_seeds,
    #        args.num_seeds,
    #        args.seeds_per_seq,
    #    )
    #else:
    logging.info("Testing all seeds and optimizing weights to generate motifs")
    #debugger = records.shuffle()
    # double for loop list comprehension
    #possible_motifs = [
    #    motif
    #    for a_seq in records.iterate_through_precompute()
    #    for motif in a_seq
    #]

    rec_num, win_length, shape_num, win_num = records.windows.shape
    motif_num = rec_num * win_num
    logging.info("{} motifs to optimize".format(motif_num))
    logging.info("Optimizing weights and calculating MI for all optimized motifs")

    #if args.seed_perc != 1:
    #    this_records, other_records = records.random_subset_by_class(args.motif_perc, split=True)
    #else:
    #    this_records = records
    #    other_records = records
    this_records = records
    other_records = records
    #if args.debug:
    #    other_records.write(outpre + "_lasso_seqs.txt")
    logging.info("Distribution of sequences per class for motif screening and regression (train set)")
    logging.info(seqs_per_bin(this_records))
    #logging.info("Distribution of sequences per class for CMI and final evaluation (test set)")
    #logging.info(seqs_per_bin(other_records))
    logging.info("Optimizing {} motifs over {} processor(s)".format(
        motif_num, args.p
    ))

    # actually run the optimization of weights, returns
    #   a list of dictionaries, each dictionary containing a motif's
    #   shapes, original weights, optimized weights, original MI,
    #   optimized MI, and other information.
    all_motifs = mp_optimize(
        records,
        dist_met,
        p = args.p,
    )

    if args.save_opt:
        with open(outpre+"_optimized_motifs.pkl", "wb") as pkl_f:
            pickle.dump(all_motifs, pkl_f)

    #all_motifs = mp_evaluate_seeds(
    #    this_records,
    #    possible_motifs,
    #    threshold_match,
    #    args.rc,
    #    p=args.p,
    #)
    #all_motifs = fast_evaluate_seeds(
    #    this_records,
    #    possible_motifs,
    #    threshold_match,
    #    args.rc,
    #    this_dist,
    #)

    if args.debug:
        print_top_motifs(all_motifs)
        print_top_motifs(all_motifs, reverse=False)
        save_mis(all_motifs, outpre+"_all_motifs_mi")

    logging.info("Filtering motifs by AIC individually")

    good_motifs = []
    for motif in all_motifs:
        passed = aic_motifs([motif], this_records)
        if len(passed) > 0:
            good_motifs.append(motif) 
    if len(good_motifs) < 1: 
        logging.info("No motifs found")
        sys.exit()
    if args.debug:
        print_top_motifs(good_motifs)
        print_top_motifs(good_motifs, reverse=False)

    logging.info("{} motifs survived".format(len(good_motifs)))

    logging.info("Finding minimum match scores for each motif")
    if args.debug:
        good_motif_pkl_fname = "{}_good_motifs.pkl".format(outpre)
        logging.info("Writing motifs after CMI filter and prior to regression to {}.".format(good_motif_pkl_fname))
        with open(good_motif_pkl_fname, "wb") as pkl_f:
            pickle.dump(good_motifs, pkl_f)
        #outmotifs = inout.ShapeMotifFile()
        #outmotifs.add_motifs(good_motifs)
        #outmotifs.write_file(outpre+"_called_motifs_before_regression.dsp", records)

    X = [
        generate_dist_vector(this_records, this_motif['motif'], rc=args.rc)
        for this_motif in good_motifs
    ] 
    X = np.stack(X, axis=1)
    X = StandardScaler().fit_transform(X)
    y = this_records.get_values()
    logging.info("Running L1 regularized logistic regression with CV to determine reg param")

    clf = LogisticRegressionCV(
        Cs=100,
        cv=5,
        multi_class='multinomial',
        penalty='l1',
        solver='saga',
        max_iter=10000,
    ).fit(X, y)

    best_c = cvlogistic.find_best_c(clf)

    clf_f = LogisticRegression(
        C=best_c,
        multi_class='multinomial',
        penalty='l1',
        solver='saga',
        max_iter=10000,
    ).fit(X,y)

    good_motif_index = cvlogistic.choose_features(clf_f, tol=0)
    if len(good_motif_index) < 1:
        logging.info("No motifs found")
        sys.exit()

    cvlogistic.write_coef_per_class(clf_f, outpre + "_coef_per_class.txt")
    final_good_motifs = [good_motifs[index] for index in good_motif_index]
    logging.info("{} motifs survived".format(len(final_good_motifs)))
    if args.debug:
        cvlogistic.plot_score(clf, outpre+"_score_logit.png")
        for cls in clf.classes_:
            cvlogistic.plot_coef_paths(clf, outpre+"_coef_path%s.png"%cls)
        print_top_motifs(final_good_motifs)
        print_top_motifs(final_good_motifs, reverse=False)
        logging.info("Writing motifs after regression")
        outmotifs = inout.ShapeMotifFile()
        outmotifs.add_motifs(final_good_motifs)
        outmotifs.write_file(outpre+"_called_motifs_after_regression.dsp", records)

    for motif in final_good_motifs:
        add_motif_metadata(this_records, motif) 
        logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
        logging.info("MI: {}".format(motif['mi']))
        logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
        logging.info("Category Entropy: {}".format(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat {} is {}".format(
                key,
                motif['enrichment'][key]
            ))
            logging.info("Enrichment for Cat {} is {}".format(
                key,
                two_way_to_log_odds(motif['enrichment'][key])
            ))
    logging.info("Generating initial heatmap for passing motifs")
    if len(final_good_motifs) > 25:
        logging.info("Only plotting first 25 motifs")
        enrich_hm = smv.EnrichmentHeatmap(final_good_motifs[:25])
    else:
        enrich_hm = smv.EnrichmentHeatmap(final_good_motifs)

    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_before_hm.txt")
    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_before_hm.pdf")
        enrich_hm.display_motifs(outpre+"motif_before_hm.pdf")
    if args.optimize:
        logging.info("Optimizing motifs using {} processors".format(args.p))
        final_motifs = mp_optimize_motifs(
            final_good_motifs,
            other_records,
            args.optimize_perc,
            p=args.p,
        )
        if args.optimize_perc != 1:
            logging.info("Testing final optimized motifs on full database")
            for i,this_entry in enumerate(final_motifs):
                logging.info("Computing MI for motif {}".format(i))
                this_discrete = generate_peak_vector(
                    other_records,
                    this_entry['motif'],
                    this_entry['threshold'],
                    args.rc,
                )
                this_entry['mi'] = other_records.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = other_records.shannon_entropy()
                this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
    else:
        if args.motif_perc != 1:
            logging.info("Testing final optimized motifs on held out database")
            for i,this_entry in enumerate(final_good_motifs):
                logging.info("Computing MI for motif {}".format(i))
                this_discrete = generate_peak_vector(
                    other_records,
                    this_entry['motif'],
                    this_entry['threshold'],
                    args.rc,
                )
                this_entry['mi'] = other_records.mutual_information(this_discrete)
                this_entry['motif_entropy'] = inout.entropy(this_discrete)
                this_entry['category_entropy'] = other_records.shannon_entropy()
                this_entry['enrichment'] = other_records.calculate_enrichment(this_discrete)
                this_entry['discrete'] = this_discrete
        final_motifs = final_good_motifs

    logging.info(
        "Filtering motifs by Conditional MI using {} as a cutoff".format(args.mi_perc)
    )
    novel_motifs = filter_motifs(
        final_motifs,
        other_records,
        args.mi_perc,
    )

    if args.debug:
        print_top_motifs(novel_motifs)
        print_top_motifs(novel_motifs, reverse=False)
    logging.info("{} motifs survived".format(len(novel_motifs)))
    for i, motif in enumerate(novel_motifs):
        logging.info("motif: {}".format(motif['motif'].as_vector(cache=True)))
        logging.info("MI: {}".format(motif['mi']))
        if args.infoz > 0:
            logging.info("Calculating Z-score for motif {}".format(i))
            # calculate zscore
            zscore, passed = info_zscore(
                motif['discrete'],
                other_records.get_values(),
                args.infoz,
            )
            motif['zscore'] = zscore
            logging.info("Z-score: {}".format(motif['zscore']))
        if args.infoz > 0 and args.inforobust > 0:
            logging.info("Calculating Robustness for motif {}".format(i))
            num_passed = info_robustness(
                motif['discrete'],
                other_records.get_values(), 
                args.infoz,
                args.inforobust,
                args.fracjack,
            )
            motif['robustness'] = "{}/{}".format(num_passed,args.inforobust)
            logging.info("Robustness: {}".format(motif['robustness']))
        logging.info("Motif Entropy: {}".format(motif['motif_entropy']))
        logging.info("Category Entropy: {}".format(motif['category_entropy']))
        for key in sorted(motif['enrichment'].keys()):
            logging.info("Two way table for cat {} is {}".format(
                key,
                motif['enrichment'][key]
            ))
            logging.info("Enrichment for Cat {} is {}".format(
                key,
                two_way_to_log_odds(motif['enrichment'][key])
            ))
        if args.optimize:
            logging.info("Optimize Success?: {}".format(motif['opt_success']))
            logging.info("Optimize Message: {}".format(motif['opt_message']))
            logging.info("Optimize Iterations: {}".format(motif['opt_iter']))
    logging.info("Generating final heatmap for motifs")
    enrich_hm = smv.EnrichmentHeatmap(novel_motifs)
    enrich_hm.enrichment_heatmap_txt(outpre+"_enrichment_after_hm.txt")

    if not args.txt_only:
        enrich_hm.display_enrichment(outpre+"_enrichment_after_hm.pdf")
        enrich_hm.display_motifs(outpre+"_motif_after_hm.pdf")
        if args.optimize:
            logging.info("Plotting optimization for final motifs")
            enrich_hm.plot_optimization(outpre+"_optimization.pdf")

    logging.info("Writing final motifs")
    outmotifs = inout.ShapeMotifFile()
    outmotifs.add_motifs(novel_motifs)
    outmotifs.write_file(outpre+"_called_motifs.dsp", records)
    #final = opt.minimize(lambda x: -optimize_mi(x, data=records, sample_perc=args.optimize_perc), motif_to_optimize, method="nelder-mead", options={'disp':True})
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=records), motif_to_optimize)
    #logging.info(final)
