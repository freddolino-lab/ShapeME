import inout
import sys
import os
import argparse
import numpy as np
import shapemotifvis as smv
import pickle
import glob
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', action='store', type=str, required=True,
        help='input text file with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
        help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
        help='parameter names')
    parser.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    parser.add_argument('--out_dir', type=str, required=True,
        help="Direcotry (within 'data_dir') to which output files will be written.")
    parser.add_argument('-o', type=str, required=True,
        help="Prefix to apply to output files.")
    parser.add_argument('--kmer', type=int, default=15,
        help="Length of windows in which to search for motifs")
    parser.add_argument('--continuous', type=int, default=None,
        help="Number of bins to discretize continuous input data with")
        
    args = parser.parse_args()
    out_pref = args.o
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    this_dist = inout.constrained_inv_logit_manhattan_distance
    # read in shapes
    shape_fname_dict = {
        n:os.path.join(in_direc,fname) for n,fname
        in zip(args.param_names, args.params)
    }
    records = inout.RecordDatabase(
        os.path.join(in_direc, args.infile),
        shape_fname_dict,
    )

    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        records.discretize_quant(args.continuous)

    records.determine_center_spread()
    records.normalize_shape_values()

    records.compute_windows(wsize = args.kmer)
    weights = records.initialize_weights()[:,:,None]

    opt_direc = os.path.join(out_direc, "optimizations")
    opt_fname_search = os.path.join(
        opt_direc,
        "{}_optim_*_adapt_*_fatol_*_temp_*_stepsize_*_alpha_*_max_count_*_batch_*.pkl".format(out_pref),
    )
    opt_fnames = glob.glob(opt_fname_search)
    opt_plot_basename = re.sub(
        r"_batch_\d+\.pkl",
        "",
        opt_fnames[0]
    )

    opt_results = []
    for fname in opt_fnames:
        with open(fname, 'rb') as f:
            opt_results.extend(pickle.load(f))

    smv.plot_shapes_and_weights(
        opt_results,
        opt_plot_basename + "_shapes_and_weights.png",
        records,
        alpha = 0.05, # NOTE: this will need changed in the future to grab the info that will be in the pickle file. For now, that information isn't there.
        top_n = 30,
        opacity = 1,
        legend_loc = "upper left",
    )
    smv.plot_optim_trajectory(
        opt_results,
        opt_plot_basename + "_MI_trajectory.png",
        top_n = 30,
        opacity = 0.35
    )

