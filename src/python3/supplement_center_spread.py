import inout
import sys
import os
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import welfords
import pickle

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', action='store', type=str, required=True,
        help='input text file with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help="Sets the upper and lower limits on the match threshold during optimization. Defaults to 0 for the lower limit and 10 for the upper limit.")
    parser.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the shapes' z-scores during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, pre-normalized weights during optimization. Defaults to -4 for the lower limit and 4 for the upper limit.")
    parser.add_argument('--temperature', type=float, default=0.1,
        help="Sets the temperature argument for scipy.optimize.basinhopping")
    parser.add_argument('--t_adj', type=float, default=0.001,
        help="Fraction by which temperature decreases each iteration ofsimulated annealing.")
    parser.add_argument('--stepsize', type=float, default=0.25,
        help="Sets the stepsize argument for scipy.optimize.basinhopping")
    parser.add_argument('--opt_niter', type=int, default=100,
        help="Sets the number of simulated annealing iterations to undergo during optimization. Defaults to 100.")
    parser.add_argument('--kmer', type=int,
                         help='kmer size to search for. Default=15', default=15)
    parser.add_argument('--nonormalize', action="store_true",
                         help='don\'t normalize the input data by robustZ')
    parser.add_argument('--threshold_sd', type=float, default=2.0, 
            help="std deviations below mean for seed finding. Only matters for greedy search. Default=2.0")
    parser.add_argument('--init_threshold_seed_num', type=float, default=500.0, 
            help="Number of randomly selected seeds to compare to records in the database during initial threshold setting. Default=500.0")
    parser.add_argument('--init_threshold_recs_per_seed', type=float, default=20.0, 
            help="Number of randomly selected records to compare to each seed during initial threshold setting. Default=20.0")
    parser.add_argument('--init_threshold_windows_per_record', type=float, default=2.0, 
            help="Number of randomly selected windows within a given record to compare to each seed during initial threshold setting. Default=2.0")
    parser.add_argument('--motif_perc', type=float, default=1,
            help="fraction of data to EVALUATE motifs on. Default=1")
    parser.add_argument('--continuous', type=int, default=None,
            help="number of bins to discretize continuous input data with")
    parser.add_argument('--alpha', type=float, default=0.0,
            help="Lower limit on transformed weight values prior to normalization to sum to 1. Defaults to 0.0.")
    parser.add_argument('--max_count', type=int, default=1,
            help="Maximum number of times a motif can match each of the forward and reverse strands in a reference.")
    #parser.add_argument('--infoz', type=int, default=2000, 
    #        help="Calculate Z-score for final motif MI with n data permutations. default=2000. Turn off by setting to 0")
    #parser.add_argument('--inforobust', type=int, default=10, 
    #        help="Calculate robustness of final motif with x jacknifes. Default=10. Requires infoz to be > 0.")
    #parser.add_argument('--fracjack', type=int, default=0.3, 
    #        help="Fraction of data to hold out in jacknifes. Default=0.3.")
    parser.add_argument('-o', type=str, required=True, help="Prefix to apply to output files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory from which input files will be read.")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory (within 'data_dir') into which output files will be written.")
    parser.add_argument('-p', type=int, default=5,
        help="number of processors. Default=5")
    parser.add_argument('--batch_size', type=int, default=2000,
        help="Number of records to process seeds from at a time. Set lower to avoid out-of-memory errors. Default=2000")
    #parser.add_argument("--debug", action="store_true",
    #    help="print debugging information to stderr. Write extra txt files.")
    #parser.add_argument('--txt_only', action='store_true', help="output only txt files?")
    #parser.add_argument('--save_opt', action='store_true', help="write motifs to pickle file after initial weights optimization step?")

    
    args = parser.parse_args()
    out_pref = args.o
    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    if not os.path.isdir(out_direc):
        os.mkdir(out_direc)

    level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s', level=level, stream=sys.stdout) 
    logging.getLogger('matplotlib.font_manager').disabled = True

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

    # read in the values associated with each sequence and store them
    # in the sequence database
    if args.continuous is not None:
        #records.read(args.infile, float)
        #logging.info("Discretizing data")
        #records.discretize_quant(args.continuous)
        #logging.info("Quantizing input data using k-means clustering")
        records.quantize_quant(args.continuous)

    logging.info("Distribution of sequences per class:")
    logging.info(seqs_per_bin(records))

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
    optim_str = "shapes_weights_threshold"

    temp = args.temperature
    step = args.stepsize

    good_motif_out_fname = os.path.join(
        out_direc,
        "{}_post_opt_cmi_filtered_motifs_optim_{}_temp_{}_stepsize_{}_alpha_{}_max_count_{}.pkl".format(
            out_pref,
            optim_str,
            temp,
            step,
            alpha,
            max_count,
        ),
    )

    with open(good_motif_out_fname, 'rb') as inf:
        good_motifs = pickle.load(inf)

    # supplement motif data with shape center and spread from database
    good_motif_dict = {
        'names' : [],
        'indices' : [],
        'centers' : [],
        'spreads' : [],
        'motifs' : good_motifs,
    }

    for name,shape_idx in records.shape_name_lut.items():
        this_center = records.shape_centers[shape_idx]
        this_spread = records.shape_spreads[shape_idx]
        good_motif_dict['names'].append(name)
        good_motif_dict['indices'].append(shape_idx)
        good_motif_dict['centers'].append(this_center)
        good_motif_dict['spreads'].append(this_spread)

    with open(good_motif_out_fname + "center_spread.pkl", 'wb') as outf:
        pickle.dump(good_motif_dict, outf)


