import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import sys
import json
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import cvlogistic

from pathlib import Path

this_path = Path(__file__).parent.absolute()
rust_bin = os.path.join(this_path, '../rust_utils/target/release/search_for_motifs')

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

def motif_score(motif_vec, data_vec, metric):
    return metric(motif_vec, data_vec)

def search_for_motifs(seq, motif, threshold, metric, wsize, wstart=0, wend=None, threshold_above=False, rc=False):
    matches = []
    motif_vec = motif.as_vector()
    if args.rc:
        motif.rev_comp()
        motif_vec_rc = motif.as_vector()
        motif.rev_comp()
    for i, window in enumerate(seq.sliding_windows(wsize, 1, wstart, wend)):
        score = motif_score(motif_vec, window.as_vector(), metric)
        above_thresh = score > threshold
        if above_thresh == threshold_above:
            matches.append([i+wstart, i+wstart+wsize, score, "+"])
        if rc:
            score = motif_score(motif_vec_rc, window.as_vector(), metric)
            above_thresh = score > threshold
            if above_thresh == threshold_above:
                matches.append([i+wstart, i+wstart+wsize, score, "-"])
    return matches

def write_matches_bed(fhandle, name, matches, motif_name):
    for match in matches:
        fhandle.write("%s\t%i\t%i\t%s\t%.4f\t%s\n"%(name, match[0], match[1], motif_name, match[2], match[3]))

def write_matches_fimo(fhandle, name, matches, motif_name):
    for match in matches:
        fhandle.write("%s\t%s\t%i\t%i\t%s\t%.4f\t.\t.\t.\n"%(motif_name,name, match[0], match[1], match[3], match[2]))

def write_matches_count(fhandle, name, matches, motif_name):
    fhandle.write("%s\t%s\t%i\n"%(name,motif_name,len(matches)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfiles with shape scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--data_dir', type=str, help="Directory containing data")
    parser.add_argument('--out_dir', type=str, help="Directory to which to write outputs")
    #parser.add_argument('--motifs', type=str, help="motif with motifs to search for")
    #parser.add_argument('--threshold', type=float, help="overridding threshold for a match",
    #        default=None)
    #parser.add_argument('--threshold_above', action='store_true', 
    #        help="consider matches as being above the threshold")
    #parser.add_argument('--ignorestart', type=int,
    #                     help='# bp to ignore at start of each sequence', default=2)
    #parser.add_argument('--ignoreend', type=int,
    #                     help='# bp to ignore at end of each sequence', default=2)
    #parser.add_argument('--outfmt', type=str, default=".bed",
    #        help=".bed for full matches or .txt for counts or .fimo for fimo-like format")
    #parser.add_argument('--rc', action="store_true",
    #        help="search the reverse complement with each seed as well?")
    #parser.add_argument('-o', type=str, default="-")

    logging.warning("Reading in files")

    args = parser.parse_args()

    in_direc = args.data_dir
    out_direc = args.out_dir
    out_direc = os.path.join(in_direc, out_direc)

    config_fname = os.path.join(out_direc, 'config.json')
    
    with open(config_fname, 'r') as f:
        args_dict = json.load(f)

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
    records.shape_centers = args_dict['centers']
    records.shape_spreads = args_dict['spreads']
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

    logging.info("Distribution of sequences per class:")
    logging.info(seqs_per_bin(records))

    logging.info("Getting distance between motifs and each record")

    RUST = "{} {}".format(
        rust_bin,
        config_fname,
    )

    retcode = subprocess.call(RUST, shell=True)

    if retcode != 0:
        sys.exti("Rust binary returned non-zero exit status")

    good_motifs = inout.read_motifs_from_rust(os.path.join(out_direc, "evaluated_motifs.json"))


    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    # Bring in logistic regression parameters
    # Run the prediction using the trained logistic reg model
    # Get the precision/recall curve and AUC

    write_types = {".bed": write_matches_bed, ".txt": write_matches_count,
            ".fimo": write_matches_fimo}
    try:
        write = write_types[args.outfmt]
    except KeyError:
        raise ValueError("%s not recognized as valid output fmt"%(args.outfmt))

    motifs = inout.ShapeMotifFile()
    motifs.read_file(args.motifs)
    for key in motifs.cent_spreads[0]:
        if key not in args.param_names:
            raise ValueError("Missing %s file needed to search with motifs"%(key))

    all_params = [read_parameter_file(x) for x in args.params]
    # creating a seqdatabase object for easy iteration. NOT using the category
    # part of the object and functions associated with that part will not
    # work
    genome = inout.SeqDatabase(names=[])
    ## MIGHT WANT TO CONSIDER MOVING THIS UGLY INITALIZATION AS METHOD IN
    ## THE ACTUAL CLASS
    # populate SeqDatabase with empty param vectors
    for name in all_params[0].names:
        genome.names.append(name)
        # don't care about these values but might as well populate them
        genome.values.append(0)
        # this is really what a I needed, a parameter object for each
        # sequence
        genome.params.append(dsp.ShapeParams(data={}, names=[]))
    genome.values = np.array(genome.values)
    # also might want to make this a method for SeqDatabase
    for name, param in zip(genome.names, genome):
        for this_param, this_param_name in zip(all_params, args.param_names):
            try:
                param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))
            except KeyError:
                raise KeyError("Missing entry %s in fasta file for %s"%(name, this_param_name))
    if args.o == "-":
        outfile = sys.stdout
    else:
        outfile = open(args.o+args.outfmt, mode="w")
    for i, motif in enumerate(motifs):
        if args.threshold:
            this_threshold = args.threshold
        else:
            this_threshold = motif['threshold']
        logging.warning("Searching with motif %s and threshold %.4f"%(motif['name'], this_threshold))
        logging.warning("Normalizing parameters")
        genome.set_center_spread(motifs.cent_spreads[i])
        genome.normalize_params()
        for name, seq in zip(genome.names, genome):
            matches = search_for_motifs(seq, motif['seed'], this_threshold, 
                                        dsp.manhattan_distance, len(motif['seed']),
                                        wstart=args.ignorestart, wend=args.ignoreend,
                                        threshold_above = args.threshold_above, rc=args.rc)
            write(outfile, name, matches, motif['name'])
        genome.unnormalize_params()
    outfile.close()
