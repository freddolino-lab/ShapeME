import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import sys


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

def search_for_motifs(seq, motif, threshold, metric, wsize, wstart=0, wend=None, threshold_above=False):
    matches = []
    for i, window in enumerate(seq.sliding_windows(wsize, 1, wstart, wend)):
        score = motif_score(motif.as_vector(cache=True), window.as_vector(), metric)
        above_thresh = score > threshold
        if above_thresh == threshold_above:
            matches.append([i+wstart, i+wstart+wsize, score])
    return matches

def write_matches_bed(fhandle, name, matches, motif_name):
    for match in matches:
        fhandle.write("%s\t%i\t%i\t%s\t%.4f\n"%(name, match[0], match[1], motif_name, match[2]))

def write_matches_fimo(fhandle, name, matches, motif_name):
    for match in matches:
        fhandle.write("%s\t%s\t%i\t%i\t.\t%.4f\t.\t.\t.\n"%(motif_name,name, match[0], match[1], match[2]))

def write_matches_count(fhandle, name, matches, motif_name):
    fhandle.write("%s\t%s\t%i\n"%(name,motif_name,len(matches)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--motifs', type=str, help="motif with motifs to search for")
    parser.add_argument('--threshold', type=float, help="overridding threshold for a match",
            default=None)
    parser.add_argument('--threshold_above', action='store_true', 
            help="consider matches as being above the threshold")
    parser.add_argument('--ignorestart', type=int,
                         help='# bp to ignore at start of each sequence', default=2)
    parser.add_argument('--ignoreend', type=int,
                         help='# bp to ignore at end of each sequence', default=2)
    parser.add_argument('--outfmt', type=str, default=".bed",
            help=".bed for full matches or .txt for counts or .fimo for fimo-like format")
    parser.add_argument('-o', type=str, default="-")

    logging.warning("Reading in files")

    args = parser.parse_args()
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
                                        threshold_above = args.threshold_above)
            write(outfile, name, matches, motif['name'])
        genome.unnormalize_params()
    outfile.close()
