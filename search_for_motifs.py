import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np


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

def motif_match(motif_vec, data_vec, metric, threshold, direction="below"):
    if direction == "below":
        return metric(motif_vec, data_vec) < threshold
    elif direction == "above":
        return metric(motif_vec, data_vec) > threshold

    return metric(motif_vec, data_vec) < threshold

def search_for_motifs(seq, motif, threshold, metric, wsize, wstart=0, wend=None):
    matches = []
    for i, window in enumerate(seq.sliding_windows(wsize, 1, wstart, wend)):
        if motif_match(motif.as_vector(cache=True), window.as_vector(), metric, threshold):
            matches.append([i+wstart, i+wstart+wsize])
    return matches

def write_matches_bed(fhandle, name, matches, motif_name):
    for match in matches:
        fhandle.write("%s\t%i\t%i\t%s\n"%(name, match[0], match[1], motif_name))

def write_matches_count(fhandle, name, matches, motif_name):
    fhandle.write("%s\t%s\t%i\n"%(name,motif_name,len(matches)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')
    parser.add_argument('--motifs', type=str, help="motif with motifs to search for")
    parser.add_argument('--ignorestart', type=int,
                         help='# bp to ignore at start of each sequence', default=2)
    parser.add_argument('--ignoreend', type=int,
                         help='# bp to ignore at end of each sequence', default=2)
    parser.add_argument('--outfmt', type=str, default=".bed", help=".bed for full matches or .txt for counts")
    parser.add_argument('-o', type=str, default="matches_out_")

    logging.warning("Reading in files")

    args = parser.parse_args()
    write_types = {".bed": write_matches_bed, ".txt": write_matches_count}
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
    outfile = open(args.o+args.outfmt, mode="w")
    for i, motif in enumerate(motifs):
        logging.warning("Normalizing parameters")
        genome.set_center_spread(motifs.cent_spreads[i])
        genome.normalize_params()
        for name, seq in zip(genome.names, genome):
            matches = search_for_motifs(seq, motif['seed'], motif['threshold'], 
                                        dsp.manhattan_distance, len(motif['seed']),
                                        wstart=args.ignorestart, wend=args.ignoreend)
            write(outfile, name, matches, motif['name'])
        genome.unnormalize_params()
    outfile.close()
