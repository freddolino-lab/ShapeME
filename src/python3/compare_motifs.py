import inout as io
import logging
import argparse
import sys
import os
import numpy as np

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--ref_motifs_file", type=str, required=True,
        help=f"Meme or dsm file with a single query shape motif definition.")
    parser.add_argument("-q", "--query_motif_file", type=str,  required=True,
        help=f"Meme or dsm file with a single query shape motif definition.")
    parser.add_argument("-f", "--force_shift", type=int, default=1000000,
        help=f"Force the given shift of the query relative to the reference. Default is to try all positions in a shift.")
    parser.add_argument("--ignore_weights", action="store_true",
        help="Include at command line if only shapes should affect distances. If true, a simple weighted Manhattan distance is calculated for each comparison.")
    parser.add_argument("--end_gap_penalty", type=float, default=1,
        help="Sets the penalty for shifted alignments in which there are positions in query or reference without a match in the other. Default 1 sets the penalty such that the missing values are assumed to be the mean values for the shape parameters of interest. Different values of the penalty simply scale the resulting distances linearly.")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    ref_motifs_file = args.ref_motifs_file
    query_motif_file = args.query_motif_file
    ignore_weights_arg = args.ignore_weights
    gap_penalty = args.end_gap_penalty
    shift = args.force_shift

    ref_motifs = io.Motifs()
    ref_motifs.read_file( ref_motifs_file )
    ref_seq_motifs,ref_shape_motifs = ref_motifs.split_seq_and_shape_motifs()

    if len(ref_seq_motifs) > 0:
        sys.exit("Motif comparison not implemented for sequence motifs at this time. Use a reference motif file with only shape motifs for comparison.")

    query_motifs = io.Motifs()
    query_motifs.read_file( query_motif_file )
    query_seq_motifs,query_shape_motifs = query_motifs.split_seq_and_shape_motifs()

    if len(query_seq_motifs) > 0:
        sys.exit("Motif comparison not implemented for sequence motifs at this time. Use motif files with only one shape motif for comparison.")
    if len(query_shape_motifs) > 1:
        sys.exit("Motif comparison implemented for only a single shape motif against another signle shape motif. More than one shape motif was found in your reference motif file. Exiting now.")

    query_motif = query_shape_motifs[0]

    distances = []
    for ref_motif in ref_shape_motifs:
        results = query_motif.distance_to_motif(
            ref_motif,
            ignore_weights_arg,
            gap_penalty = gap_penalty,
            shift = shift,
        )
        distances.append(results)

    print(distances)

if __name__ == '__main__':
    main()

