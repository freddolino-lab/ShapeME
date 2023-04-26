#!/usr/bin/env python3

"""
Wrapper script for identifying sequence motifs using streme
"""

import argparse
import inout
import subprocess
import tempfile
import sys
import os
import numpy as np

def run_streme(seq_fname, yvals_fname, positive_cats, threshold, out_direc):
    '''Runs streme to find motifs present in `seq_fname` enriched
    in peak regions identified as 1 in `yvals_fname`.

    Args:
    -----
    seq_fname : str
        Absolute path to file containing fasta records corresponding
        to the values in yvals_fname.
    yvals_fname : str
        Path to a npy file containing integer category assignments for each
        sequence in seq_fname fasta file.
    positive_cats : str
        In the case with more than two categories in y-values, denotes which
        categories to "1". Must take the
        form of a comma-separated list.
        For example, \"4\" would use category
        4 as the positive set and all other as negative. By contrast, \"3,4\"
        would use 3 and 4 as the positive set and all others as the negative set.
    threshold : float
        p-value threshold passed to the streme call. See streme documentation
        for details.
    out_direc : str
        Absolute path to the directory to be created, or clobbered if it already
        exists, by streme. Will contain streme output.
    '''

    fa_file = inout.FastaFile()
    pos_fa_file = inout.FastaFile()
    neg_fa_file = inout.FastaFile()
    pos_cats = [ int(_) for _ in positive_cats.split(",") ]

    print(f"pos_cats: {pos_cats}")

    with open(seq_fname, "r") as f:
        fa_file.read_whole_file(f)
    with open(yvals_fname, "rb") as f:
        yvals = np.load(f)
        
    for (i,yval) in enumerate(yvals):
        name = fa_file.names[i]
        entry = fa_file.pull_entry(name)
        if yval in pos_cats:
            pos_fa_file.add_entry(entry)
        else:
            neg_fa_file.add_entry(entry)

    with tempfile.NamedTemporaryFile("w") as pos_f:
        pos_fa_file.write(pos_f)
        tmp_pos = pos_f.name

        with tempfile.NamedTemporaryFile("w") as neg_f:
            neg_fa_file.write(neg_f)
            tmp_neg = neg_f.name

            STREME = f"streme --evalue --thresh {threshold} "\
                f" --p {tmp_pos} --n {tmp_neg} --dna "\
                f"--oc {out_direc}"
            print()
            print("Running streme command:")
            print(STREME)
            result = subprocess.run(STREME, shell=True, check=True)

    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_fname', action='store', type=str, required=True,
        help="fasta file containing sequences in which to search for motifs")
    parser.add_argument('--yvals_fname', action='store', type=str, required=True,
        help=f"tsv file of paired sequence names (column 1) "\
            f"and binary y-values (column 2)")
    parser.add_argument('--pos_cats', action="store", type=str, required=False,
        default="1",
        help=f"Denotes which categories in `--yvals_fname` "\
            f"to use as the positive "\
            f"set for sequence motif calling using streme. Example: "\
            f"\"4\" category 4 as the positive set, whereas \"3,4\" would use "\
            f"categories 3 and 4 as the positive set.")
    parser.add_argument('--threshold', action='store', type=str, default=0.05,
        help=f"e-value threshold for streme to continue searching for motifs. "\
            f"Default: %(default)s. See streme documentation for details")
    parser.add_argument('--out_direc', action='store', type=str, required=True,
        help=f"Absolute path to the directory to be created, or clobbered if it "\
            f"already exists, by streme. Will contain streme output.")
    args = parser.parse_args()

    result = run_streme(
        args.seq_fname,
        args.yvals_fname,
        args.pos_cats,
        args.threshold,
        args.out_direc,
    )

    print(result.stdout)
    print(result.stderr, file=sys.stderr)


if __name__ == '__main__':
    main()

