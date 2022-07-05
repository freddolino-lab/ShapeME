#!/usr/bin/env python3

"""
Wrapper script for identifying sequence motifs using streme
"""

import argparse
import inout
import subprocess
import tempfile
import sys

def run_streme(seq_fname, yvals_fname, threshold, out_direc):
    '''Runs streme to find motifs present in `seq_fname` enriched
    in peak regions identified as 1 in `yvals_fname`.

    Args:
    -----
    seq_fname : str
        Absolute path to file containing fasta records corresponding
        to the values in yvals_fname.
    yvals_fname : str
        Tab-separated values file with two columns. First column is
        sequence name, to correspond with sequence names found in
        the fasta headers in seq_fname. Second column indicates whether
        each sequence indicated in column one was a hit "1" or not "0".
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
    with open(seq_fname, "r") as f:
        fa_file.read_whole_file(f)
    with open(yvals_fname, "r") as f:
        # skip first line
        f.readline()
        for (i,line) in enumerate(f):
            line_num = i+2
            name,yval = line.strip().split("\t")
            header = f">{name}"
            if yval == "1":
                entry = fa_file.pull_entry(header[1:])
                pos_fa_file.add_entry(entry)
            elif yval == "0":
                entry = fa_file.pull_entry(header[1:])
                neg_fa_file.add_entry(entry)
            else:
                raise inout.StremeClassException(yval, line_num)

    with tempfile.NamedTemporaryFile(mode="w") as pos_f:
        tmp_pos = pos_f.name
        pos_fa_file.write(pos_f)
        with tempfile.NamedTemporaryFile(mode="w") as neg_f:
            tmp_neg = neg_f.name
            neg_fa_file.write(neg_f)
            STREME = f"streme --thresh {threshold} "\
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
    parser.add_argument('--threshold', action='store', type=str, default=0.05,
        help=f"p-value threshold for streme to continue searching for motifs. "\
            f"Default: %(default)s. See streme documentation for details")
    parser.add_argument('--out_direc', action='store', type=str, required=True,
        help=f"Absolute path to the directory to be created, or clobbered if it "\
            f"already exists, by streme. Will contain streme output.")
    args = parser.parse_args()

    result = run_streme(
        args.seq_fname,
        args.yvals_fname,
        args.threshold,
        args.out_direc,
    )

    print(result.stdout)
    print(result.stderr, file=sys.stderr)


if __name__ == '__main__':
    main()


