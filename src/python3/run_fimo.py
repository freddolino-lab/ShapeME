#!/usr/bin/env python3

"""
Wrapper for running fimo.
"""

import argparse
import subprocess
import sys

def run_fimo(seq_fa, meme_file, out_dir, thresh=None):
    '''Runs fimo.

    Args:
    -----
    seq_fa : str
        Absolute path to file containing sequences in which to search
        for motifs defined in meme_file
    meme_file : str
        Absolute path to meme file defining sequence motifs to find
    out_dir : str
        Absolute path to directory for fimo to either create or
        clobber. Will contain fimo output.
    '''

    print()
    print(
        f"Running fimo on all sequences in {seq_fa} "\
        f"using motifs in {meme_file}\n"
    )

    FIMO = f"fimo --max-strand --motif-pseudo 0.0 "\
        f"--oc {out_dir} "
    if thresh is not None:
        FIMO += f"--thresh {thresh} "
    FIMO += f"{meme_file} {seq_fa}"
    print("Running fimo command:")
    print(FIMO)
    result = subprocess.run(
        FIMO,
        shell=True,
        check=True,
    )
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', action='store', type=float, required=False,
        help=f"Fimo threshold.")
    parser.add_argument('--seq_fname', action='store', type=str, required=True,
        help=f"Absolute path to file containing sequences in which to search "\
            f"for motifs defined in meme_file")
    parser.add_argument('--meme_file', action='store', type=str, required=True,
        help=f"Absolute path to meme file defining sequence motifs to find")
    parser.add_argument('--out_direc', action='store', type=str, required=True,
        help=f"Absolute path to the directory to be created, or clobbered if it "\
            f"already exists, by fimo. Will contain fimo output.")
    #parser.add_argument('--log_file', action='store', type=str, default=None,
    #    help=f"Name of log file to write fimo stdout to.")
    #parser.add_argument('--err_file', action='store', type=str, default=None,
    #    help=f"Name of file to write fimo stderr to.")
    args = parser.parse_args()

    result = run_fimo(
        args.seq_fname,
        args.meme_file,
        args.out_direc,
        args.thresh,
    )
    #print(result.stdout.decode())
    #print(result.stderr.decode())
    #with open(args.log_file, "w") as outf:
    #    outf.write(result.stdout.decode())
    #with open(args.err_file, "w") as errf:
    #    errf.write(result.stderr.decode())

if __name__ == '__main__':
    main()

