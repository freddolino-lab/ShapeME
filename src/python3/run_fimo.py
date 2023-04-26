#!/usr/bin/env python3

"""
Wrapper for running fimo.
"""

import argparse
import subprocess
import sys

def run_fimo(seq_fa, meme_file, out_dir):
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
        f"using motifs in {meme_file}"
    )

    FIMO = f"fimo --max-strand --motif-pseudo 0.0 "\
        f"--oc {out_dir} "\
        f"{meme_file} {seq_fa}"
    print("Running fimo command:")
    print(FIMO)
    result = subprocess.run(
        FIMO,
        shell=True,
        check=True,
        capture_output=True,
    )
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_fname', action='store', type=str, required=True,
        help=f"Absolute path to file containing sequences in which to search "\
            f"for motifs defined in meme_file")
    parser.add_argument('--meme_file', action='store', type=str, required=True,
        help=f"Absolute path to meme file defining sequence motifs to find")
    parser.add_argument('--out_direc', action='store', type=str, required=True,
        help=f"Absolute path to the directory to be created, or clobbered if it "\
            f"already exists, by fimo. Will contain fimo output.")
    args = parser.parse_args()

    result = run_fimo(
        args.seq_fname,
        args.meme_file,
        args.out_direc,
    )
    print(result.stdout)
    print(result.stderr, file=sys.stderr)

if __name__ == '__main__':
    main()

