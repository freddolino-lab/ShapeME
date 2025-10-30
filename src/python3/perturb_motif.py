import inout as io
import warnings
import ipdb
import logging
import argparse
import sys
import os
import numpy as np

from pathlib import Path

# Define a function to be called when a RuntimeWarning is encountered
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    # Print the warning message
    print(warnings.formatwarning(message, category, filename, lineno, line))
    # Start the debugger
    ipdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--motif_file", type=str, required=True,
        help=f"Meme or dsm file with a single shape motif definition.")
    parser.add_argument("-t", "--type", type=str, default="both",
        choices = ["shapes", "weights", "both"],
        help=f"The type of motif parameter to perturb. Can be one of 'shapes', 'weights', or 'both'.")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
        help="The value of 'alpha' used during shape motif inference. This is usually 0.1.")
    parser.add_argument("-n", "--n_perturb", type=int, default=1000,
        help="The number of permutations to perform")
    parser.add_argument("-o", "--output_file", type=str, required=False,
        help="The name of the output file into which perturbed motifs will be written.")
    parser.add_argument("--separate", action="store_true", default=False,
        help="Add at command line to save each perturbed version of the motif to a different file. If set, --output_file is ignored.")

    args = parser.parse_args()
    return args


def main():

    # Set the warning filter to call the above function on RuntimeWarning
    #warnings.simplefilter('always', RuntimeWarning)
    #warnings.showwarning = warn_with_traceback

    args = parse_args()

    motif_file = args.motif_file
    param_type = args.type
    alpha = args.alpha
    out_name = args.output_file
    n_perturb = args.n_perturb
    separate_files = args.separate
 
    motif = io.Motifs()
    motif.read_file( motif_file )

    seq_motif,shape_motif = motif.split_seq_and_shape_motifs()

    if len(seq_motif) > 0:
        sys.exit("Motif perturbation not implemented for sequence motifs at this time. Use motif files with only one shape motif for comparison.")
    if len(shape_motif) > 1:
        sys.exit("Motif perturbation implemented for only a single shape motif. Use a motif file with only one shape motif. Exiting now.")

    shape_motif = shape_motif[0]

    perturbed_list = [
        perturbed_motif for perturbed_motif in 
        shape_motif.generate_perturbations(
            which_array = param_type,
            alpha = alpha,
            n_perturb = n_perturb,
        )
    ]

    perturbed_motifs = io.Motifs()
    if separate_files:
        for i,perturbed_motif in enumerate(perturbed_list):
            perturbed_motifs.motifs = [perturbed_motif]
            perturbed_motifs[0].name = f"perturb_{i}"
            fname = f"perturbed_motif_{i}.dsm"
            perturbed_motifs.write_file(fname)

    else:
        perturbed_motifs.motifs = perturbed_list
        perturbed_motifs.write_file(out_name)

if __name__ == "__main__":
    main()
