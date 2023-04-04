import sys
import os
import inout
import argparse
import shapemotifvis as smv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--motif_file', type=str,
            help="Directory to which to write outputs.")
    parser.add_argument('--plot_file', type=str,
            help="Output file containing the plot.")
    parser.add_argument('--top_n', type=int, default=5,
            help="Number of motifs to plot.")

    args = parser.parse_args()

    motifs = inout.Motifs()
    motifs.read_file( args.motif_file )

    seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()

    #shape_lut = {v:k for k,v in shape_motifs.shape_row_lut.items()}
    #print(shape_lut)

    smv.plot_logo(
        shape_motifs,
        args.plot_file,
        shape_motifs.shape_row_lut,
        top_n = args.top_n
    )

if __name__ == '__main__':
    main()


