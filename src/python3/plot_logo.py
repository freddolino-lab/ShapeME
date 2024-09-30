import sys
import os
import inout
import argparse
import shapemotifvis as smv


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--motif_file', type=str,
            help="Directory to which to write outputs.")
    parser.add_argument('--out_direc', type=str,
            help="Output directory to write plots.")

    args = parser.parse_args()

    final_motif_plot_suffix = os.path.join(
        args.out_direc,
        "{}_final_motif.pdf"
    )

    motifs = inout.Motifs()
    motifs.read_file( args.motif_file )
    seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()

    shape_lut = {v:k for k,v in shape_motifs.shape_row_lut.items()}
    #print(shape_lut)
    plot_fnames = smv.plot_logos(
        motifs,
        final_motif_plot_suffix,
        shape_lut,
    )

    print(f"plot_fnames: {plot_fnames}")

if __name__ == '__main__':
    main()


