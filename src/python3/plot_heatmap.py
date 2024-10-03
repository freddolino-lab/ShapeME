import sys
import os
import inout
import argparse
import shapemotifvis as smv
import pickle


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--motif_file', type=str,
            help="Directory to which to write outputs.")
    parser.add_argument('--coefs_fname', type=str)
    parser.add_argument('--out_direc', type=str,
            help="Output directory to write plots.")
    parser.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")
    parser.add_argument('--shape_names', nargs="+", type=str, required=True,
        help='shape names (MUST BE IN SAME ORDER AS CORRESPONDING SHAPE FILES)')
    parser.add_argument('--score_file', action='store', type=str, required=True,
        help='input text file with names and scores for training data')
    parser.add_argument('--shape_files', nargs="+", type=str, required=True,
        help='input files with shape scores')
    parser.add_argument('--max_count', type=int, default=1)
    parser.add_argument('--fimo_file', type=str)
    parser.add_argument('--rust_results', type=str)
    

    args = parser.parse_args()

    in_direc = args.data_dir
    shape_names = args.shape_names
    shape_files = args.shape_files
    in_fname = args.score_file

    records = inout.construct_records(
        in_direc,
        shape_names,
        shape_files,
        in_fname,
    )
    records.set_category_lut()

    out_heatmap_fname = os.path.join(
        args.out_direc,
        "heatmap.pdf"
    )

    rust_motifs = inout.read_shape_motifs(
        args.rust_results, 
        records.shape_name_lut,
    )

    motifs = inout.Motifs()
    motifs.read_file( args.motif_file )
    for motif in rust_motifs:
        for final_motif in motifs:
            if motif.identifier == final_motif.identifier:
                final_motif.hits = motif.hits

    with open(args.coefs_fname, "rb") as coef_f:
        coef_dict = pickle.load(coef_f)
    motifs.var_lut = coef_dict["var_lut"]
    motifs.set_X(
        max_count=args.max_count,
        fimo_fname=args.fimo_file,
        rec_db=records,
        qval_thresh=0.05,
        nosort=False,
        var_lut=motifs.var_lut,
        test=False,
    )

    motifs.get_enrichments(records)

    smv.plot_motif_enrichment_seaborn(
        motifs,
        out_heatmap_fname,
        records = records,
    )

if __name__ == '__main__':
    main()


