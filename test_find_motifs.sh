#!/usr/bin/env bash

shape_files="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.EP synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.HelT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.MGW synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.ProT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.Roll"
shape_names="EP HelT MGW ProT Roll"
data_dir="/home/jeremy/motif_finder_data/DNAshape_tests"
infile="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
out_prefix="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA"

# calculate and save initial MI
#python find_motifs.py --exit_after_initial_mi --max_count 1 --alpha 0.0 --param_names $shape_names --params $shape_files --no_optimize -o $out_prefix --data_dir $data_dir $infile

# load MI file and filter by CMI
#python find_motifs.py --exit_after_cmi_filter --mi_file --optim_vars shapes weights threshold --max_count 1 --alpha 0.0 --param_names $shape_names --params $shape_files --no_optimize -o $out_prefix --data_dir $data_dir $infile

# load CMI file and optimize
python find_motifs.py --cmi_file --optim_vars weights threshold --fatol 0.025 --max_count 1 --alpha 0.0 --param_names $shape_names --params $shape_files -o $out_prefix --data_dir $data_dir $infile
