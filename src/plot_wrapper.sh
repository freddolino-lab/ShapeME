#!/usr/bin/env bash
set -e

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate motifer

SRC_DIR="/home/schroedj/src/DNAshape_motif_finder/"

shape_files="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.EP synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.HelT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.MGW synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.ProT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.Roll"
shape_names="EP HelT MGW ProT Roll"
data_dir="/nfs/turbo/umms-petefred/schroedj/DNAshape_tests/synthetic_data"
out_dir="test"
infile="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
out_prefix="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA"

# calculate and save initial MI
python ${SRC_DIR}plot_optims.py \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile

