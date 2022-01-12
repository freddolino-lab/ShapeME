#!/usr/bin/env bash
set -e

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate motifer

SRC_DIR="/home/schroedj/src/DNAshape_motif_finder/"
alpha=$1
max_count=$2
cores=$3

opt_vars="shapes weights threshold"

shape_files="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.EP synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.HelT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.MGW synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.ProT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.Roll"
shape_names="EP HelT MGW ProT Roll"
data_dir="/nfs/turbo/umms-petefred/schroedj/DNAshape_tests/synthetic_data"
out_dir="test"
infile="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
out_prefix="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA"

temp=0.1
step=0.25
fatol=0.025
thresh_bounds="0 10"
shape_bounds="-4 4"
weight_bounds="-4 4"

## calculate initial MI then exit
#python ${SRC_DIR}find_motifs.py \
#    --param_names $shape_names \
#    --params $shape_files \
#    -o $out_prefix \
#    --data_dir $data_dir \
#    --out_dir $out_dir \
#    --infile $infile \
#    -p $cores \
#    --optim_vars $opt_vars \
#    --alpha $alpha \
#    --max_count $max_count \
#    --temperature $temp \
#    --stepsize $step \
#    --fatol $fatol \
#    --threshold_constraints $thresh_bounds \
#    --shape_constraints $shape_bounds \
#    --weights_constraints $weight_bounds \
#    --exit_after_initial_mi \
#    
## do CMI filtering
#python ${SRC_DIR}find_motifs.py \
#    --param_names $shape_names \
#    --params $shape_files \
#    -o $out_prefix \
#    --data_dir $data_dir \
#    --out_dir $out_dir \
#    --infile $infile \
#    -p $cores \
#    --optim_vars $opt_vars \
#    --alpha $alpha \
#    --max_count $max_count \
#    --temperature $temp \
#    --stepsize $step \
#    --fatol $fatol \
#    --threshold_constraints $thresh_bounds \
#    --shape_constraints $shape_bounds \
#    --weights_constraints $weight_bounds \
#    --mi_file \
#    --exit_after_cmi_filter
#
## optimize filtered seeds
#python ${SRC_DIR}find_motifs.py \
#    --param_names $shape_names \
#    --params $shape_files \
#    -o $out_prefix \
#    --data_dir $data_dir \
#    --out_dir $out_dir \
#    --infile $infile \
#    -p $cores \
#    --optim_vars $opt_vars \
#    --alpha $alpha \
#    --max_count $max_count \
#    --temperature $temp \
#    --stepsize $step \
#    --fatol $fatol \
#    --threshold_constraints $thresh_bounds \
#    --shape_constraints $shape_bounds \
#    --weights_constraints $weight_bounds \
#    --cmi_file \
#    --exit_after_optimization

# plot some information about the optimizations
python ${SRC_DIR}plot_optims.py \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile

# CMI-filter optimized motifs
python ${SRC_DIR}find_motifs.py \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile \
    -p $cores \
    --optim_vars $opt_vars \
    --alpha $alpha \
    --max_count $max_count \
    --temperature $temp \
    --stepsize $step \
    --fatol $fatol \
    --threshold_constraints $thresh_bounds \
    --shape_constraints $shape_bounds \
    --weights_constraints $weight_bounds \
    --exit_after_cmi_motifs \
    --optim_file

# plot some information about the cmi-filtered motifs
python ${SRC_DIR}plot_optims.py \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --motifs_file "${out_prefix}_post_opt_cmi_filtered_motifs_optim_shapes_weights_threshold_adapt_False_fatol_${fatol}_temp_${temp}_stepsize_${step}_alpha_${alpha}_max_count_${max_count}.pkl" \
    --infile $infile

