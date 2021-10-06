#!/usr/bin/env bash
set -e

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate motifer

SRC_DIR="/home/schroedj/src/DNAshape_motif_finder/"
ALPHA=$1
MAX_COUNT=$2
PROCS=$3
OPTIM_VARS="weights threshold"
SHAPE_CONSTRAINTS="-4 4"
WEIGHT_CONSTRAINTS="-4 4"
THRESH_CONTSTRAINTS="0 10"
TEMP=0.1
STEP=0.25
FATOL=0.025

shape_files="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.EP synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.HelT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.MGW synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.ProT synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.fa.Roll"
shape_names="EP HelT MGW ProT Roll"
data_dir="/scratch/sigbio_project_root/sigbio_project7/schroedj/DNAshape_tests/synthetic_data"
out_dir="test"
infile="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA.txt"
out_prefix="synthetic_data_2000_recs_frac_0.2_seqlen_60_TBX5motif_TCTCACACCT_10bp_AGGTGTGAGA"

# calculate and save initial MI
python ${SRC_DIR}find_motifs.py \
    -p $PROCS \
    --exit_after_initial_mi \
    --max_count $MAX_COUNT \
    --alpha $ALPHA \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile

# load MI file and filter by CMI
python ${SRC_DIR}find_motifs.py \
    -p $PROCS \
    --exit_after_cmi_filter \
    --mi_file \
    --optim_vars $OPTIM_VARS \
    --max_count $MAX_COUNT \
    --alpha $ALPHA \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile

# load CMI file and optimize
python ${SRC_DIR}find_motifs.py \
    -p $PROCS \
    --cmi_file \
    --exit_after_optimization \
    --optim_vars $OPTIM_VARS \
    --fatol $FATOL \
    --temperature $TEMP \
    --stepsize $STEP \
    --shape_constraints $SHAPE_CONSTRAINTS \
    --weight_constraints $WEIGHT_CONSTRAINTS \
    --threshold_constraints $THRESH_CONSTRAINTS \
    --max_count $MAX_COUNT \
    --alpha $ALPHA \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile

# load optimized motifs and filter them by CMI
python ${SRC_DIR}find_motifs.py \
    -p $PROCS \
    --optim_file \
    --optim_vars $OPTIM_VARS \
    --fatol $FATOL \
    --temperature $TEMP \
    --stepsize $STEP \
    --max_count $MAX_COUNT \
    --alpha $ALPHA \
    --param_names $shape_names \
    --params $shape_files \
    -o $out_prefix \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --infile $infile
