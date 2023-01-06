#!/usr/bin/env bash

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate motifer

BINDIR="/home/x-schroeder/src/DNAshape_motif_finder"
#CONVERTER="${BINDIR}/convert_narrowpeak_to_fire.py"
GENOMEFA="/anvil/projects/x-mcb140220/schroedj/motif_data/ncbi-genomes-2021-12-14/GCF_000001405.26_GRCh38_chr_chroms.fa"
TFNAME=$1
PEAKFILE=$2
INPUTDIR="${TFNAME}_input/"
KFOLD=5
CENTER="height"
WINSIZE=60

# if the directory doesn't exist, make it
if [[ ! -d $INPUTDIR ]]; then
    mkdir ${INPUTDIR}
fi

# make input fasta, shape parameter, and fire files
bash ${BINDIR}/create_input_files.sh ${INPUTDIR} ${TFNAME} ${PEAKFILE} ${GENOMEFA} ${BINDIR} ${WINSIZE} ${CENTER} ${KFOLD}

