BINDIR="/home/schroedj/src/DNAshape_motif_finder"
#CONVERTER="${BINDIR}/convert_narrowpeak_to_fire.py"
PEAKFILE="wgEncodeAwgTfbsSydhGm12878Brca1a300IggmusUniPk.narrowPeak"
GENOMEFA="/corexfs/schroedj/databases/Homo_sapiens/Ensembl/GRCh37/Sequence/WholeGenomeFasta/genome.fa"
TFNAME="BRCA1"
INPUTDIR="${TFNAME}_input/"
KFOLD=5
CENTER="height"
WINSIZE=30

# if the directory doesn't exist, make it
if [[ ! -d $INPUTDIR ]]; then
    mkdir ${INPUTDIR}
fi

# make input fasta, shape parameter, and fire files
bash create_input_files.sh ${INPUTDIR} ${TFNAME} ${PEAKFILE} ${GENOMEFA} ${BINDIR} ${WINSIZE} ${CENTER} ${KFOLD}

# sample the files made above
#python3 scripts/sample_fasta.py ${INPUTDIR}${TFNAME}_${WINSIZE}_bp_${CENTER}_train_0.txt ${INPUTDIR}sampled_ ${INPUTDIR}${TFNAME}_${WINSIZE}_bp_${CENTER}_train_0.fa.*

#mkdir ELK1_input
#bash create_input_files.sh ELK1_input/ ELK1 /home/mbwolfe/src/DNAshape_motif_finder/ENCODE_uniformly_processed_ChIP/wgEncodeAwgTfbsSydhGm12878Elk112771IggmusUniPk.narrowPeak /home/mbwolfe/src/tinker_motif_finder/DNAshape_motif_finder
