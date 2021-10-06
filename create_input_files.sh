DATADIR=$1
GENE=$2
PEAKFILE=$3
GENOME=$4
SCRIPTDIR=$5
WSIZE=$6
CENTER=$7
PREFIX="${DATADIR}${GENE}_${WSIZE}_bp_${CENTER}"
FOLDS=$8

python ${SCRIPTDIR}/convert_narrowpeak_to_fire.py \
    ${PEAKFILE} \
    ${GENOME} \
    ${PREFIX} --rmchr --kfold ${FOLDS} --center_metric ${CENTER}

# set up k-fold iterator
((end=$FOLDS - 1))
iter=$(seq 0 $end)

# iterate to make fold datasets
for NUM in $iter;
do
    Rscript ${SCRIPTDIR}/calc_shape.R ${PREFIX}_train_${NUM}.fa
    Rscript ${SCRIPTDIR}/calc_shape.R ${PREFIX}_test_${NUM}.fa
done