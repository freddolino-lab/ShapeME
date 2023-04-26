# SCHEME

SCHEME is a tool for identifying local structural motifs that inform
protein/DNA interaction.

## Preparing input data

### Starting with narrowpeak file defining "positive" regions

If you are starting from a narrowpeak file, read this section carefully
for instructions to create input files for SCHEME.

Enter the directory containing your narrowpeak file.

In the below code example, substitute `<np_fname>` with your
narropeak file name, `<ref_fasta>` with the full path of the
reference genome fasta file, `<out_prefix>` with the prefix to
use for the ouptut fasta files, `<windowsize>` with the width
of the chunks of the genome you would like to search for
motifs within, and `<foldnum>` with the number of cross-validation
folds you would like. For example, use 5 to do 5-fold cross-validation.

```bash
singularity exec -B $(pwd):$(pwd) \
    scheme_0.0.1.sif \
    python /src/python3/convert_narrowpeak_to_fire.py \
        <np_fname> \
        <ref_fasta> \
        <out_prefix> \
        --wsize <windowsize> \
        --kfold <foldnum> \
        --nrand 3 \
        --center_metric "height"
```

For each fold, a pair of training and test data sequences will
be prepared, along with their corresponding input text files
denoting whether each sequence arose from the "positive" set or
the "negative" set.

### Calculating local shapes from sequences

Next, local shapes must be calculated for each sequence in your training
and testing data.

Enter the directory containing your training and testing sequence files.

Below, substitute the name of the fasta file for `<data_fasta>`. Do this
for each train/test file for each fold.

```bash
singularity exec -B $(pwd):$(pwd) \
    scheme_0.0.1.sif \
    python /src/python3/convert_seqs_to_shapes.py <data_fasta>
```

The above code will create five shape files for each fasta file you have.

1. \*.fa.EP - electrostatic potential
2. \*.fa.HelT - helical twist
3. \*.fa.MGW - minor groove width
4. \*.fa.ProT - propeller twist
5. \*.fa.Roll - roll


