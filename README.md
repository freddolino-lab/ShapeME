# SCHEME

SCHEME is a tool for identifying local structural motifs that inform
protein/DNA interaction.

# Preparing input data

## Generating input sequences and category assignments

### Making categorical (or binary) inputs

#### Starting with narrowpeak file defining "positive" regions

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

TODO: add section for making categorical inputs from continuous data

### Making continuous inputs

Note that these continuous inputs will be quantized into categories
by the `find_motifs.py` script.

## Calculating local shapes from sequences

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

# Running SCHEME

SCHEME can be run to detect only shape motifs, only sequence motifs (in this
case SCHEME is basically a wrapper for STREME), or to incorporate shape and
sequence motifs into a single model.

## Infer only shape motifs

Enter the directory containing your sequence files, shape files,
and input score files. Run the code beloe, with the
following substitutions:

+ \<shape\_names\>
    + These are the short names of the shape parameters. We typically set this
    to "EP HelT MGW ProT Roll"
+ \<shape\_files\>
    + The files containing each shape value for each fasta record.
    + NOTE: These files MUST be in the same order as your `--shape_names` argument.
        For example, with the order of shape names above, you would use something
        like the following for your `--shape_files` argument: "pref.fa.EP pref.fa.HelT pref.fa.MGW pref.fa.ProT pref.fa.Roll"
+ \<out\_prefix\>
    + Prefix to be placed at the beginning of output files.
+ \<out\_dir\>
    + Directory into which output files will be written. This directory is created
    by the script if it does not already exist.
+ \<infile\>
    + The file containing input scores, which could be binary, categorical, or
        continuous. If they are continuous, you MUST set the `--continuous` flag
        at the command line to set the number of bins into which to discretize
        the input scores. For instance, `--continuous 10` would create 10 approximately
        evenly populated bins into which to allocate the input data.
+ \<cores\>
    + Sets the number of parallel processes to use for shape motif inference.
    + The value you use will depend on the available resources of your system.
    + We have found that this algorithm scales well to up to 64 cores, but beyond
        that number the returns diminish quickly.
  
```bash
alpha=0.01
temp=0.25
t_adj=0.0002
step=0.25
thresh_bounds="0 10"
shape_bounds="-4 4"
weight_bounds="-4 4"
niter=20000
batch_size=200

kmer=10

singularity exec -B $(pwd):$(pwd) \
    scheme_0.0.1.sif \
    python /src/python3/find_motifs.py \
        --param_names <shape_names> \
        --params <shape_files> \
        -o <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
        --infile <infile> \
        -p <cores> \
        --alpha ${alpha} \
        --max_count ${max_count} \
        --temperature ${temp} \
        --t_adj ${t_adj} \
        --opt_niter ${niter} \
        --stepsize ${step} \
        --threshold_constraints ${thresh_bounds} \
        --shape_constraints ${shape_bounds} \
        --weights_constraints ${weight_bounds} \
        --batch_size ${batch_size} \
        --max-batch-no-new-seed 10 \
        --kmer ${kmer}" \
        > log.log \
        2> log.err
```

## Infer only sequence motifs

NOTE: the below code still needs tested.

In the below code, substitute `<comma_sep_cats>` with a comma-separated
list of the categories to be considered as the "positive" set by STREME
during sequence motif finding.

```bash
singularity exec -B $(pwd):$(pwd) \
    scheme_0.0.1.sif \
    python /src/python3/find_motifs.py \
        --seq_fasta ${seq_file} \
        -o <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
        --infile <infile> \
        --seq_motif_positive_cats <comma_sep_cats> \
        --no_shape_motifs \
        > log.log \
        2> log.err
```

## Infer both shape and sequence motifs

See the above explanations for what to substitute for variables
in `<var_name>` lines below.

```bash
alpha=0.01
temp=0.25
t_adj=0.0002
step=0.25
thresh_bounds="0 10"
shape_bounds="-4 4"
weight_bounds="-4 4"
niter=20000
batch_size=200

kmer=10

singularity exec -B $(pwd):$(pwd) \
    scheme_0.0.1.sif \
    python /src/python3/find_motifs.py \
        --param_names <shape_names> \
        --params <shape_files> \
        -o <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
        --infile <infile> \
        -p <cores> \
        --alpha ${alpha} \
        --max_count ${max_count} \
        --temperature ${temp} \
        --t_adj ${t_adj} \
        --opt_niter ${niter} \
        --stepsize ${step} \
        --threshold_constraints ${thresh_bounds} \
        --shape_constraints ${shape_bounds} \
        --weights_constraints ${weight_bounds} \
        --batch_size ${batch_size} \
        --max-batch-no-new-seed 10 \
        --kmer ${kmer}" \
        --seq_motif_positive_cats <comma_sep_cats> \
        --find_seq_motifs \
        --seq_fasta ${seq_file} \
        --write_all_files \
        --kmer ${kmer}" > $JOBFILE
```
