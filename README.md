TODO:
1. add input file format specs (will add example\_files directory)
    [ ] place example files in directory
2. add instructions for running examples
3. add docs for every argument for every script
    [x] find\_motifs.py
    [ ] evaluate\_motifs.py
4. consider adding wrapper script for all steps (scheme.py)

# SCHEME

SCHEME is a tool for identifying local structural motifs that inform
protein/DNA interaction.

# Preparing input data

The input files required by SCHEME are:

1. scores file
    + A tab-delimited file with one header line and two columns
    + Column 1: "name" - the name of each sequence (in order) found in 
        the fasta files.
    + Column 2: "score" - the score associated with each sequence in
        the fasta files. For identification of motifs that inform peaks vs
        non-peaks, the score column should contain 0 or 1, where 0 would indicate
        a non-peak sequence, and 1 would indicate a peak seqeunce. The scores
        can also be categorical or continuous. If using continuous data,
        we recommend the user convert their scores to robust z-scores using
        a tool such as [`bgtools`](https://github.com/jwschroeder3/bgtools.git).
2. shape fasta files - shapes we typically use are below.
    + electrostatic potential (EP)
    + helical twist (HelT)
    + minor groove width (MGW)
    + propeller twist (ProT)
    + roll (Roll)
3. Only if running sequence motif finding - sequence fasta file

As noted above, the sequence names in the fasta files and in the score
file must be in the same order.

We provide utilities which should help to prepare, in most use cases,
the score file and the shape fasta files.

## Generating input sequences and category assignments

### Making/using categorical (or binary) inputs

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

TODO: I think wsize must be less than the minimum narrowpeak region width,
but I have to check on that and insert a note on it here.

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

### Using continuous inputs

Continuous inputs will be quantized into categories by the `find_motifs.py` script.

The user must simply create the score file with the continuous scores of interest,
keeping in mind that the file must have two, tab-separated columns and
must have a header with column names "name" and "score".

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

The above code will create five shape files for each fasta file you have,
where "\*" will be replaced with your file prefix.

1. \*.fa.EP - electrostatic potential
2. \*.fa.HelT - helical twist
3. \*.fa.MGW - minor groove width
4. \*.fa.ProT - propeller twist
5. \*.fa.Roll - roll

### Using continuous scores

We recommend that if the user is using categorical data that they first
convert their scores to robust z-scores using a tool such as
[`bgtools`](https://github.com/jwschroeder3/bgtools.git),
then manually create the required input file with paired sequence names
and scores.

Then, within `find_motifs.py`, 

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
        like the following for your `--shape_files` argument:
        `prefix.fa.EP prefix.fa.HelT prefix.fa.MGW prefix.fa.ProT prefix.fa.Roll`,
        of course, replacing `prefix` with the actual file prefix.
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
        --score_file <infile> \
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

NOTE: the below code still needs tested with singularity exec.

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
        --score_file <infile> \
        --seq_motif_positive_cats <comma_sep_cats> \
        --no_shape_motifs \
        > log.log \
        2> log.err
```

## Infer both shape and sequence motifs

NOTE: the below code still needs tested with singularity exec.

See the above explanations for what to substitute for variables
in `<var_name>` notation below.

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
        --score_file <infile> \
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
