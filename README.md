TODO:
1. add input file format specs (will add example\_files directory)
    [ ] place example files in directory
2. add instructions for running examples
3. add docs for every argument for every script\\
    [ ] evaluate\_motifs.py\\
    [ ] ShapeMe.py\\

# ShapeMe

ShapeMe is a tool for identifying local structural motifs that inform
protein/DNA interaction.

# Preparing input data

The input files required by ShapeMe are:

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
    + For example scores files, see the txt files in the `examples/binary_example`,
        `examples/categorical_example`, and `examples/continuous_example`, directories.
3. sequence fasta file
    + For example sequence fasta files, see the `*.fa` files in the `examples/binary_example`,
        `examples/categorical_example`, and `examples/continuous_example`, directories.

The sequence names in the fasta files and in the score
file must be in the same order.

We provide utilities which should help to prepare, in most use cases,
the score file and the shape fasta files.

## Generating input sequences and category assignments

### Making/using categorical (or binary) inputs

#### Starting with narrowpeak file defining "positive" regions

If you are starting from a narrowpeak file, read this section carefully
for instructions to create input files for ShapeME.

Enter the directory containing your narrowpeak file.

In the below code example, substitute `<np_fname>` with your
narropeak file name, `<ref_fasta>` with the full path of the
reference genome fasta file, `<out_prefix>` with the prefix to
use for the ouptut fasta files, `<windowsize>` with the width
of the chunks of the genome you would like to search for
motifs within.

TODO: I think wsize must be less than the minimum narrowpeak region width,
but I have to check on that and insert a note on it here.

```bash
singularity exec -B $(pwd):$(pwd) \
    shapeme_<version>.sif \
    python /src/python3/convert_narrowpeak_to_fire.py \
        <np_fname> \
        <ref_fasta> \
        <out_prefix> \
        --wsize <windowsize> \
        --nrand 3 \
        --center_metric "height"
```

The above command will create a fasta file of sequences and
a scores file denoting whether each sequence arose from the "positive" set or
the "negative" set. Using `--nrand 3` will create a fasta file and corresponding
score file with three times as many sequences in the negative set (score is 0) as those
in the positive set (score is 1).

### Using continuous inputs

Continuous inputs will be quantized into categories by the `ShapeME.py` script.

The user must simply create the score file with the continuous scores of interest,
keeping in mind that the file must have two, tab-separated columns and
must have a header with column names "name" and "score".

## Calculating local shapes from sequences

Local shapes will be calculated for each sequence in your fasta file
when the `ShapeME.py` script is run.

`ShapeME.py` will split your input data into the desired number of
folds for k-fold cross validation (the number of folds being defined at
the command line). Five shape files will be created for
each set of training and testing data generated for each fold.
The files will have the following names,
where "\*" will be replaced with your file prefix.

1. \*.fa.EP - electrostatic potential
2. \*.fa.HelT - helical twist
3. \*.fa.MGW - minor groove width
4. \*.fa.ProT - propeller twist
5. \*.fa.Roll - roll

### Using continuous scores

We recommend that if the user is using continuous data that they first
convert their scores to robust z-scores using a tool such as
[`bgtools`](https://github.com/jwschroeder3/bgtools.git),
then manually create the required input file with paired sequence names
and scores.

Then, when running `ShapeME.py`, set `--continuous <n>` at the command line,
where `<n>` must be replaced with the number of binds to quantize the
continuous scores into.

# Running ShapeME

ShapeME can be run to detect only shape motifs, only sequence motifs (in this
case ShapeME is basically a wrapper for STREME), or to incorporate shape and
sequence motifs into a single model.

We distribute ShapeME as a singularity container, which can be run on any
computer with a Linux environment that has singularity installed.

The ShapeME container can be downloaded from our
[google drive](https://drive.google.com/drive/folders/1e7N4iYO7BHuuZG4q-H7xBk1c6bE9GmOt?usp=share_link)
location.

In all instructions below, you should substitute the characters `<version>` with
the actual version number of the continer you're using in every instance of
`shapeme_<version>.sif`.

## Infer only shape motifs

### Inference on provided example data

#### Binary input values

From within the `examples/binary_example` directory, run the following,
updating the value of `nprocs` to something that is suitable to the system
on which you are running ShapeME:

```bash
nprocs=8

singularity exec -B $(pwd):$(pwd) \
    shapeme_<version>.sif \
    python /src/python3/find_motifs.py \
        --score_file test_data_binary_plus_train_0.txt \
        --shape_names EP HelT MGW ProT Roll \
        --shape_files test_data_binary_plus_train_0.fa.EP test_data_binary_plus_train_0.fa.HelT test_data_binary_plus_train_0.fa.MGW test_data_binary_plus_train_0.fa.ProT test_data_binary_plus_train_0.fa.Roll \
        --out_prefix binary_example \
        --data_dir $(pwd) \
        --out_dir shapeme_shape_output \
        --kmer 10 \
        --alpha 0.01 \
        --max_count 1 \
        --temperature 0.25 \
        --t_adj 0.0002 \
        --opt_niter 20000 \
        --stepsize 0.25 \
        --threshold_constraints "0 10" \
        --shape_constraints "-4 4" \
        --weights_constraints "-4 4" \
        --batch_size 200 \
        --max-batch-no-new-seed 10 \
        --nprocs ${nprocs} \
        > log.log \
        2> log.err
```

### Using your own data

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
max_count=1

kmer=10

singularity exec -B $(pwd):$(pwd) \
    shapeme_<version>.sif \
    python /src/python3/find_motifs.py \
        --score_file <infile> \
        --shape_names <shape_names> \
        --shape_files <shape_files> \
        --out_prefix <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
        --kmer ${kmer} \
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
        --nprocs <cores> \
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
    shapeme_<version>.sif \
    python /src/python3/find_motifs.py \
        --score_file <infile> \
        --seq_fasta ${seq_file} \
        --out_prefix <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
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
max_count=1

kmer=10

singularity exec -B $(pwd):$(pwd) \
    shapeme_<version>.sif \
    python /src/python3/find_motifs.py \
        --score_file <infile> \
        --shape_names <shape_names> \
        --shape_files <shape_files> \
        --out_prefix <out_prefix> \
        --data_dir $(pwd) \
        --out_dir <out_dir> \
        --kmer ${kmer} \
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
        --seq_motif_positive_cats <comma_sep_cats> \
        --find_seq_motifs \
        --seq_fasta ${seq_file} \
        --write_all_files \
        --nprocs <cores>
```
