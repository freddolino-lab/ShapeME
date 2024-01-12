# ShapeME - Shape-based Motif Elicitation

ShapeME is a tool for finding informative motifs in DNA structure. 

In contrast to conventional motif finders which search for motifs in sequence
space, ShapeME instead identifies motifs in local DNA structure space using
sequence-based predictions of DNA structure parameters including minor groove
width, electrostatic potential, helical twist, propeller twist, and roll. DNA
structural parameters are predicted from input sequences using `DNAshapeR`
([Chiu et al. 2016](https://doi.org/10.1093/bioinformatics/btv735)).

ShapeME excels at discovering DNA structural motifs that explain
binary data such as bound and unbound regions on a chromosome for a given
factor, however, ShapeME is also more flexible than a typical motif finder and
can be used to identify motifs predictive of data with more than two categories
including RNA-seq expression levels, SELEX-seq enrichment scores, experiments
with more than one condition being tested etc.  Any score that can be binned
into categories on a per-sequence basis can be used for motif inference using
ShapeME. This powerful mutual information-based approach is inspired by the `FIRE`
algorithm developed for sequence motif inference ([Elemento et al.
2007](https://doi.org/10.1016/j.molcel.2007.09.027)).

For most ShapeME users, we recommend running ShapeME using our [web interface]
(https://seq2fun.dcmb.med.umich.edu/shapeme).

For users who would like to run ShapeME on their own hardware, we recommend they
download our Apptainer container using `apptainer pull` to run ShapeME on a
linux operating system. This will require the user to install
[`Apptainer`](https://apptainer.org/docs/user/main/quick_start.html)
on their system and add the Sylabs remote enpoint to their list of repositories
to search for containers (brief instructions are below, but a user should
visit read the official `apptainer` documentation).

The main script most users will run is `ShapeME.py`. See below for documentation
on the required inputs to `ShapeME.py`.

# Preparing input data

The input files required by ShapeME are:

1. scores file
    + A tab-delimited file with one header line and two columns
    + Column 1: "name" - the name of each sequence found in 
        the fasta file, in the same order as sequences occur in fasta file.
    + Column 2: "score" - the score associated with each sequence in
        the fasta files. For identification of motifs that inform peaks vs
        non-peaks, the score column should contain 0 or 1, where 0 would indicate
        a non-peak sequence, and 1 would indicate a peak seqeunce. The scores
        can also be categorical or continuous. If using continuous data,
        we recommend the user convert their scores to robust z-scores using
        a tool such as [`bgtools`](https://github.com/jwschroeder3/bgtools.git).
    + For example scores files, see the txt files in the `examples/binary_example`,
        `examples/categorical_example`, and `examples/continuous_example`, directories.
2. sequence fasta file
    + For example sequence fasta files, see the `*.fa` files in the `examples/binary_example`,
        `examples/categorical_example`, and `examples/continuous_example`, directories.

The sequence names in the fasta files and in the score
file *must* be in the same order.

<!--
We provide utilities which should help to prepare, in most use cases,
the score file and the shape fasta file.

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
apptainer exec -B $(pwd):$(pwd) \
    shapeme.sif \
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
-->

## Calculating local shapes from sequences

Local shape files will be calculated for each sequence in your fasta file
when the `ShapeME.py` script is run.

`ShapeME.py` will split your input data into the desired number of
folds for k-fold cross validation (the number of folds being defined at
the command line). Five shape files will be created for
each set of training and testing data generated for each fold.
The files will have the following names,
where "\*" will be replaced with a random sequence of characters
generated by the tempfile python module, followed by "\_main" or "\_fold\_k",
where k is the zero-indexed fold number for crossfold validation.

1. \*.fa.EP - minor groove electrostatic potential
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
where `<n>` must be replaced with the number of equally populated bins to
quantize the continuous scores into.

# Running ShapeME

ShapeME can be run to detect only shape motifs, only sequence motifs (in this
case ShapeME is basically a wrapper for
[`STREME`](https://meme-suite.org/meme/doc/streme.html) with extra motif pruning
steps to avoid reporting motifs with overlapping information), or to incorporate
shape and sequence motifs into a single model.

We distribute ShapeME as an Apptainer container, which can be run on any
computer with a Linux environment that has Apptainer installed.

The ShapeME container can be downloaded from our
[google drive](https://drive.google.com/drive/folders/1e7N4iYO7BHuuZG4q-H7xBk1c6bE9GmOt?usp=sharing)
location.

In all instructions below, you should substitute `/path/to`
with the actual path to the location with the ShapeME Apptainer container.

For more information on how to set up `apptainer` for your system please follow
the `apptainer` documentation [here](https://apptainer.org/docs/user/main/quick_start.html).

Once you have set up `apptainer` and added the SylabsCloud repository
to your remote endpoints by running `apptainer remote add SylabsCloud cloud.sycloud.io`,
you will need to direct `apptainer` to use the SylabsCloud repository by running
`apptainer remote use SylabsCloud`. You should now be able to pull the latest version
of our container by running
`apptainer pull shapeme.sif library://schroedj/appliances/shapeme:latest`.
Verify the container by running `apptainer verify shapeme.sif`.

## Inference on provided example data

For the following examples, we recommend the user download the example
data provided with this repository.

### Binary input values

#### Only shape motifs

From within the `examples/binary_example` directory, run the following, updating
the number of processors (`nprocs`) to something that is suitable to the system
on which you are running ShapeME:

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Only sequence motifs

From within the `examples/binary_example` directory, run the code below,
updating the value of the number of processors (`nprocs`) to something that is
suitable to the system on which you are running ShapeME.

In the below code, `--seq_motif_positive_cats 1` is set. The `--seq_motif_positive_cats`
argument sets which categories in the `--score_file` will be assigned to the "positive set"
for `STREME`. We've set it to 1 here because this example is simple binary data,
with 0 as the negative set and 1 as the positive set. For categorical
data, we would set the positive set with a comma-separated
list of the categories to be considered as the "positive" set by `STREME`
during sequence motif finding. For instance, if you are using
scores with 10 bins (values of 0 through 9) and if want to identify
motifs in bins 8 and 9, set `--seq_motif_positive_cats 8,9` at the
command line.

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --no_shape_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 1 \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Infer both shape and sequence motifs

From within the `examples/binary_example` directory, run the following,
noting that the command below is identical to that in
the above section for detecting only shape motifs, except
that this command does not apply the `--no_shape_motifs`
flag.

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 1 \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

### Categorical input values

#### Only shape motifs

From within the `examples/categorical_example` directory, run the following,
updating the value of the number of processors (`nprocs`) to something that is
suitable to the system on which you are running ShapeME:

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Only sequence motifs

From within the `examples/categorical_example` directory, run the following,
updating the value of the number of processors (`nprocs`) to something that is
suitable to the system on which you are running ShapeME:

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --no_shape_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 4 \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Infer both shape and sequence motifs

From within the `examples/categorical_example` directory, run the following,
noting that the command below is identical to that in
the above section for detecting only shape motifs, except
that this command does not apply the `--no_shape_motifs`
flag.

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 4 \
        --score_file seqs.txt \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

### Continuous input values

#### Only shape motifs

From within the `examples/continuous_example` directory, run the following,
updating the value of the number of processors (`nprocs`) to something that is
suitable to the system on which you are running ShapeME:

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --score_file seqs.txt \
        --continuous 10 \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Only sequence motifs

From within the `examples/continuous_example` directory, run the following,
updating the value of the number of processors (`nprocs`) to something that is
suitable to the system on which you are running ShapeME:

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --no_shape_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 8,9 \
        --score_file seqs.txt \
        --continuous 10 \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

#### Infer both shape and sequence motifs

From within the `examples/continuous_example` directory, run the following,
noting that the command below is identical to that in
the above section for detecting only shape motifs, except
that this command does not apply the `--no_shape_motifs`
flag.

```bash
nprocs=8
data_dir=$(pwd)

apptainer exec -B ${data_dir} \
    /path/to/shapeme.sif \
    python /src/python3/ShapeME.py infer \
        --find_seq_motifs \
        --data_dir ${data_dir} \
        --seq_fasta seqs.fa \
        --seq_motif_positive_cats 8,9 \
        --score_file seqs.txt \
        --continuous 10 \
        --crossval_folds 5 \
        --nprocs ${nprocs} \
        > ${data_dir}/shapeme.log \
        2> ${data_dir}/shapeme.err
```

## Using your own data

Enter the directory containing your sequence files, shape files,
and input score files. The following required arguments must be present, but
read below further for a comprehensive list of arguments that can
be accepted by ShapeME:

+ \<data_dir\>
    + The location containing your input fasta file and scores file.
+ \<seq\_fasta\>
    + The name of the file containing the fasta records. *Use only the basename.
    The file must be in `--data_dir`.*
+ \<score\_file\>
    + The file containing input scores, which could be binary scores, categorical scores, or
        continuous values. If they are continuous, you MUST set the `--continuous` flag
        at the command line to set the number of bins into which to discretize
        the input scores. For instance, `--continuous 10` would create 10 approximately
        evenly populated bins into which to allocate the input data. *Use only the basename.
        The file must be in `--data_dir`.*
+ \<crossval\_folds\>
    + We usually set this to 5, indicating that in addition
    to a round of motif inference on the entire dataset, 5-fold crossvalidation
    will be performed to assess the reproducibility of motif inference on your
    data.

In addition to the required arguments above, one additional argument to be aware of is `--nprocs`:

+ \<nprocs\>
    + Sets the number of parallel processes to use for shape motif inference.
    + The value you use will depend on the available resources of your system.
    + We have found that this algorithm scales well to up to 32 cores, but beyond
        that, returns diminish quickly.

## ShapeME.py arguments

In addition to the arguments demonstrated in the examples, there are many other
arguments that can be set to adjust ShapeME behavior. Below is the output of
running `python ShapeME.py infer --help`, which give the exhaustive list of
arguments that can be passed to `ShapeME.py infer`, along with a brief
description of each argument.

    usage: ShapeME.py infer [-h] --data_dir DATA_DIR [--skip_inference]
                            [--skip_evaluation] [--out_prefix OUT_PREFIX]
                            [--force] --crossval_folds CROSSVAL_FOLDS --score_file
                            SCORE_FILE [--kmer KMER] [--max_count MAX_COUNT]
                            [--continuous CONTINUOUS]
                            [--threshold_sd THRESHOLD_SD]
                            [--init_threshold_seed_num INIT_THRESHOLD_SEED_NUM]
                            [--init_threshold_recs_per_seed INIT_THRESHOLD_RECS_PER_SEED]
                            [--init_threshold_windows_per_record INIT_THRESHOLD_WINDOWS_PER_RECORD]
                            [--max_batch_no_new_seed MAX_BATCH_NO_NEW_SEED]
                            [--nprocs NPROCS]
                            [--threshold_constraints THRESHOLD_CONSTRAINTS THRESHOLD_CONSTRAINTS]
                            [--shape_constraints SHAPE_CONSTRAINTS SHAPE_CONSTRAINTS]
                            [--weights_constraints WEIGHTS_CONSTRAINTS WEIGHTS_CONSTRAINTS]
                            [--temperature TEMPERATURE] [--t_adj T_ADJ]
                            [--stepsize STEPSIZE] [--opt_niter OPT_NITER]
                            [--alpha ALPHA] [--batch_size BATCH_SIZE]
                            [--find_seq_motifs] [--no_shape_motifs]
                            [--seq_fasta SEQ_FASTA]
                            [--seq_motif_positive_cats SEQ_MOTIF_POSITIVE_CATS]
                            [--streme_thresh STREME_THRESH]
                            [--seq_meme_file SEQ_MEME_FILE] [--write_all_files]
                            [--exhaustive] [--max_n MAX_N] [--log_level LOG_LEVEL]

    options:
      -h, --help            show this help message and exit
      --data_dir DATA_DIR   Directory from which input files will be read.
      --skip_inference      Include this flag at the command line to skip motif
                            inference. This is useful if you've already run
                            inference on all folds.
      --skip_evaluation     Include this flag at the command line to skip
                            evaluation of motifs.
      --out_prefix OUT_PREFIX
                            Optional. Sets the prefix to place on output
                            directories. If None (the default), infers the
                            appropriate prefix based on whether you set either or
                            both of --find_seq_motifs or --no_shape_motifs
      --force               Forces each fold to run, clobbering any extant output
                            directories.
      --crossval_folds CROSSVAL_FOLDS
                            Number of folds into which to split data for k-fold
                            cross-validation
      --score_file SCORE_FILE
                            input text file with names and scores for training
                            data
      --kmer KMER           kmer size to search for shape motifs. Default=15
      --max_count MAX_COUNT
                            Maximum number of times a motif can match each of the
                            forward and reverse strands in a reference. Default: 1
      --continuous CONTINUOUS
                            number of bins to discretize continuous input data
                            with
      --threshold_sd THRESHOLD_SD
                            std deviations below mean for seed finding. Only
                            matters for greedy search. Default=2.000000
      --init_threshold_seed_num INIT_THRESHOLD_SEED_NUM
                            Number of randomly selected seeds to compare to
                            records in the database during initial threshold
                            setting. Default=500
      --init_threshold_recs_per_seed INIT_THRESHOLD_RECS_PER_SEED
                            Number of randomly selected records to compare to each
                            seed during initial threshold setting. Default=20
      --init_threshold_windows_per_record INIT_THRESHOLD_WINDOWS_PER_RECORD
                            Number of randomly selected windows within a given
                            record to compare to each seed during initial
                            threshold setting. Default=2
      --max_batch_no_new_seed MAX_BATCH_NO_NEW_SEED
                            Sets the number of batches of seed evaluation with no
                            new motifs added to the set of motifs to be optimized
                            prior to truncating the initial search for motifs.
      --nprocs NPROCS       number of processors. Default: 1
      --threshold_constraints THRESHOLD_CONSTRAINTS THRESHOLD_CONSTRAINTS
                            Sets the upper and lower limits on the match threshold
                            during optimization. Defaults to 0 for the lower limit
                            and 10 for the upper limit.
      --shape_constraints SHAPE_CONSTRAINTS SHAPE_CONSTRAINTS
                            Sets the upper and lower limits on the shapes'
                            z-scores during optimization. Defaults to -4 for the
                            lower limit and 4 for the upper limit.
      --weights_constraints WEIGHTS_CONSTRAINTS WEIGHTS_CONSTRAINTS
                            Sets the upper and lower limits on the pre-
                            transformed, pre-normalized weights during
                            optimization. Defaults to -4 for the lower limit and 4
                            for the upper limit.
      --temperature TEMPERATURE
                            Sets the temperature argument for simulated annealing.
                            Default: 0.400000
      --t_adj T_ADJ         Fraction by which temperature decreases each iteration
                            of simulated annealing. Default: 0.001000
      --stepsize STEPSIZE   Sets the stepsize argument simulated annealing. This
                            defines how far a given value can be modified for
                            iteration i from its value at iteration i-1. A higher
                            value will allow farther hops. Default: 0.250000
      --opt_niter OPT_NITER
                            Sets the number of simulated annealing iterations to
                            undergo during optimization. Default: 10000.
      --alpha ALPHA         Lower limit on transformed weight values prior to
                            normalization to sum to 1. Default: 0.000000
      --batch_size BATCH_SIZE
                            Number of records to process seeds from at a time. Set
                            lower to avoid out-of-memory errors. Default: 2000
      --find_seq_motifs     Add this flag to call sequence motifs using streme in
                            addition to calling shape motifs.
      --no_shape_motifs     Add this flag to turn off shape motif inference. This
                            is useful if you basically want to use this script as
                            a wrapper for streme to just find sequence motifs.
      --seq_fasta SEQ_FASTA
                            Name of fasta file (located within data_dir, do not
                            include the directory, just the file name) containing
                            sequences in which to search for motifs
      --seq_motif_positive_cats SEQ_MOTIF_POSITIVE_CATS
                            Denotes which categories in `--infile` (or after
                            quantization for a continous signal in the number of
                            bins denoted by the `--continuous` argument) to use as
                            the positive set for sequence motif calling using
                            streme. Example: "4" would use category 4 as the
                            positive set, whereas "3,4" would use categories 3 and
                            4 as the positive set.
      --streme_thresh STREME_THRESH
                            Threshold for including motifs identified by streme.
                            Default: 0.050000
      --seq_meme_file SEQ_MEME_FILE
                            Name of meme-formatted file (file must be located in
                            data_dir) to be used for searching for known sequence
                            motifs of interest in seq_fasta
      --write_all_files     Add this flag to write all motif meme files,
                            regardless of whether the model with shape motifs,
                            sequence motifs, or both types of motifs was most
                            performant.
      --exhaustive          Add this flag to perform and exhaustive initial search
                            for seeds. This can take a very long time for datasets
                            with more than a few-thousand binding sites. Setting
                            this option will override the --max_rounds_no_new_seed
                            option.
      --max_n MAX_N         Sets the maximum number of fasta records to use for
                            motif inference. This is useful when runs are taking
                            prohibitively long.
      --log_level LOG_LEVEL
                            Sets log level for logging module. Valid values are
                            DEBUG, INFO, WARNING, ERROR, CRITICAL.
 
<!--
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

apptainer exec -B $(pwd):$(pwd) \
    shapeme.sif \
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
-->

# Expected output

Several files are generated during a given ShapeME run.

The first file of interest to most users will be
`{data_dir}/shape_main_output/report.html`, where `{data_dir}` is the directory
indicated by the `--data_dir` argument used in the `ShapeME.py infer` call.
Opening the report in a web browser will provide three summaries of the ShapeME run:

1. A summary of the motifs identified,
including each motif's logo, name, adjusted mutual information, z-score, robustness,
and e-value. In the logo representation, opacity of a shape symbol denotes the
weight for that shape at that position in the motif. A transparent symbol denotes
the weight is essentially zero, and that the shape's value is unimportant at that
position.
2. A heatmap with each motif's log2(fold-enrichment) in each score category
is present in the report. For shape motifs, a row is present for each informative
"hit" category for the given motif. For example, if `--max_count 2` was set, the
possible hit categories for a motif are [0,1], [0,2], [1,1], [1,2], and [2,2]. If
each hit category was informative, each will have a row in the heatmap to
indicate for that motif, how enriched that hit category was in each
score category (score categories are columns in the heatmap). For sequence motifs
a single row will be present in the heatmap, as the hits categories for sequence
motifs can only be 0 or 1.
<!-- TODO: AUPR reporting for seq, and seq_and_shape models is under development
3. A plot showing the area under the precision recall curve (AUPR) for the
overall motif model arrived at by ShapeME. For categorical data, AUPRs are shown
for each category. The open circle is the AUPR for inference on the entire dataset,
small dots are AUPRs for each fold of k-fold crossvalidation, and the large dot
is the mean cross-validated AUPR. The horizontal bar is the standard deviation
of the cross-validated AUPR. Note that the AUPR plot is a work in progress,
especially for sequence motif reporting and mixed shape/sequence motif reporting.
-->

If an error was encountered during the ShapeME run, then the report will simply
be a web page containing the text of the error message.

The second file of interest is `{data_dir}/shape_main_output/final_motifs.dsm`,
which contains the numeric values for each motif, including each shape's
z-score at each position in the motif, and the weight applied to each
shape at each position in the motif. For users familiar with the meme file format,
the dsm file will look familiar, as its format was inspired by the meme format.

# Building the Apptainer container locally

Note that if `apptainer` is set up on you system, you should not need
to build your own container. It is our intent that users will submit jobs
via the [web interface to ShapeME](https://seq2fun.dcmb.med.umich.edu/shapeme),
and should a user desire to use ShapeME at the command line, that
they use the instructions above to pull the container directly from
the [Sylabs remote endpoint](https://cloud.sylabs.io/library/schroedj/appliances/shapeme).

If you should find you need to build your own Apptainer container locally rather
than use our pre-built container hosted at cloud.sylabs.io,
take the following steps, substituting
`<src_direc>` and `<path/to/container.sif>` with the location in which you
would like the ShapeME source code and the absolute path to the container
after it is built, respectively:

```bash
# enter the directory you wish to contain the ShapeME source code
cd <src_direc>
git clone https://github.com/freddolino-lab/ShapeME.git
cd ShapeME/singularity
apptainer build <path/to/container.sif> shapeme.def
```

Note that you must have Apptainer installed on your system to build or run
the container.

