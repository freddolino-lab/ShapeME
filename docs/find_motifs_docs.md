# Documentation of arguments to `find_motifs.py`

The behavior of shape- and sequence-based motif inference run using the
`find_motifs.py` script can be modified in many ways, documented below.

Brief documentation on each argument can also be accessed by running
`singularity exec scheme.sif python find_motifs.py --help`

## Required arguments

### score\_file

`--score_file` should be the base name of the
file containing one line for each sequence. There are two, tab-delimited
columns. The file *must have one header line*. The columns are as follows:

1. the name of the sequence
2. the score assigned to the sequence

The sequence names in columns one must be the sequence names, in order,
found in the files passed in the `--shape_files` argument or the `--seq_fasta`
argument if identifying sequence motifs.

### shape\_files

`--shape_files` is a space-separated list of the fasta-like formatted files
containing shape scores for each sequence. For example, if you were using
the shapes "Roll" and "MGW", you would use this argument as follows:

`--shape_files prefix.fa.Roll prefix.fa.MGW`

### shape\_names

`--shape_names` is a space-separated list of shape names used in motif
finding. So for the example in the `--shape\_files` section above,
the `--shape_names` argurment would be as follows:

`--shape_names Roll MGW`

Note that is it absolutely necessary that the shape names and their
corresponding shape files are in the same order.

### out\_prefix

`--out_prefix` sets the characters to prepend to the main output files
produced by SCHEME.

### data\_dir

`--data_dir` should be the absolute path where all input files
are located.

### out\_dir

`--out_dir` is the basename of the directory into which output files will be written.
This directory will be created within `--data_dir` if it does not already
exist at that location. For example, if `find_motifs.py` is run with
`--data_dir /home/user/scheme_inputs` and `--out_dir scheme_results`,
then the output files produced by SCHEME will be in
`/home/user/scheme_inputs/scheme_results`

## Optional arguments

We have endeavoured to set the following optional arguments to reasonable
default values for most use cases, but users should familiarize themselves
with these arguments and consider adjusting them if they would like
to tweak the behavior of SCHEME.

For all of these options, the default values could change with any
new SCHEME release, so it is highly recommended that the user
run `singularity exec scheme.sif python find_motifs.py --help`
whenever switching to a new singularity container to check that
the default values are, indeed, desireable.

### kmer

`--kmer` sets the width of the motif(s) that can be returned by SCHEME.

### max\_count

`--max_count` sets the maximum number of times a motif can match
each of the forward and reverse strands in a reference.
This option's default value will always be 1.

### continuous

`--continuous` sets the number of bins to discretize continuous input
data into. The records in the input data will be binned into
approximately-evenly populated bins. If the user would like to
bias the binning in some way, we recommend they create their own
custom categorical scores to pass as their `--score_file` argument.

### threshold\_sd

SCHEME takes a first-pass search of the input data, looking for seeds
whose manhattan distance is below some threshold value (the threshold
for seed finding can be tuned using this argument). 

Initially, manhattan distances between randomly-selected seeds are
calculated from randomly-selected targets. The mean and standard devaition
of those distances are calculated.

`--threshold_sd` sets how many std deviations below the mean will be used
as the initial threshold for seed finding.

### init\_threshold\_seed\_num

`--init_threshold_seed_num` set the number of randomly selected seeds
to compare to records in the database during initial threshold setting.
It is unlikely that this number will need adjusting by the user.

### init\_threshold\_recs\_per\_seed

`--init_threshold_recs_per_seed` sets the number of randomly selected
records to compare to each seed during initial threshold setting.
It is unlikely that this number will need adjusting by the user.

### init\_threshold\_windows\_per\_record

`--init_threshold_windows_per_record` sets the number of randomly
selected windows within a given record to compare to each seed
during initial threshold setting.
It is unlikely that this number will need adjusting by the user.

### batch\_size

To avoid using excessive memory, input records are evaluated in
batches for the initial seed search. `--batch_size` sets the number
of records to process seeds from at a time. If you are getting out-of-memory
errors, you should reduce this number.

### max\_batch\_no\_new\_seed

`--max_batch_no_new_seed` sets the maximum number of batches of seed
evaluation that will be performed with no new motifs added to the set
of seedss to be optimized.

If a new seed is identified in an batch i, with no new seeds identified
from batches i through i + `max_batch_no_new_seed`, the initial
seed evaluation will terminate, and SCHEME will move on the motif
optimization.

In our experience we have found that performing and exhaustive search
for seeds is very rarely necessary, and that only a handful of batches
are required to identify all the seeds present in the input data.
Therefore, we typically find a value of 10 to be useful for this argument.

### exhaustive

`--exhaustive` overrides the `--max-rounds-no-new-seed` argument to instead
perform an exhaustive initial search for seeds. Depending on
the size of the input data, this may GREATLY increase the time required
to run SCHEME. The potential benefit is that is could enable the detection
of more initial seeds to be later optimized.

### nprocs

`--nprocs` sets number of processors for parallel evaluation of seeds
and optimization of motifs. In our experience, SCHEME scales well on
up to 64 cores, which diminishing returns above that number.

### threshold\_constraints

After the initial seed identification phase, seeds are optimized
to motifs by maximization of their mutual information with the
input score categories.

`--threshold_constraints` sets the upper and lower limits on the
match threshold during optimization.

### shape\_constraints

`--shape_constraints` sets the upper and lower limits on the shapes'
z-scores during optimization.

### weights\_constraints

TODO: improve this doc, doesn't help user much

`-weights_constraints` sets the upper and lower limits on
the pre-transformed, pre-normalized weights during optimization.

### temperature

The optimization algorithm employed by SCHEME is simulated annealing.

`--temperature` sets the starting temperature for simulated
annealing.

### t\_adj

`--t_adj` sets the fraction by which temperature decreases
each iteration of simulated annealing.

### stepsize

`--stepsize` defines how far a given value (shape, weight, or threshold)
can be nudged between simulated annealing iterations. A higher value
will allow larger jumps.

### alpha

TODO: improve this doc, doesn't help user much

`--alpha` sets the lower limit on transformed weight values prior to
normalization to sum to 1.

### opt\_niter

`--opt_niter` sets the number of simulated annealing iterations to
undergo during optimization. A higher number will yeild better
motifs, but will increase run time.

### find\_seq\_motifs

`--find_seq_motifs` - Add this flag to call sequence motifs in addition
to calling shape motifs. Sequence motif finding is performed by wrapping
STREME and FIMO into `find_motifs.py`. If using our singularity container,
STREME and FIMO should just work without further modification to you
compute environment.

### no\_shape\_motifs

`--no_shape_motifs` - Add this flag to turn off shape motif inference.
This is useful if you basically want to use this script as a wrapper for
streme to just find sequence motifs.

### seq\_fasta

This is a conditionall-required argument; if you have `--find_seq_motifs`
set, you must give `find_motifs.py` a sequence fasta file. Records in the
fasta file must be in the same order as they appear in the input scores file.

`--seq_fasta` should be the basename of fasta file (located within `in_direc`,
containing sequences in which to search for motifs.

### seq\_motif\_positive\_cats

TODO: consider adjusting

`--seq_motif_positive_cats` denotes which categories in `--infile`
(or after quantization of a continous signal in the number of
bins denoted by the `--continuous` argument) to use as the positive
set for sequence motif calling using streme.

Example: "4" would use category 4 as the positive set, whereas 
"3,4" would use categories 3 and 4 as the positive set.

For continuous data, we often set something like the following `--continous 10
--seq_motif_positive_cats 9`. NOTE: category IDs are zero-indexed,
so although there are 10 categories in
this example, the numbering of categories begins at 0, so category 9 is
the highest-value category when `--continuous 10` is used.

### streme\_thresh

STREME outputs more motifs than are "significant".

`--streme_thresh` sets the threshold for including motifs identified
by streme. We typically use 0.05.

### seq\_meme\_file

TODO: needs a bit better documentation

`--seq_meme_file` sets the name of meme-formatted file (file must be
located in `data_dir`) to be used for searching for known sequence
motifs of interest in `seq_fasta`. This option enables incorporation
of known sequence motifs into a combined sequence/shape model.

### shape\_rust\_file

TODO: get details on whether this requires `--no_shape_motifs`, add here.

This option is rarely used in practice, but can be useful if a
SCHEME run was performed to identify shape motifs, and that result
is to be later used to combine a shape motif model with a sequence
motif model.

`--shape_rust_file` - is the name of a json file containing output
from rust binary. It will be read so that shape motifs can be used
and potentially combined with sequence motifs, without the need
to re-run prior shape motif inference.

### write\_all\_files

If both sequence and shape motifs were identified, only the best model
(shape only, sequence only, both) will have motifs written the the
output directory. Use this flag to force SCHEME to write all models'
motifs.

`--write_all_files` - Add this flag to write all motif meme files,
regardless of whether the model with shape motifs, sequence motifs,
or both types of motifs was most performant.

### log\_level

`--log_level` sets log level for logging module. Valid values are DEBUG,
INFO, WARNING, ERROR, CRITICAL.

Setting to DEBUG will output more logging to stdout. Very little will
be printed to stdout if set to CRITICAL.

