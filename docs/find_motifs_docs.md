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

`--shape_names`
        help='parameter names (MUST BE IN SAME ORDER AS CORRESPONDING PARAMETER FILES)')

### out\_prefix

    parser.add_argument('--out_prefix', type=str, required=True,
        help="Prefix to apply to output files.")

### data\_dir

    parser.add_argument('--data_dir', type=str, required=True,
        help="Directory from which input files will be read.")

### out\_dir

    parser.add_argument('--out_dir', type=str, required=True,
        help="Directory (within 'data_dir') into which output files will be written.")

## Optional arguments

### kmer

    parser.add_argument('--kmer', type=int,
        help='kmer size to search for. Default=%(default)d', default=15)

### max\_count

    parser.add_argument('--max_count', type=int, default=1,
        help=f"Maximum number of times a motif can match "\
            f"each of the forward and reverse strands in a reference. "\
            f"Default: %(default)d")

### continuous

    parser.add_argument('--continuous', type=int, default=None,
        help="number of bins to discretize continuous input data with")

### max\_batch\_no\_new\_seed

    parser.add_argument("--max_batch_no_new_seed", type=int, default=10,
        help=f"Sets the number of batches of seed evaluation with no new motifs "\
            f"added to the set of motifs to be optimized prior to truncating the "\
            f"initial search for motifs.")

### nprocs

    parser.add_argument('--nprocs', type=int, default=1,
        help="number of processors. Default: %(default)d")

### threshold\_constraints

    parser.add_argument('--threshold_constraints', nargs=2, type=float, default=[0,10],
        help=f"Sets the upper and lower limits on the match "\
            f"threshold during optimization. Defaults to 0 for the "\
            f"lower limit and 10 for the upper limit.")

### shape\_constraints

    parser.add_argument('--shape_constraints', nargs=2, type=float, default=[-4,4],
        help=f"Sets the upper and lower limits on the shapes' z-scores "\
            f"during optimization. Defaults to -4 for the lower limit "\
            f"and 4 for the upper limit.")

### weights\_constraints

    parser.add_argument('--weights_constraints', nargs=2, type=float, default=[-4,4],
        help="Sets the upper and lower limits on the pre-transformed, "\
            f"pre-normalized weights during optimization. Defaults to -4 "\
            f"for the lower limit and 4 for the upper limit.")

### temperature

    parser.add_argument('--temperature', type=float, default=0.4,
        help=f"Sets the temperature argument for simulated annealing. "\
            f"Default: %(default)f")

### t\_adj

    parser.add_argument('--t_adj', type=float, default=0.001,
        help=f"Fraction by which temperature decreases each iteration of "\
            f"simulated annealing. Default: %(default)f")

### stepsize

    parser.add_argument('--stepsize', type=float, default=0.25,
        help=f"Sets the stepsize argument for scipy.optimize.basinhopping. "\
            f"Default: %(default)f")

### alpha

    parser.add_argument('--alpha', type=float, default=0.0,
        help=f"Lower limit on transformed weight values prior to "\
            f"normalization to sum to 1. Default: %(default)f")

### opt\_niter

    parser.add_argument('--opt_niter', type=int, default=100,
        help=f"Sets the number of simulated annealing iterations to "\
            f"undergo during optimization. Default: %(default)d.")

### threshold\_sd

    parser.add_argument('--threshold_sd', type=float, default=2.0, 
        help=f"std deviations below mean for seed finding. "\
            f"Only matters for greedy search. Default=%(default)f")

### init\_threshold\_seed\_num

    parser.add_argument('--init_threshold_seed_num', type=int, default=500, 
        help=f"Number of randomly selected seeds to compare to records "\
            f"in the database during initial threshold setting. Default=%(default)d")

### init\_threshold\_recs\_per\_seed

    parser.add_argument('--init_threshold_recs_per_seed', type=int, default=20, 
        help=f"Number of randomly selected records to compare to each seed "\
            f"during initial threshold setting. Default=%(default)d")

### init\_threshold\_windows\_per\_record

    parser.add_argument('--init_threshold_windows_per_record', type=int, default=2, 
        help=f"Number of randomly selected windows within a given record "\
            f"to compare to each seed during initial threshold setting. "\
            f"Default=%(default)d")

### batch\_size

    parser.add_argument('--batch_size', type=int, default=2000,
        help=f"Number of records to process seeds from at a time. Set lower "\
            f"to avoid out-of-memory errors. Default: %(default)d")

### find\_seq\_motifs

    parser.add_argument('--find_seq_motifs', action="store_true",
        help=f"Add this flag to call sequence motifs using streme in addition "\
            f"to calling shape motifs.")

### no\_shape\_motifs

    parser.add_argument("--no_shape_motifs", action="store_true",
        help=f"Add this flag to turn off shape motif inference. "\
            f"This is useful if you basically want to use this script "\
            f"as a wrapper for streme to just find sequence motifs.")

### seq\_fasta

    parser.add_argument("--seq_fasta", type=str, default=None,
        help=f"Name of fasta file (located within in_direc, do not include the "\
            f"directory, just the file name) containing sequences in which to "\
            f"search for motifs")

### seq\_motif\_positive\_cats

    parser.add_argument('--seq_motif_positive_cats', required=False, default="1",
        action="store", type=str,
        help=f"Denotes which categories in `--infile` (or after quantization "\
            f"for a continous signal in the number of bins denoted by the "\
            f"`--continuous` argument) to use as the positive "\
            f"set for sequence motif calling using streme. Example: "\
            f"\"4\" would use category 4 as the positive set, whereas "\
            f"\"3,4\" would use categories 3 and 4 as "\
            f"the positive set.")

### streme\_thresh

    parser.add_argument('--streme_thresh', default = 0.05,
        help="Threshold for including motifs identified by streme. Default: %(default)f")

### seq\_meme\_file

    parser.add_argument("--seq_meme_file", type=str, default=None,
        help=f"Name of meme-formatted file (file must be located in data_dir) "\
            f"to be used for searching for known sequence motifs of interest in "\
            f"seq_fasta")

### shape\_rust\_file

    parser.add_argument("--shape_rust_file", type=str, default=None,
        help=f"Name of json file containing output from rust binary")

### write\_all\_files

    parser.add_argument("--write_all_files", action="store_true",
        help=f"Add this flag to write all motif meme files, regardless of whether "\
            f"the model with shape motifs, sequence motifs, or both types of "\
            f"motifs was most performant.")

### exhaustive

    parser.add_argument("--exhaustive", action="store_true", default=False,
        help=f"Add this flag to perform and exhaustive initial search for seeds. "\
            f"This can take a very long time for datasets with more than a few-thousand "\
            f"binding sites. Setting this option will ignore the --max-rounds-no-new-seed "\
            f"option.")

### log\_level

    parser.add_argument("--log_level", type=str, default="INFO",
        help=f"Sets log level for logging module. Valid values are DEBUG, "\
                f"INFO, WARNING, ERROR, CRITICAL.")

