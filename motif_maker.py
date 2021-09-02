#!/usr/bin/env python3

import inout
import numpy as np
import os
import argparse
import subprocess

"""
A script for generating a synthetic dataset for testing motif identification
tools. Generates N records with Y peaks. Calls DNAShapeR on the sequences and
creates an inout.RecordDatabase object, then writes the recorddatabase as a
pickle file.
"""


def random_sequence_generator(length=60):
    '''Make random DNA sequence of specified length'''

    bases = ['A','C','T','G']
    seq = np.random.choice(bases, length)
    seq_str = ''.join(seq)

    return seq_str

def make_random_seqs(n_records, length=60):
    '''Make n_records random sequences.'''

    fa_seqs = inout.FastaFile()
    for n in range(n_records):
        seq_header = ">peak_{:0=5d}".format(n+1)
        seq = random_sequence_generator(length)
        fa_seq = inout.FastaEntry(
            header = seq_header,
            seq = seq,
        )
        fa_seqs.add_entry(fa_seq)

    return fa_seqs

def make_binary_y_vals(n_records, frac_true=0.2):
    '''Make vector of ground-truth'''
    
    y_vals = np.random.binomial(1, p=frac_true, size=n_records)
    return y_vals

def substitute_motif(fa_rec, motif_seq, motif_pos = None):
    "Substitute the motif's sequence at randomly chosen position"

    # randomly choose start position for motif within this record
    seq_len = len(fa_rec.seq)
    motif_len = len(motif_seq)
    if motif_pos is None:
        motif_pos = np.random.choice(
            np.arange(seq_len-motif_len), size=1
        )[0]

    # substitute motif at randomly chosen position
    upstream_seq = fa_rec.seq[:motif_pos]
    downstream_seq = fa_rec.seq[(motif_pos+motif_len):]
    new_seq = upstream_seq + motif_seq + downstream_seq
    fa_rec.seq = new_seq

def substitute_motif_into_records(fa_file, y_vals, motif_seq, motif_pos = None):
    '''Iterates through records in fa_file and y_vals,
    substituting motif when appropriate.
    '''

    for i,fa_rec in enumerate(fa_file):
        y = y_vals[i]
        if y == 1:
            substitute_motif(fa_rec, motif_seq, motif_pos)

def predict_shapes():


    rinterface.initr()

    try:
        rinterface.endr()
    except Error as e:
        rinterface.endr()
        print(e)
        sys.exit("An error occured, ending R session")

    pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', action='store', type=str,
                         help='name of directory into which to write files')
    parser.add_argument('--recnum', action='store', type=int,
                         help="number of records to write to output files")
    parser.add_argument('--fracpeaks', action='store', type=float,
                         help="fraction of records to make peaks")
    parser.add_argument('--outpre', action='store', type=str,
                         help="prefix to place at beginning of output file names.")
    parser.add_argument('--seqlen', action='store', type=int, default=60,
                         help="Length of sequences in synthetic dataset (default is 60).")
    parser.add_argument('--motif', action='store', type=str,
                         help="Sequence of the motif to place in the peaks")
    parser.add_argument('--motif_peak_frac', action='store', type=float, default=1.0,
                         help="Fraction of peaks to place motif into (default is 1.0)")
    parser.add_argument('--motif_nonpeak_frac', action='store', type=float, default=0.0,
                         help="Fraction of non-peak records to place motif into (default is 0.0)")
    args = parser.parse_args()

    out_dir = args.outdir
    seq_len = args.seqlen
    rec_num = args.recnum
    frac_peaks = args.fracpeaks
    motif = args.motif
    motif_peak_frac = args.motif_peak_frac
    motif_nonpeak_frac = args.motif_nonpeak_frac
    out_pre = args.outpre

    motif_len = len(motif)

    fa_seqs = make_random_seqs(rec_num, length = seq_len)
    y_vals = make_binary_y_vals(rec_num, frac_true = frac_peaks)

    # fa_seqs modified in-place here to include the motif at a 
    #  randomly chosen site in each record where y_val is 1
    substitute_motif_into_records(
        fa_seqs,
        y_vals,
        motif,
    )

    with open(os.path.join(out_dir, out_pre + ".txt") ,'w') as fire_file:
        fire_file.write("name\tscore\n")
        for i,fa_rec in enumerate(fa_seqs):
            fire_file.write("{}\t{}\n".format(fa_rec.header[1:], y_vals[i]))

    fa_fname = os.path.join(out_dir, out_pre + ".fa")
    with open(fa_fname, "w") as fa_file:
        fa_seqs.write(fa_file)

    RSCRIPT = "Rscript calc_shape.R {}"
    RSCRIPT = RSCRIPT.format(fa_fname)
    subprocess.call(RSCRIPT, shell=True)


if __name__ == '__main__':
    main()

