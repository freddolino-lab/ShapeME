#!/usr/bin/env python3

import inout
import numpy as np
import os
import sys
import argparse
import subprocess
from pathlib import Path
from sklearn.model_selection import KFold

this_path = Path(__file__).parent.absolute()

"""
Creates a sequence database and its associated y-values file.
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

def make_categorical_y_vals(n_records, weights=None, n_cats=10):
    '''Make vector of categorical input values'''
    if weights is None:
        weights = np.tile(1.0/n_cats, n_cats)
    y_vals = np.arange(n_cats)
    y_vals = np.random.choice(y_vals, n_records, p=weights)
    # randomly shuffle categories
    np.random.shuffle(y_vals)
    return y_vals

def make_continuous_y_vals(n_records, cat_centers, cat_sd, noise_center, noise_sd, n_cats=10, pivot_cat=False):

    # error out if user didn't pass correct number of centers
    assert len(cat_centers) == n_cats, "ERROR: the number of category centers passed to make_continuous_y_vals must match the number of categories!"
    if pivot_cat:
        tmp_cats = n_cats + 1
    else:
        tmp_cats = n_cats

    y_cats = make_categorical_y_vals(n_records, n_cats=tmp_cats)
    if isinstance(cat_sd, float):
        cat_sd = np.repeat(cat_sd, tmp_cats)
    print(cat_sd)
    # set up y-vals array as noise
    y_vals = np.random.normal(noise_center, noise_sd, n_records)
    pivot_cat = 1
    for category in range(n_cats):
        n_cat_records = len(y_cats[y_cats == category])
        # sample n_cat_records time from this category's mean/sd
        samples = np.random.normal(
            cat_centers[category],
            cat_sd[category],
            n_cat_records,
        )
        # add category value to the noise that's already present
        y_vals[y_cats == category] += samples
        pivot_cat += 1

    if pivot_cat:
        n_cat_records = len(y_cats[y_cats == pivot_cat])
        pivot_samples = np.random.normal(
            np.max(cat_centers) + 5.0,
            0.1,
            n_cat_records,
        )
        y_vals[y_cats == pivot_cat] += pivot_samples

    return y_vals,y_cats
    

def substitute_motif(fa_rec, motif, count_by_strand = (1,0),
                     inter_motif_dist = 5, motif_pos = None):
    '''Substitute the motif's sequence at randomly chosen position.
    
    Args:
    -----
    fa_rec : FastaEntry
    motif : instance of Motif class
    motif_pos : int
    count_by_strand : tuple
        Number of occurrances of motif on (+,-) strands.
    inter_motif_dist : int
    motif_strand : str

    Modifies:
    ---------
    fa_rec : FastaEntry
        Modifes the seq attribute of fa_rec in place.
    '''

    # randomly choose start position for motif within this record
    seq_len = len(fa_rec.seq)
    motif_len = len(motif)

    number_of_occurrences = np.sum(count_by_strand)
    total_len = (
        motif_len * number_of_occurrences
        + inter_motif_dist * (number_of_occurrences - 1)
    )

    if motif_pos is None:
        motif_pos = np.random.choice(
            np.arange(seq_len-total_len), size=1
        )[0]

    # substitute motif at randomly chosen position
    for i,occurrences in enumerate(count_by_strand):
        motif_instance_seq = motif.make_seq_from_pwd()
        for j in range(occurrences):
            upstream_seq = fa_rec.seq[:motif_pos]
            downstream_seq = fa_rec.seq[(motif_pos+motif_len):]
            fa_rec.seq = upstream_seq + motif_instance_seq + downstream_seq
            motif_pos += motif_len + inter_motif_dist


def complement(sequence):

    rc_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    comp_seq = []
    for base in sequence:
        comp_seq.append(rc_dict[base])

    return ''.join(comp_seq)[::-1]


def substitute_motif_into_records(fa_file, y_cats, motifs,
                                  count_by_strand = (1,0),
                                  inter_motif_distance = 5,
                                  motif_pos = None, yval=1, motif_frac = 1.0):
    '''Iterates through records in fa_file and y_vals,
    substituting motif when appropriate.
    '''

    for i,fa_entry in enumerate(fa_file):
        y = y_cats[i]
        if y == yval:
            if np.random.rand(1) < motif_frac:
                if len(motifs) > 1:
                    raise "Insertion of known motifs from a pwm is currently only implemented for meme files containing a single sequence motif pwm."
                else:
                    motif = motifs[0]
                substitute_motif(
                    fa_entry,
                    motif,
                    count_by_strand,
                    inter_motif_distance,
                    motif_pos,
                )

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
    parser.add_argument('--meme_file', action='store', type=str,
                         help="Name of meme-formatted file containing the PWMs from which to draw motif ocurrences")
    parser.add_argument('--motif_peak_frac', action='store', type=float, default=1.0,
                         help="Fraction of peaks to place motif into (default is 1.0)")
    parser.add_argument('--motif_nonpeak_frac', action='store', type=float, default=0.0,
                         help="Fraction of non-peak records to place motif into (default is 0.0)")
    parser.add_argument('--motif-count-plus', action='store', type=int, default=1,
                         help="Number of occurances of the motif in the plus strand")
    parser.add_argument('--motif-count-minus', action='store', type=int, default=1,
                         help="Number of occurances of the motif in the minus strand")
    parser.add_argument('--inter-motif-distance', action='store', type=int, default=5,
                         help="Distance between motif occurrances")
    parser.add_argument('--dtype', action='store', type=str, choices=['continuous','categorical'],
                        help="The type of data to be output.")
    parser.add_argument('--ncats', action='store', type=int, default=2, help="The number of categories containing motifs (see --pivot-category if you want to add one extra category without any motif).")
    parser.add_argument('--pivot-category', action='store_true', default=False, help="If this flag is set, an extra category will be present in the synthetic data in which no motif is expected to be enriched. Ignored if ncats=1, since in that case, cat 1 will have the motif and cat 0 will not.")
    parser.add_argument('--folds', action='store', type=int, default=5, help="The number of folds for k-fold CV.")

    args = parser.parse_args()

    folds = args.folds
    dtype = args.dtype
    ncats = args.ncats
    out_dir = args.outdir
    seq_len = args.seqlen
    rec_num = args.recnum
    frac_peaks = args.fracpeaks
    meme_file = args.meme_file
    plus_strand_count = args.motif_count_plus
    minus_strand_count = args.motif_count_minus
    inter_motif_dist = args.inter_motif_distance
    motif_peak_frac = args.motif_peak_frac
    motif_nonpeak_frac = args.motif_nonpeak_frac
    out_pre = args.outpre

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    motifs = inout.parse_meme_file(meme_file)
    motif_len = len(motifs[0])

    fa_seqs = make_random_seqs(rec_num, length = seq_len)

    if dtype == 'categorical':

        if ncats == 2:
            # weights as [0.75, 0.25] will give about 3x as many 0's as 1's
            y_vals = make_categorical_y_vals(rec_num, weights=[0.8, 0.2], n_cats=ncats)
            # fa_seqs modified in-place here to include the motif at a 
            #  randomly chosen site in each record where y_val is cat
            substitute_motif_into_records(
                fa_seqs,
                y_vals,
                motifs,
                (plus_strand_count, minus_strand_count),
                inter_motif_dist,
                yval = 1,
                motif_frac = motif_peak_frac,
            )


    #    else:
    #        # add one to ncats if the user decided to include a pivot
    #        if args.pivot_category:
    #            ncats += 1
    #        # approximately even number of records per category
    #        y_vals = make_categorical_y_vals(rec_num, n_cats=ncats)
    #        distinct_cats = np.unique(y_vals)
    #        if args.pivot_category:
    #            distinct_cats = distinct_cats[:-1]

    #        for motif,cat in zip(motifs, distinct_cats):
    #            # fa_seqs modified in-place here to include the motif at a 
    #            #  randomly chosen site in each record where y_val is cat
    #            substitute_motif_into_records(
    #                fa_seqs,
    #                y_vals,
    #                motif,
    #                (plus_strand_count, minus_strand_count),
    #                inter_motif_dist,
    #                yval = cat,
    #                motif_frac = motif_peak_frac,
    #            )

    #if dtype == 'continuous':
    #    y_vals,y_cats = make_continuous_y_vals(
    #        rec_num,
    #        cat_centers = np.linspace(
    #            start= -(ncats-1.0)/2.0,
    #            stop= ncats/2.0,
    #            num=ncats
    #        ),
    #        cat_sd = 0.25,
    #        noise_center = 0.0,
    #        noise_sd = 0.25,
    #        n_cats=ncats,
    #        pivot_cat = args.pivot_category,
    #    )
    #    motif_cats = np.unique(y_cats)
    #    if args.pivot_category:
    #        motif_cats = motif_cats[:-1]

    #    for motif,cat in zip(motifs, motif_cats):
    #        # fa_seqs modified in-place here to include the motif at a 
    #        #  randomly chosen site in each record where y_val is cat
    #        substitute_motif_into_records(
    #            fa_seqs,
    #            y_cats,
    #            motif,
    #            (plus_strand_count, minus_strand_count),
    #            inter_motif_dist,
    #            yval = cat,
    #            motif_frac = motif_peak_frac,
    #        )

    if dtype == 'categorical':
        y = y_vals
    #if dtype == 'continuous':
    #    y = y_cats

    # sub some instances of motif into off-target bins at given rate
    if ncats == 2:
        substitute_motif_into_records(
            fa_seqs,
            y,
            motifs,
            (1, 0),
            inter_motif_dist,
            yval = 0,
            motif_frac = motif_nonpeak_frac/2,
        )
        substitute_motif_into_records(
            fa_seqs,
            y,
            motifs,
            (0, 1),
            inter_motif_dist,
            yval = 0,
            motif_frac = motif_nonpeak_frac/2,
        )
    #else:

    #    distinct_cats = np.unique(y)
    #    if args.pivot_category:
    #        distinct_cats = distinct_cats[:-1]

    #    for i,(motif,cat) in enumerate(zip(motifs, distinct_cats)):

    #        other_cats = distinct_cats[distinct_cats != cat]
    #        for other_y in other_cats:
    #            substitute_motif_into_records(
    #                fa_seqs,
    #                y,
    #                motif,
    #                (1, 0),
    #                inter_motif_dist,
    #                yval = other_y,
    #                motif_frac = motif_nonpeak_frac/2,
    #            )
    #            substitute_motif_into_records(
    #                fa_seqs,
    #                y,
    #                motif,
    #                (0, 1),
    #                inter_motif_dist,
    #                yval = other_y,
    #                motif_frac = motif_nonpeak_frac/2,
    #            )

    with open(os.path.join(out_dir, out_pre + "_main.txt"),'w') as main_fire_file:
        main_fire_file.write("name\tscore\n")
        for j,fa_rec in enumerate(fa_seqs):
            main_fire_file.write("{}\t{}\n".format(fa_rec.header[1:], y_vals[j]))
    main_fa_fname = os.path.join(out_dir, out_pre + "_main.fa")
    with open(main_fa_fname, "w") as main_fa_file:
        fa_seqs.write(main_fa_file)

    RSCRIPT = "Rscript {}/utils/calc_shape.R {{}}".format(this_path)
    subprocess.call(RSCRIPT.format(main_fa_fname), shell=True)

    kf = KFold(n_splits=folds)
    for i,(train_index, test_index) in enumerate(kf.split(y_vals)):
        #print(i)
        #print(len(train_index))
        #print(len(test_index))
        train_fa = fa_seqs[train_index]
        train_y = y_vals[train_index]
        test_fa = fa_seqs[test_index]
        test_y = y_vals[test_index]

        with open(os.path.join(out_dir, out_pre + "_train_{}.txt".format(i)) ,'w') as train_fire_file:
            train_fire_file.write("name\tscore\n")
            for j,fa_rec in enumerate(train_fa):
                train_fire_file.write("{}\t{}\n".format(fa_rec.header[1:], train_y[j]))

        with open(os.path.join(out_dir, out_pre + "_test_{}.txt".format(i)) ,'w') as test_fire_file:
            test_fire_file.write("name\tscore\n")
            for j,fa_rec in enumerate(test_fa):
                test_fire_file.write("{}\t{}\n".format(fa_rec.header[1:], test_y[j]))

        train_fa_fname = os.path.join(out_dir, out_pre + "_train_{}.fa".format(i))
        with open(train_fa_fname, "w") as train_fa_file:
            train_fa.write(train_fa_file)

        RSCRIPT = "Rscript {}/utils/calc_shape.R {{}}".format(this_path)
        subprocess.call(RSCRIPT.format(train_fa_fname), shell=True)

        test_fa_fname = os.path.join(out_dir, out_pre + "_test_{}.fa".format(i))
        with open(test_fa_fname, "w") as test_fa_file:
            test_fa.write(test_fa_file)

        subprocess.call(RSCRIPT.format(test_fa_fname), shell=True)


if __name__ == '__main__':
    main()

