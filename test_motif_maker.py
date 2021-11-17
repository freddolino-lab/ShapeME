#!/usr/bin/env python3

import unittest
import inout
import motif_maker as mm

"""
Unit tests for motif_maker.py
"""

class Test(unittest.TestCase):

    def setUp(self):
        self.orig_seqs = ['ATGCTGACGATGACGT', 'ATGCTGACGATGACGT', 'ATGCTGACGATGACGT']
        self.orig_seq_name = ["peak_1", "peak_2", "peak_3"]
        self.y_vals = [1,0,1]
        self.motif_seq = 'AATGCG'
        self.sub_seqs = ['ATGAATGCGATGACGT', 'ATGCTGACGATGACGT', 'ATGAATGCGATGACGT']
        self.zero_sub_seqs = ['AATGCGACGATGACGT', 'ATGCTGACGATGACGT', 'AATGCGACGATGACGT']
        self.two_hits_seqs = ['AAATGCGCGCGCATTT', 'ATGCTGACGATGACGT', 'AAATGCGCGCGCATTT']

        self.motif_len = len(self.motif_seq)
        self.motif_pos = 3
        self.count_by_strand = (1,0)

        self.fa_seqs = inout.FastaFile()

        for i,seq in enumerate(self.orig_seqs):
            fa_entry = inout.FastaEntry(
                header = ">"+self.orig_seq_name[i],
                seq = seq,
            )
            self.fa_seqs.add_entry(fa_entry)
        
    def test_motif_sub_len(self):
        "Test that after substitution, we still have same overall lengths"
        fa_seqs = self.fa_seqs.copy()
        mm.substitute_motif_into_records(
            fa_seqs,
            self.y_vals,
            self.motif_seq,
            count_by_strand = (1,1),
            inter_motif_distance = 1,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(len(fa_rec.seq), len(self.orig_seqs[i]))

    def test_motif_sub(self):
        "Test that after substitution, we get correct sequence"
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
            self.count_by_strand,
            motif_pos = self.motif_pos,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(fa_rec.seq, self.sub_seqs[i])

    def test_multi_motif_sub(self):
        "Test multiple substitutions"
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
            count_by_strand = (1,1),
            motif_pos = 1,
            inter_motif_distance = 2,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(fa_rec.seq, self.two_hits_seqs[i])

    def test_zero_idx_motif_sub(self):
        "Test that after substitution at index zero, we get correct sequence"
        motif_pos = 0
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
            self.count_by_strand,
            motif_pos = motif_pos,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(fa_rec.seq, self.zero_sub_seqs[i])


if __name__ == '__main__':
    unittest.main()

