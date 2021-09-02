#!/usr/bin/env python3

import unittest
import inout
import motif_maker as mm

"""
Unit tests for motif_maker.py
"""

class Test(unittest.TestCase):

    def setUp(self):
        self.orig_seqs = ['ATGCTGACGATGACGT', 'ATGCTGACGATGACGT']
        self.orig_seq_name = ["peak_1", "peak_2"]
        self.y_vals = [1,0]
        self.motif_seq = 'AATGCG'
        self.sub_seqs = ['ATGAATGCGATGACGT', 'ATGCTGACGATGACGT']
        self.zero_sub_seqs = ['AATGCGACGATGACGT', 'ATGCTGACGATGACGT']
        self.motif_len = len(self.motif_seq)
        self.motif_pos = 3

        self.fa_seqs = inout.FastaFile()

        for i,seq in enumerate(self.orig_seqs):
            fa_entry = inout.FastaEntry(
                header = ">"+self.orig_seq_name[i],
                seq = seq,
            )
            self.fa_seqs.add_entry(fa_entry)
        
    def test_motif_sub_len(self):
        "Test that after substitution, we still have same overall lengths"
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(len(fa_rec.seq), len(self.orig_seqs[i]))

    def test_motif_sub(self):
        "Test that after substitution, we get correct sequence"
        motif_pos = self.motif_pos
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
            motif_pos,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(fa_rec.seq, self.sub_seqs[i])

    def test_zero_idx_motif_sub(self):
        "Test that after substitution at index zero, we get correct sequence"
        motif_pos = 0
        mm.substitute_motif_into_records(
            self.fa_seqs,
            self.y_vals,
            self.motif_seq,
            motif_pos,
        )
        for i,fa_rec in enumerate(self.fa_seqs):
            self.assertEqual(fa_rec.seq, self.zero_sub_seqs[i])

if __name__ == '__main__':
    unittest.main()

