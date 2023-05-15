import unittest
import sys
import os
from pathlib import Path

this_path = os.path.join(Path(__file__).parent.absolute(), "../")
sys.path.insert(0, this_path)

import inout

class TestMotifMethods(unittest.TestCase):
    pass

class TestRecordsMethods(unittest.TestCase):

    def setUp(self):
        self.records = inout.construct_records(
            in_direc = "test_data",
            shape_names = ["EP", "HelT", "MGW", "ProT", "Roll"],
            shape_files = ["test_data_binary_plus_test_0.fa.EP", "test_data_binary_plus_test_0.fa.HelT", "test_data_binary_plus_test_0.fa.MGW", "test_data_binary_plus_test_0.fa.ProT", "test_data_binary_plus_test_0.fa.Roll"],
            in_fname = "test_data_binary_plus_test_0.txt",
        )
        self.records.initialize_weights()

        # read seq fasta, keep indices        
        with open("test_data/test_data_binary_plus_test_0.fa","r") as seq_f:
            seqs = inout.FastaFile()
            seqs.read_whole_file(seq_f)
        self.seqs = seqs

        self.seed = 42
 
    def test_sample(self):

        orig_len = 600
        self.assertEqual(len(self.records), orig_len)
        self.assertEqual(len(self.seqs), orig_len)

        # with rng seed = 42, these indices are retained
        true_inds = [461, 261, 513, 415, 53]
        n_samps = 5

        ret_inds = self.records.sample(n_samps, inplace=True, rng_seed=self.seed)
        shape_recs = list(self.records.record_name_lut.keys())

        # assert I'm getting the right number of records
        self.assertEqual(len(self.records), n_samps)
        # assert I'm getting the right indices
        for target_idx, result_idx in zip(true_inds,ret_inds):
            self.assertEqual(target_idx, result_idx)

        ret_seqs = self.seqs[ret_inds]
        # assert I'm getting the right number of records
        self.assertEqual(len(ret_seqs), n_samps)
        seq_recs = ret_seqs.names
        # assert I'm getting the same records by their names
        for seq_rec,shape_rec in zip(seq_recs, shape_recs):
            self.assertEqual(seq_rec,shape_rec)

    def test_kfold_no_seq(self):

        # get 5 records
        n_samps = 5
        ret_inds = self.records.sample(n_samps, inplace=True, rng_seed=self.seed)

        folds = self.records.split_kfold(n_samps, rng_seed=self.seed)
        self.assertIsNone(folds[1][1][1])

    def test_kfold_with_seq(self):

        # get 5 records
        n_samps = 5
        ret_inds = self.records.sample(n_samps, inplace=True, rng_seed=self.seed)
        ret_seqs = self.seqs[ret_inds]

        folds = self.records.split_kfold(n_samps, ret_seqs, rng_seed=self.seed)

        for k,fold in enumerate(folds):
            shape_name = list(fold[1][0].record_name_lut.keys())[0]
            seq_name = fold[1][1].names[0]

            out_direc = "test_outs"
            train_base = f"train_fold_{k}"
            test_base = f"test_fold_{k}"
            train_shapes = fold[0][0]
            test_shapes = fold[1][0]

            train_shapes.write_to_files(out_direc, train_base)
            test_shapes.write_to_files(out_direc, test_base)

            self.assertEqual(shape_name, seq_name)

        self.assertEqual(len(folds), n_samps)


