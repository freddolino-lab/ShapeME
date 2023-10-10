import unittest
import sys
import os
import numpy as np
from pathlib import Path
import pickle

this_path = os.path.join(Path(__file__).parent.absolute(), "../")
sys.path.insert(0, this_path)

import inout

class TestFastaMethods(unittest.TestCase):

    def setUp(self):
        self.vals_in_files = {
            "EP": {
                "peak_00001": np.array([ np.NaN,np.NaN,-6.81,-7.20,-5.43,-6.73,np.NaN,np.NaN ]),
                "peak_00002": np.array([ np.NaN,np.NaN,-6.99,-5.23,-7.65,-6.00,np.NaN,np.NaN ]),
            },
            "HelT": {
                "peak_00001": np.array([ np.NaN,33.01,34.56,31.80,35.73,33.98,35.06,np.NaN ]),
                "peak_00002": np.array([ np.NaN,35.38,31.08,34.22,35.34,34.84,35.04,np.NaN ]),
            },
            "MGW": {
                "peak_00001": np.array([ np.NaN,np.NaN,5.34,4.99,5.27,5.74,np.NaN,np.NaN ]),
                "peak_00002": np.array([ np.NaN,np.NaN,5.47,5.67,4.94,4.18,np.NaN,np.NaN ]),
            },
            "ProT" : {
                "peak_00001": np.array([ np.NaN,np.NaN,-9.01,-10.15,-7.40,-8.60,np.NaN,np.NaN ]),
                "peak_00002": np.array([ np.NaN,np.NaN,-5.31,-6.43,-7.96,-5.67,np.NaN,np.NaN ]),
            },
            "Roll": {
                "peak_00001": np.array([ np.NaN,2.56,4.14,-4.35,5.65,-2.69,3.24,np.NaN ]),
                "peak_00002": np.array([ np.NaN,-2.31,-2.96,4.10,-1.65,-3.71,-2.46,np.NaN ]),
            },
        }
        self.EP_name = "test_data/small_test_data.fa.EP"
        self.HelT_name = "test_data/small_test_data.fa.HelT"
        self.MGW_name = "test_data/small_test_data.fa.MGW"
        self.ProT_name = "test_data/small_test_data.fa.ProT"
        self.Roll_name = "test_data/small_test_data.fa.Roll"

        self.EP_info = inout.parse_shape_fasta(self.EP_name)
        self.HelT_info = inout.parse_shape_fasta(self.HelT_name)
        self.MGW_info = inout.parse_shape_fasta(self.MGW_name)
        self.ProT_info = inout.parse_shape_fasta(self.ProT_name)
        self.Roll_info = inout.parse_shape_fasta(self.Roll_name)

    def test_read_shape_fa(self):

        result = {
            "EP": self.EP_info,
            "HelT": self.HelT_info,
            "MGW": self.MGW_info,
            "ProT": self.ProT_info,
            "Roll": self.Roll_info,
        }

        for shape_name,shape_vals in result.items():
            target_shapes = self.vals_in_files[shape_name]
            for peak_name,peak_vals in shape_vals.items():
                target_peak = target_shapes[peak_name]
                for i,val in enumerate(peak_vals):
                    if np.isnan(val):
                        self.assertTrue(np.isnan(target_peak[i]))
                    else:
                        self.assertEqual(val, target_peak[i])

    def test_make_shape_array(self):
        # should be shape (R,P,S,2), where R is the number of records in
        #the input data, P is the number of positions (the length of)
        #each record, and S is the number of shape parameters present.
        #The final axis is of length 2, one index for each strand.
        # so dim should be (2,4,5,2)
        target_shapes = np.array([
            # peak_0001
            [
                # pos1
                [
                    # EP
                    [ -6.81, -6.73 ],
                    # HelT
                    [ 34.56, 33.98 ], 
                    # MGW
                    [ 5.34, 5.74 ],
                    # ProT
                    [ -9.01, -8.6 ],
                    # Roll
                    [ 4.14, -2.69 ],
                ],

                # pos2
                [
                    # EP
                    [ -7.2, -5.43 ],
                    # HelT
                    [ 31.80, 35.73 ],
                    # MGW
                    [ 4.99, 5.27 ],
                    # ProT
                    [ -10.15,  -7.4 ],
                    # Roll
                    [ -4.35, 5.65 ],
                ],

                # pos3
                [
                    # EP
                    [ -5.43, -7.2 ],
                    # HelT
                    [ 35.73, 31.80 ],
                    # MGW
                    [ 5.27, 4.99 ],
                    # ProT
                    [ -7.4,  -10.15 ],
                    # Roll
                    [ 5.65, -4.35 ],
                ],

                # pos4
                [
                    # EP
                    [ -6.73, -6.81 ],
                    # HelT
                    [ 33.98, 34.56 ], 
                    # MGW
                    [ 5.74, 5.34 ],
                    # ProT
                    [ -8.6,  -9.01 ],
                    # Roll
                    [ -2.69, 4.14 ],
                ],
            ],

            # peak_0002
            [
                # pos1
                [
                    # EP
                    [ -6.99, -6.0 ],
                    # HelT
                    [ 31.08, 34.84 ],
                    # MGW
                    [ 5.47, 4.18 ],
                    # ProT
                    [ -5.31, -5.67 ],
                    # Roll
                    [ -2.96, -3.71 ],
                ],

                # pos2
                [
                    # EP
                    [ -5.23, -7.65 ],
                    # HelT
                    [ 34.22, 35.34 ],
                    # MGW
                    [ 5.67, 4.94 ],
                    # ProT
                    [ -6.43, -7.96 ],
                    # Roll
                    [ 4.10, -1.65 ],
                ],

                # pos3
                [
                    # EP
                    [ -7.65, -5.23 ],
                    # HelT
                    [ 35.34, 34.22 ],
                    # MGW
                    [ 4.94, 5.67 ],
                    # ProT
                    [ -7.96, -6.43 ],
                    # Roll
                    [ -1.65, 4.10 ],
                ],

                # pos4
                [
                    # EP
                    [ -6.0, -6.99 ],
                    # HelT
                    [ 34.84, 31.08 ],
                    # MGW
                    [ 4.18, 5.47 ],
                    # ProT
                    [ -5.67, -5.31 ],
                    # Roll
                    [ -3.71, -2.96 ],
                ],
            ],
        ])
        target_array_dim = target_shapes.shape

        result = inout.RecordDatabase(
            shape_dict = {
                "EP": self.EP_name,
                "HelT": self.HelT_name,
                "MGW": self.MGW_name,
                "ProT": self.ProT_name,
                "Roll": self.Roll_name,
            },
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
            y = np.zeros(2),
            record_names = ["peak_00001", "peak_00002"],
        )

        arr = result.X
        result_dim = arr.shape

        # ensure same number of axes
        self.assertEqual(len(target_array_dim), len(result_dim))

        # ensure shapes of arrays are same
        for i,dim_size in enumerate(target_array_dim):
            self.assertEqual(dim_size, result_dim[i])

        same = np.all(target_shapes == arr)
        self.assertTrue(same)



class TestMotifMethods(unittest.TestCase):

    def setUp(self):

        self.maxDiff = None
        shape_str = "EP HelT MGW ProT Roll"
        top = f"ALPHABET= ACGT\nSHAPES= {shape_str}\n"
        mo_header_1 = f"MOTIF SHAPE-1 None\n"\
            f"shape-value matrix: alength= 5 w= 10 "\
            f"threshold= 1.209 adj_mi= 0.947 "\
            f"z-score= 2037.16 robustness= 10/10\n\n"
        mo_header_2 = f"MOTIF SHAPE-2 None\n"\
            f"shape-value matrix: alength= 5 w= 10 "\
            f"threshold= 1.327 adj_mi= 0.916 "\
            f"z-score= 1408.32 robustness= 10/10\n\n"
        mo_header_3 = f"MOTIF 2-ACATGCAGTC STREME-2\n"\
            f"letter-probability matrix: alength= 4 "\
            f"w= 10 nsites= 665 adj_mi= 0.873 "\
            f"z-score= 1889.79 robustness= 10/10 "\
            f"E= 3.9e-44\n\n"
        mo_header_4 = f"MOTIF 1-ACATGCAGTCRMBVD STREME-1\n"\
            f"letter-probability matrix: alength= 4 "\
            f"w= 15 nsites= 673 adj_mi= 0.857 "\
            f"z-score= 1859.62 robustness= 10/10 "\
            f"E= 4.2e-62\n\n"
        self.true_str = top + mo_header_1 + mo_header_2 + mo_header_3 + mo_header_4

        motif_fname = "test_data/test_motifs.dsm"
        self.out_dsm_name = "test_outs/test_write_motifs.dsm"

        self.motifs = inout.Motifs()
        self.motifs.read_file( motif_fname )

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

        small_motifs_dsm_name = "test_data/small_test_motifs.dsm"
        self.small_motifs = inout.Motifs()
        self.small_motifs.read_file( small_motifs_dsm_name )

    def test_unnormalize(self):
        self.small_motifs.unnormalize()
        self.assertEqual(1,2)

    def test_motif_shape(self):
        self.assertEqual(self.motifs[0].shape()[0], 5)
        self.assertEqual(self.motifs[0].shape()[1], 10)

    def test_motif_length(self):
        self.assertEqual(len(self.motifs[0]), 10)

    def test_unstandardize(self):

        self.small_motifs.unnormalize()

        target_EP1 = np.array([0.922595 * 1 + -6, -0.187693 * 1 - 6, 1.148317 * 1 - 6, -3.284317 * 1 - 6])
        target_HelT1 = np.array([-0.933015 * 1.5 + 34, -2.269072 * 1.5 + 34, 0.964941 * 1.5 + 34, 1.207266 * 1.5 + 34])
        target_MGW1 = np.array([-3.035292 * 0.5 + 5.0, -0.839392 * 0.5 + 5.0, 1.199173 * 0.5 + 5.0, -1.742476 * 0.5 + 5.0])
        target_ProT1 = np.array([-2.719605 * 4.0 + -7, 3.987507 * 4.0 + -7, 0.115779 * 4.0 + -7, -3.089078 * 4.0 + -7])
        target_Roll1 = np.array([-2.461327 * 1 + -2, 2.132439 * 1 + -2, 0.806532 * 1 + -2, -0.975881 * 1 + -2])

        target_EP2 = np.array([-1.707178 * 1 + -6, 1.910599 * 1 + -6, 0.075611 * 1 + -6, -3.472651 * 1 + -6])
        target_HelT2 = np.array([-3.167590 * 1.5 + 34, -2.577007 * 1.5 + 34, 3.154192 * 1.5 + 34, 1.591505 * 1.5 + 34])
        target_MGW2 = np.array([-1.881626 * 0.5 + 5.0, -0.126901 * 0.5 + 5.0, 0.958019 * 0.5 + 5.0, 3.951848 * 0.5 + 5.0])
        target_ProT2 = np.array([-1.114200 * 4.0 + -7, -0.827422 * 4.0 + -7, 1.045116 * 4.0 + -7, 2.987222 * 4.0 + -7])
        target_Roll2 = np.array([4.000000 * 1 + -2, -1.552374 * 1 + -2, 3.597537 * 1 + -2, 0.429428 * 1 + -2])

        result1 = self.small_motifs[0].motif
        result2 = self.small_motifs[1].motif
        self.assertTrue(np.all(target_EP1 == result1[0,:]))
        self.assertTrue(np.all(target_HelT1 == result1[1,:]))
        self.assertTrue(np.all(target_MGW1 == result1[2,:]))
        self.assertTrue(np.all(target_ProT1 == result1[3,:]))
        self.assertTrue(np.all(target_Roll1 == result1[4,:]))

        self.assertTrue(np.all(target_EP2 == result2[0,:]))
        self.assertTrue(np.all(target_HelT2 == result2[1,:]))
        self.assertTrue(np.all(target_MGW2 == result2[2,:]))
        self.assertTrue(np.all(target_ProT2 == result2[3,:]))
        self.assertTrue(np.all(target_Roll2 == result2[4,:]))

    def test_standardize(self):
        "Test for correct adjusting of shape values in input fasta file"

        with open("test_data/normed.npy","rb") as f:
            target_normed = np.load(f)

        EP_name = "test_data/small_test_data.fa.EP"
        HelT_name = "test_data/small_test_data.fa.HelT"
        MGW_name = "test_data/small_test_data.fa.MGW"
        ProT_name = "test_data/small_test_data.fa.ProT"
        Roll_name = "test_data/small_test_data.fa.Roll"

        EP_cent = -6.55
        EP_spread = 1.5270780000000004
        HelT_cent = 34.52
        HelT_spread = 1.497425999999997
        MGW_cent = 5.13
        MGW_spread = 0.4892580000000001
        ProT_cent = -7.03
        ProT_spread = 3.8399339999999995
        Roll_cent = -1.85
        Roll_spread = 1.7642939999999998

        result = inout.RecordDatabase(
            shape_dict = {
                "EP": EP_name,
                "HelT": HelT_name,
                "MGW": MGW_name,
                "ProT": ProT_name,
                "Roll": Roll_name,
            },
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
            y = np.zeros(2),
            record_names = ["peak_00001", "peak_00002"],
        )

        transforms = self.motifs.transforms

        result.normalize_shapes_from_values(
            centers = (transforms["EP"][0], transforms["HelT"][0], transformt["MGW"][0], transforms["ProT"][0], transforms["Roll"][0]), 
            spreads = (transforms["EP"][1], transforms["HelT"][1], transformt["MGW"][1], transforms["ProT"][1], transforms["Roll"][1]), 
        )
        arr = result.X

        same = np.all(arr == target_normed)
        self.assertTrue(same)

    def test_sliding_window(self):
        "tests correct shape of sliding_window_view"

        motif = self.motifs[0]
        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/test_data_binary_plus_test_0.fa.EP", "HelT": "test_data/test_data_binary_plus_test_0.fa.HelT", "MGW": "test_data/test_data_binary_plus_test_0.fa.MGW", "ProT": "test_data/test_data_binary_plus_test_0.fa.ProT", "Roll": "test_data/test_data_binary_plus_test_0.fa.Roll"},
            infile = "test_data/test_data_binary_plus_test_0.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        slide_windows = motif.make_sliding_window_view(records.X["peak_00001"])
        # target_shape ends up having 47, 10x5 windows for each of 2 strands
        target_shape = (1,47,1,2,1,10,5,1)
        final_shape = (47,2,10,5)

        self.assertEqual(len(final_shape), len(slide_windows.shape))
        for i,dim_size in enumerate(final_shape):
            self.assertEqual(dim_size, slide_windows.shape[i])

    def test_scan_exception(self):
        "Test whether we identify hits for a single motif in target sequence."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/ragged_small_test_data.fa.EP", "HelT": "test_data/ragged_small_test_data.fa.HelT", "MGW": "test_data/ragged_small_test_data.fa.MGW", "ProT": "test_data/ragged_small_test_data.fa.ProT", "Roll": "test_data/ragged_small_test_data.fa.Roll"},
            infile = "test_data/ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        motif = self.motifs[0]
        with self.assertRaises(Exception) as context:
            motif.scan(records)
        
        self.assertTrue("Record peak_00001 is too short. It is only 5 base pairs long after trimming two basepairs of NA values from each end of the shape values. The motif is 10 long. Exiting without returning motif hits. Consider running again without record peak_00001 in the fasta file." in str(context.exception))

    def test_scan(self):
        "Test whether we identify hits for a single motif in target sequence."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/ragged_small_test_data.fa.EP", "HelT": "test_data/ragged_small_test_data.fa.HelT", "MGW": "test_data/ragged_small_test_data.fa.MGW", "ProT": "test_data/ragged_small_test_data.fa.ProT", "Roll": "test_data/ragged_small_test_data.fa.Roll"},
            infile = "test_data/ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        motif = self.motifs[0]
        motif.scan(records)
 
    def test_identify(self):
        "Test whether we identify motifs at correct positions."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/ragged_small_test_data.fa.EP", "HelT": "test_data/ragged_small_test_data.fa.HelT", "MGW": "test_data/ragged_small_test_data.fa.MGW", "ProT": "test_data/ragged_small_test_data.fa.ProT", "Roll": "test_data/ragged_small_test_data.fa.Roll"},
            infile = "test_data/ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )

        hits = self.motifs.identify(records)
        ###########################################################
        ## note: figure out score and p-value/q-value
        ###########################################################
        truth = [
            ("SHAPE-1", "None", "peak_00001", 1, 4, "+", 10, 0, 0, ""),
            ("SHAPE-2", "None", "peak_00002", 1, 4, "-", 10, 0, 0, ""),
        ]

        self.assertEqual(hits, truth)

    def test_read_dsm(self):
        "Test whether we correctly parse a dsm file"
        read_str = str(self.motifs)
        self.assertEqual(read_str, self.true_str)

    def test_write_dsm(self):
        "Test whether we correctly write a dsm file"
        self.motifs.write_file(self.out_dsm_name)
        written_motifs = inout.Motifs()
        written_motifs.read_file( self.out_dsm_name )
        written_str = str(written_motifs)
        self.assertEqual(written_str, self.true_str)


class TestUtilies(unittest.TestCase):

    def test_parse_cmi_filter_result(self):
        self.assertEqual(1,2)


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

    def test_sample_seqs(self):

        n_samps = 5
        y = self.records.y
        retained_indices,samp_recs,samp_y = self.seqs.sample(
            n_samps,
            y,
            rng_seed=self.seed,
        )
        print(retained_indices)
        self.assertEqual(len(samp_y), len(samp_recs))


class TestRaggedRecordsMethods(unittest.TestCase):

    def setUp(self):
        self.records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/ragged_small_test_data.fa.EP", "HelT": "test_data/ragged_small_test_data.fa.HelT", "MGW": "test_data/ragged_small_test_data.fa.MGW", "ProT": "test_data/ragged_small_test_data.fa.ProT", "Roll": "test_data/ragged_small_test_data.fa.Roll"},
            infile = "test_data/ragged_small_test_data.txt",
        )

    def test_standardize(self):
        "Test for correct adjusting of shape values in input fasta file from ragged record database"

        EP_name = "test_data/ragged_small_test_data.fa.EP"
        HelT_name = "test_data/ragged_small_test_data.fa.HelT"
        MGW_name = "test_data/ragged_small_test_data.fa.MGW"
        ProT_name = "test_data/ragged_small_test_data.fa.ProT"
        Roll_name = "test_data/ragged_small_test_data.fa.Roll"

        EP_cent = -6.55
        EP_spread = 1.5270780000000004
        HelT_cent = 34.52
        HelT_spread = 1.497425999999997
        MGW_cent = 5.13
        MGW_spread = 0.4892580000000001
        ProT_cent = -7.03
        ProT_spread = 3.8399339999999995
        Roll_cent = -1.85
        Roll_spread = 1.7642939999999998

        #with open("test_data/ragged_target_shapes.pkl", "rb") as f:
        #    unnormed_shapes = pickle.load(f)

        #target_shapes = {
        #    "peak_00001": (unnormed_shapes["peak_00001"] - np.array([
        #            # record placeholder
        #            [
        #                # position placeholder
        #                [
        #                    # EP
        #                    [EP_cent],
        #                    # HelT
        #                    [HelT_cent],
        #                    # MGW
        #                    [MGW_cent],
        #                    # ProT
        #                    [ProT_cent],
        #                    # Roll
        #                    [Roll_cent],
        #                ],
        #            ],
        #        ])) / np.array([
        #            # record placeholder
        #            [
        #                # position placeholder
        #                [
        #                    # EP
        #                    [EP_spread],
        #                    # HelT
        #                    [HelT_spread],
        #                    # MGW
        #                    [MGW_spread],
        #                    # ProT
        #                    [ProT_spread],
        #                    # Roll
        #                    [Roll_spread],
        #                ],
        #            ],

        #        ]),
        #    "peak_00002": (unnormed_shapes["peak_00002"] - np.array([
        #            # record placeholder
        #            [
        #                # position placeholder
        #                [
        #                    # EP
        #                    [EP_cent],
        #                    # HelT
        #                    [HelT_cent],
        #                    # MGW
        #                    [MGW_cent],
        #                    # ProT
        #                    [ProT_cent],
        #                    # Roll
        #                    [Roll_cent],
        #                ],
        #            ],
        #        ])) / np.array([
        #            # record placeholder
        #            [
        #                # position placeholder
        #                [
        #                    # EP
        #                    [EP_spread],
        #                    # HelT
        #                    [HelT_spread],
        #                    # MGW
        #                    [MGW_spread],
        #                    # ProT
        #                    [ProT_spread],
        #                    # Roll
        #                    [Roll_spread],
        #                ],
        #            ],
        #        ]),
        #}

        with open("test_data/ragged_normed_target_shapes.pkl", "rb") as f:
            target_shapes = pickle.load(f)

        records = inout.RaggedRecordDatabase(
            shape_dict = {
                "EP": EP_name,
                "HelT": HelT_name,
                "MGW": MGW_name,
                "ProT": ProT_name,
                "Roll": Roll_name,
            },
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
            y = np.zeros(2),
            record_names = ["peak_00001", "peak_00002"],
        )
        records.normalize_shapes_from_values(
            centers = (EP_cent, HelT_cent, MGW_cent, ProT_cent, Roll_cent), 
            spreads = (EP_spread, HelT_spread, MGW_spread, ProT_spread, Roll_spread),
        )

        result = records.X
        for rec_name,rec_target in target_shapes.items():
            same = np.all(rec_target == result[rec_name])
            self.assertTrue(same)

    def test_read_ragged_records(self):
        # should have two keys, "peak_00001" and "peak_00002"
        # each key's value should be array of
        # shape (1,P,S,2), where P is the number of positions (the length of)
        #each record, and S is the number of shape parameters present.
        #The final axis is of length 2, one index for each strand.
        # so dim should be (2,4,5,2)
        target_shapes = {
            "peak_00001": np.array([
                [
                    # pos1
                    [
                        # EP
                        [ -6.81, -6.73 ],
                        # HelT
                        [ 34.56, 33.98 ], 
                        # MGW
                        [ 5.34, 5.74 ],
                        # ProT
                        [ -9.01, -8.6 ],
                        # Roll
                        [ 4.14, -2.69 ],
                    ],

                    # pos2
                    [
                        # EP
                        [ -7.2, -5.43 ],
                        # HelT
                        [ 31.80, 35.73 ],
                        # MGW
                        [ 4.99, 5.27 ],
                        # ProT
                        [ -10.15,  -7.4 ],
                        # Roll
                        [ -4.35, 5.65 ],
                    ],

                    # pos3
                    [
                        # EP
                        [ -6.00, -6.00 ],
                        # HelT
                        [ 33.00, 33.00 ],
                        # MGW
                        [ 4.50, 4.50 ],
                        # ProT
                        [ -8.00 , -8.00 ],
                        # Roll
                        [ 0.00, 0.00 ],
                    ],

                    # pos4
                    [
                        # EP
                        [ -5.43, -7.2 ],
                        # HelT
                        [ 35.73, 31.80 ],
                        # MGW
                        [ 5.27, 4.99 ],
                        # ProT
                        [ -7.4,  -10.15 ],
                        # Roll
                        [ 5.65, -4.35 ],
                    ],

                    # pos5
                    [
                        # EP
                        [ -6.73, -6.81 ],
                        # HelT
                        [ 33.98, 34.56 ], 
                        # MGW
                        [ 5.74, 5.34 ],
                        # ProT
                        [ -8.6,  -9.01 ],
                        # Roll
                        [ -2.69, 4.14 ],
                    ],
                ],
            ]),
            "peak_00002": np.array([
                [
                    # pos1
                    [
                        # EP
                        [ -6.99, -6.0 ],
                        # HelT
                        [ 31.08, 34.84 ],
                        # MGW
                        [ 5.47, 4.18 ],
                        # ProT
                        [ -5.31, -5.67 ],
                        # Roll
                        [ -2.96, -3.71 ],
                    ],

                    # pos2
                    [
                        # EP
                        [ -5.23, -7.65 ],
                        # HelT
                        [ 34.22, 35.34 ],
                        # MGW
                        [ 5.67, 4.94 ],
                        # ProT
                        [ -6.43, -7.96 ],
                        # Roll
                        [ 4.10, -1.65 ],
                    ],

                    # pos3
                    [
                        # EP
                        [ -7.65, -5.23 ],
                        # HelT
                        [ 35.34, 34.22 ],
                        # MGW
                        [ 4.94, 5.67 ],
                        # ProT
                        [ -7.96, -6.43 ],
                        # Roll
                        [ -1.65, 4.10 ],
                    ],

                    # pos4
                    [
                        # EP
                        [ -6.0, -6.99 ],
                        # HelT
                        [ 34.84, 31.08 ],
                        # MGW
                        [ 4.18, 5.47 ],
                        # ProT
                        [ -5.67, -5.31 ],
                        # Roll
                        [ -3.71, -2.96 ],
                    ],
                ],
            ]),
        }

        result = self.records.X
        for rec_name,rec_target in target_shapes.items():
            same = np.all(rec_target == result[rec_name])
            self.assertTrue(same)


if __name__ == "__main__":
    unittest.main()

