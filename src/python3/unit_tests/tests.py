import unittest
import sys
import os
import numpy as np
from pathlib import Path
import pickle
import tempfile

this_path = os.path.join(Path(__file__).parent.absolute(), "../")
sys.path.insert(0, this_path)

import inout
import evaluate_motifs as evm
import ShapeIT as shapeit

class TestFastaMethods(unittest.TestCase):

    def setUp(self):
        self.vals_in_files = {
            "EP": {
                "peak_00001": np.array([ np.NaN,np.NaN,-6.81,-7.20,-5.43,-6.73,np.NaN,np.NaN ]),
                "peak_00002": np.array([ np.NaN,np.NaN,-6.99,-5.23,-7.65,-6.00,np.NaN,np.NaN ]),
            },
            "HelT": {
                "peak_00001": np.array([ np.NaN,33.01,34.56,31.80,35.73,33.98,np.NaN ]),
                "peak_00002": np.array([ np.NaN,35.38,31.08,34.22,35.34,34.84,np.NaN ]),
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
                "peak_00001": np.array([ np.NaN,2.56,4.14,-4.35,5.65,-2.69,np.NaN ]),
                "peak_00002": np.array([ np.NaN,-2.31,-2.96,4.10,-1.65,-3.71,np.NaN ]),
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
                    -6.81, #, -6.73 ],
                    # HelT
                    34.56, #, 33.98 ], 
                    # MGW
                    5.34, #, 5.74 ],
                    # ProT
                    -9.01, #, -8.6 ],
                    # Roll
                    4.14, #, -2.69 ],
                ],

                # pos2
                [
                    # EP
                    -7.2, #, -5.43 ],
                    # HelT
                    31.80,  #, 35.73 ],
                    # MGW
                    4.99,  #, 5.27 ],
                    # ProT
                    -10.15, # -7.4 ],
                    # Roll
                    -4.35, # 5.65 ],
                ],

                # pos3
                [
                    # EP
                    -5.43,# -7.2 ],
                    # HelT#
                    35.73,# 31.80 ],
                    # MGW
                    5.27, #4.99 ],
                    # ProT#
                    -7.4, # -10.15 ],
                    # Roll#
                    5.65, #-4.35 ],
                ],

                # pos4
                [
                    # EP
                    -6.73,# -6.81 ],
                    # HelT#
                    33.98,# 34.56 ], 
                    # MGW
                    5.74, #5.34 ],
                    # ProT#
                    -8.6, # -9.01 ],
                    # Roll#
                    -2.69,# 4.14 ],
                ],
            ],

            # peak_0002
            [
                # pos1
                [
                    # EP
                    -6.99,# -6.0 ],
                    # HelT#
                    31.08,# 34.84 ],
                    # MGW
                    5.47, #4.18 ],
                    # ProT#
                    -5.31,# -5.67 ],
                    # Roll#
                    -2.96,# -3.71 ],
                ],

                # pos2
                [
                    # EP
                    -5.23,# -7.65 ],
                    # HelT#
                    34.22,# 35.34 ],
                    # MGW
                    5.67, #4.94 ],
                    # ProT#
                    -6.43,# -7.96 ],
                    # Roll#
                    4.10, #-1.65 ],
                ],

                # pos3
                [
                    # EP
                    -7.65,# -5.23 ],
                    # HelT#
                    35.34,# 34.22 ],
                    # MGW
                    4.94, #5.67 ],
                    # ProT#
                    -7.96,# -6.43 ],
                    # Roll#
                    -1.65,# 4.10 ],
                ],

                # pos4
                [
                    # EP
                    -6.0, #-6.99 ],
                    # HelT#
                    34.84,# 31.08 ],
                    # MGW
                    4.18, #5.47 ],
                    # ProT#
                    -5.67,# -5.31 ],
                    # Roll#
                    -3.71,# -2.96 ],
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

        self.seq_motif_fname = "test_data/cmi_test_data/streme.txt"
        self.seq_motifs = inout.Motifs()
        self.seq_motifs.read_file( self.seq_motif_fname )

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

    def test_set_motifs_X(self):
        self.assertTrue(False)

    def test_set_motif_X(self):
        self.assertTrue(False)

    def test_seq_seq_X(self):
        self.assertTrue(False)
        
    def test_read_rust_motifs(self):
        true_vals = np.array([1.2773111194451234,1.1180652261474313,1.013006059147333,1.127655239412062,0.7629026691324009,0.5432454456239426,2.7421093793782325,2.3773687694067434,-0.04295526996191982,-3.6919303189407024,-3.991392205961111,-1.2535928653464052,-3.8058503813818367,0.8127795518431163,-1.3426905642543492,-3.0980229039839857,4.0,-0.6709187768810462,0.3969566117871946,-0.7994539080803214,0.29510191226363697,2.5814353961228633,0.4338070325970177,-0.8136997151222107,0.48106490558119164,3.9927933088697922,-4.0,3.534429584186872,-2.0853180559715536,0.9850228290309825,-0.06932976698993176,-0.3889939363013855,1.1504453667879029,2.5416541250127156,2.5843873092003466,2.302556570897386,-1.2363800964478013,-2.4080015092947953,0.7053859201593731,1.6678869177429936,3.6921541524982255,3.2519660845498506,0.2932956922574255,-0.03809626705635591,-1.314249778964454,-3.17141849006105,2.9509085852514594,2.1989834453740134,-2.7111076702309616,-0.212721127225040]).reshape((5,10))
        true_weights = np.array([0.568892290274406,-0.6450136343239903,-0.1485513179173908,1.372188051910688,3.5597628179658294,1.3863169880184867,-4.0,3.648194646157208,2.5925808343483814,-0.29714313143762816,2.8419990700446354,2.741767710319133,-1.5375597783069763,-0.47369969290996267,-1.7373196884885127,0.756306562447081,1.2229193121942432,1.8768937696479544,1.3592332264090472,1.6692054153646239,2.034239557780019,-1.8987438576512736,2.509889226365657,0.24827958381017223,-3.5762741698439995,-0.8403198108936383,-0.08632715630927543,2.094902954017418,0.06815102132571105,2.750133328839885,1.8982379775737106,-3.262044344554027,3.103570681573637,3.9996518627128106,0.35029941675290355,3.0087437265604007,1.4459759432310253,0.7527730400746326,0.5818654503286491,1.9368563882994678,-1.5747948274141366,0.2903474205465196,2.8699768238544783,-0.23939445021728434,0.1969904902376627,-0.4785739192060973,1.0978233825402774,-2.799269975957564,3.379097859539415,-2.2635288306583705]).reshape((5,10))
        true_weights_norm = np.array([0.020657516818806867,0.011281387268454407,0.015065562409107319,0.025728698335062427,0.03128977646439759,0.025801001367132547,0.0008945491434938737,0.03136243935638456,0.029953437020393384,0.013897523626763152,0.03041591795377302,0.03024212147418164,0.005955481108042284,0.01254346239291174,0.005088023747031303,0.021996567896732443,0.024927488020495464,0.027942854764802506,0.025661866842752342,0.027121658358288277,0.02848720055684003,0.004470058324439707,0.02977666325254836,0.01821296362335116,0.0011885783567601573,0.009923135246383588,0.015559262556165902,0.028680249414877325,0.016788618944003447,0.030257222521549786,0.028020503793551577,0.0014968223023251513,0.03080244233221538,0.03159764267059777,0.019007187292250666,0.030672746544710135,0.026099577052061,0.021972086173414822,0.02075271326653299,0.028157913285255354,0.00578488634128104,0.018541905639258845,0.03046173407932766,0.014349128314324465,0.017809627577808698,0.01250677171806386,0.024203725335325003,0.002148679820787012,0.03112109369988481,0.0033215255950973695]).reshape((5,10))
        true_thresh = 1.458039000252392
        true_hits = np.array([0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1]).reshape((10,2))
        true_mi = 0.2037881325278831
        true_dists = np.array([1.4969048233071958,1.536859755213558,1.5188844857687451,1.4720898145522476,1.65134136782672,1.5075606196878724,1.4320262930144352,1.4078493719975256,1.6424845740539167,1.6121954673591696,1.5844323612089593,1.6670871890997454,1.6035563933845496,1.5545555662091393,1.4638619560195238,1.5018461563513168,1.4675240277673187,1.5786038988662603,1.429020085305714,1.4143631960768597]).reshape((10,2))
        true_positions = [[[],[]],[[],[]],[[],[]],[[4],[16]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[15],[13]]]
        true_zscore = 281.8492936200628
        true_robustness = [10,10]

        motifs = inout.read_shape_motifs(
            "test_data/test_rust_results.json",
            {"EP":0, "HelT":1, "MGW":2, "ProT":3, "Roll":4},
        )

        motif = motifs[0]

        self.assertTrue(np.allclose(true_vals, motif.motif))
        self.assertTrue(np.allclose(true_weights_norm, motif.weights))
        self.assertTrue(np.all(true_hits == motif.hits))
        self.assertEqual(true_mi, motif.mi)
        self.assertEqual(true_thresh, motif.threshold)
                

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
            centers = (transforms["EP"][0], transforms["HelT"][0], transforms["MGW"][0], transforms["ProT"][0], transforms["Roll"][0]), 
            spreads = (transforms["EP"][1], transforms["HelT"][1], transforms["MGW"][1], transforms["ProT"][1], transforms["Roll"][1]), 
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
        rec_name = "peak_00001"
        slide_windows = motif.make_sliding_window_view(records.X[rec_name], rec_name)
        # target_shape ends up having 47, 10x5 windows for each of 2 strands
        target_shape = (47,10,5)
        final_shape = (47,10,5)

        self.assertEqual(len(final_shape), len(slide_windows.shape))
        for i,dim_size in enumerate(final_shape):
            self.assertEqual(dim_size, slide_windows.shape[i])

    def test_scan_length_exception(self):
        "Test whether we identify hits for a single motif in target sequence."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/ragged_small_test_data.fa.EP", "HelT": "test_data/ragged_small_test_data.fa.HelT", "MGW": "test_data/ragged_small_test_data.fa.MGW", "ProT": "test_data/ragged_small_test_data.fa.ProT", "Roll": "test_data/ragged_small_test_data.fa.Roll"},
            infile = "test_data/ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        transforms = self.motifs.transforms
        records.normalize_shapes_from_values(
            centers = (transforms["EP"][0], transforms["HelT"][0], transforms["MGW"][0], transforms["ProT"][0], transforms["Roll"][0]), 
            spreads = (transforms["EP"][1], transforms["HelT"][1], transforms["MGW"][1], transforms["ProT"][1], transforms["Roll"][1]), 
        )

        motif = self.motifs[0]
        with self.assertRaises(Exception) as context:
            motif.scan(records)
        
        self.assertEqual("Record peak_00001 is too short. It is only 5 base pairs long after trimming two basepairs of NA values from each end of the shape values. The motif is 10 long. Exiting without returning motif hits. Consider running again without record peak_00001 in the fasta file.", str(context.exception))

    def test_scan_unnormed_exception(self):
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
        
        self.assertTrue("RaggedRecordsDatabase reports that its shape values are not standardized." == str(context.exception))

    def test_scan(self):
        "Test whether we format hits correctly for a motif."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/scan_target_ragged_small_test_data.fa.EP", "HelT": "test_data/scan_target_ragged_small_test_data.fa.HelT", "MGW": "test_data/scan_target_ragged_small_test_data.fa.MGW", "ProT": "test_data/scan_target_ragged_small_test_data.fa.ProT", "Roll": "test_data/scan_target_ragged_small_test_data.fa.Roll"},
            infile = "test_data/scan_target_ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        transforms = self.small_motifs.transforms
        #print(f"records X shape: {[_.shape for _ in records.X.values()]}")
        records.normalize_shapes_from_values(
            centers = (transforms["EP"][0], transforms["HelT"][0], transforms["MGW"][0], transforms["ProT"][0], transforms["Roll"][0]), 
            spreads = (transforms["EP"][1], transforms["HelT"][1], transforms["MGW"][1], transforms["ProT"][1], transforms["Roll"][1]), 
        )

        motif = self.small_motifs[0]
        #print(f"motif width: {len(motif)}")
        motif.threshold = 0.01
        hits = motif.scan(records)
        motif = self.small_motifs[1]
        hits.extend(motif.scan(records))

        target_hits = [
            ("SHAPE-1", "None", "peak_00001", 4, 7, "+", 0, 1, 1, ""),
            ("SHAPE-2", "None", "peak_00003", 7, 10, "-", 0, 1, 1, ""),
        ]

        self.assertEqual(len(target_hits), len(hits))
        for i,hit in enumerate(target_hits):
            for j,elem in enumerate(hit):
                self.assertEqual(elem, hits[i][j])

    def test_read_streme_meme_file(self):
        true_str = "ALPHABET= ACGT\nSHAPES= \nMOTIF 1-AAATATGAAGA STREME-1\nletter-probability matrix: alength= 4 w= 11 nsites= 65 E= 0.17\n\nMOTIF 2-AATAAAAGTTRA STREME-2\nletter-probability matrix: alength= 4 w= 12 nsites= 46 E= 2.3\n\nMOTIF 3-AATATTATAKWGA STREME-3\nletter-probability matrix: alength= 4 w= 13 nsites= 57 E= 3.3\n\nMOTIF 4-AMAAACWTWWYA STREME-4\nletter-probability matrix: alength= 4 w= 12 nsites= 39 E= 4.0\n\n"
        seq_motifs = inout.Motifs()
        seq_motifs.read_file(self.seq_motif_fname)
        self.assertEqual(str(seq_motifs), true_str)
        self.assertEqual(len(seq_motifs.transforms), 0)

    def test_parse_transforms_line(self):
        transforms_line = "EP:-6.0,1.0 HelT:34.0,1.5 MGW:5.0,0.5 ProT:-7.0,4.0 Roll:-2.0,1.0\n"
        target = {"EP": (-6.0,1.0), "HelT":(34.0,1.5), "MGW":(5.0,0.5), "ProT": (-7.0,4.0), "Roll": (-2.0,1.0)}
        result = inout.parse_transforms_line(transforms_line)
        self.assertEqual(target, result)

        bad_target = {"P": (-6.0,1.0), "HelT":(34.0,1.5), "MGW":(5.0,0.5), "ProT": (-7.0,4.0), "Roll": (-2.0,1.0)}
        self.assertNotEqual(bad_target, result)

    def test_get_transforms_from_file(self):

        shape_transforms = self.small_motifs.transforms
        seq_transforms = self.seq_motifs.transforms

        self.assertTrue(len(shape_transforms) == 5)
        self.assertTrue(len(seq_transforms) == 0)

        target = {"EP": (-6.0,1.0), "HelT":(34.0,1.5), "MGW":(5.0,0.5), "ProT": (-7.0,4.0), "Roll": (-2.0,1.0)}
        result = inout.parse_transforms_line(transforms_line)
        self.assertEqual(target, result)

        bad_target = {"P": (-6.0,1.0), "HelT":(34.0,1.5), "MGW":(5.0,0.5), "ProT": (-7.0,4.0), "Roll": (-2.0,1.0)}
        self.assertNotEqual(bad_target, result)



    def test_get_fimo_hits(self):

        seq_motifs = inout.Motifs()
        seq_motifs.read_file("test_data/small_seq_motifs.meme")
        #print(seq_motifs)
        fimo_res = shapeit.get_fimo_results(
            seq_motifs,
            "test_data/small_seq_motif_test_fimo.fa",
            "../",
            thresh = 0.005,
            testing = True,
        )
        header = "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n"
        hit1 = "2-AATAA\tSTREME-2\tseq2\t4\t8\t+\t9.24107\t0.00156\t0.0125\tAATAA\n"
        hit2 = "2-AATAA\tSTREME-2\tseq4\t2\t6\t-\t9.24107\t0.00156\t0.0125\tAATAA\n"
        hit3 = "1-AAATA\tSTREME-1\tseq1\t2\t6\t+\t9.24107\t0.00156\t0.0125\tAAATA\n"
        hit4 = "1-AAATA\tSTREME-1\tseq3\t2\t6\t-\t9.24107\t0.00156\t0.0125\tAAATA"
        #truth = header + 
        truth = hit1 + hit2 + hit3 + hit4
        self.assertEqual(fimo_res, truth)

    def test_print_hits(self):
        hits = [
            ("SHAPE-1", "None", "peak_00001", 4, 7, "+", 0, 1, 1, ""),
            ("SHAPE-2", "None", "peak_00003", 7, 10, "-", 0, 1, 1, ""),
        ]
        output = "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\nSHAPE-1\tNone\tpeak_00001\t4\t7\t+\t0\t1\t1\t\nSHAPE-2\tNone\tpeak_00003\t7\t10\t-\t0\t1\t1\t\n"
        self.assertTrue(False)

    def test_identify(self):
        "Test whether we identify motifs at correct positions."

        records = inout.RaggedRecordDatabase(
            shape_dict = {"EP": "test_data/scan_target_ragged_small_test_data.fa.EP", "HelT": "test_data/scan_target_ragged_small_test_data.fa.HelT", "MGW": "test_data/scan_target_ragged_small_test_data.fa.MGW", "ProT": "test_data/scan_target_ragged_small_test_data.fa.ProT", "Roll": "test_data/scan_target_ragged_small_test_data.fa.Roll"},
            infile = "test_data/scan_target_ragged_small_test_data.txt",
            shift_params = ["HelT", "Roll"],
            exclude_na = True,
        )
        
        transforms = self.small_motifs.transforms
        records.normalize_shapes_from_values(
            centers = (transforms["EP"][0], transforms["HelT"][0], transforms["MGW"][0], transforms["ProT"][0], transforms["Roll"][0]), 
            spreads = (transforms["EP"][1], transforms["HelT"][1], transforms["MGW"][1], transforms["ProT"][1], transforms["Roll"][1]), 
        )
 
        hits = self.small_motifs.identify(records)
        ###########################################################
        ## note: figure out score and p-value/q-value
        ###########################################################
        # output should match fimo.tsv columns:
        # id alt_id seq_name sequence_name start end strand score p-value q-value matched_sequence
        truth = "SHAPE-1\tNone\tpeak_00001\t4\t7\t+\t0\t1\t1\t\nSHAPE-2\tNone\tpeak_00003\t7\t10\t-\t0\t1\t1\t\n"
        self.assertEqual(hits, truth)

        truth_no_hits = ""
        records.normalize_shapes_from_values(centers = (1000,1000,1000,1000,1000), spreads = (5,5,5,5,5))
        no_hits = self.small_motifs.identify(records)
        self.assertEqual(no_hits, truth_no_hits)

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

    def test_parse_robustness_line(self):
        line="adj_mi= -0.00000000000000011990391686347508, robustness= (0, 10), zscore= inf\n"
        mi,rob,z = inout.parse_robustness_output(line)
        self.assertEqual(mi, -0.00000000000000011990391686347508)
        self.assertEqual(rob[0], 0)
        self.assertEqual(rob[1], 10)
        self.assertTrue(np.isinf(z))

        line="adj_mi= 1.0, robustness= (10, 10), zscore= 200.35\n"
        mi,rob,z = inout.parse_robustness_output(line)
        self.assertEqual(mi, 1.0)
        self.assertEqual(rob[0], 10)
        self.assertEqual(rob[1], 10)
        self.assertEqual(z, 200.35)


class TestUtilities(unittest.TestCase):

    def test_parse_transforms_line(self):
        line = "EP:-6.68,1.586381999999999 HelT:34.54,1.4677740000000028 MGW:5.15,0.5040839999999998 ProT:-7.55,3.4989360000000005 Roll:-1.86,1.8384239999999998\n"
        truth = {
            "EP":(-6.68,1.586381999999999),
            "HelT":(34.54,1.4677740000000028),
            "MGW":(5.15,0.5040839999999998),
            "ProT":(-7.55,3.4989360000000005),
            "Roll":(-1.86,1.8384239999999998),
        }
        result = inout.parse_transforms_line(line)
        for k,v in truth.items():
            r_v = result[k]
            self.assertEqual(v[0], r_v[0])
            self.assertEqual(v[1], r_v[1])

    def test_parse_cmi_filter_result(self):
        self.assertTrue(False)


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
        # shape (P,S), where P is the number of positions (the length of)
        #each record, and S is the number of shape parameters present.
        # so dim should be (4,5)
        target_shapes = {
            "peak_00001": np.array([
                    # pos1
                    [
                        # EP
                        -6.81,# -6.73 ],
                        # HelT#
                        34.56,# 33.98 ], 
                        # MGW
                        5.34, #5.74 ],
                        # ProT#
                        -9.01,# -8.6 ],
                        # Roll#
                        4.14, #-2.69 ],
                    ],

                    # pos2
                    [
                        # EP
                        -7.2, #-5.43 ],
                        # HelT#
                        31.80,# 35.73 ],
                        # MGW
                        4.99, #5.27 ],
                        # ProT#
                        -10.15,#  -7.4 ],
                        # Roll#
                        -4.35,# 5.65 ],
                    ],

                    # pos3
                    [
                        # EP
                        -6.00,# -6.00 ],
                        # HelT#
                        33.00,# 33.00 ],
                        # MGW
                        4.50, #4.50 ],
                        # ProT#
                        -8.00, #, -8.00 ],
                        # Roll#
                        0.00, #0.00 ],
                    ],

                    # pos4
                    [
                        # EP
                        -5.43,# -7.2 ],
                        # HelT#
                        35.73,# 31.80 ],
                        # MGW
                        5.27, #4.99 ],
                        # ProT#
                        -7.4, # -10.15 ],
                        # Roll#
                        5.65, #-4.35 ],
                    ],

                    # pos5
                    [
                        # EP
                        -6.73,# -6.81 ],
                        # HelT#
                        33.98,# 34.56 ], 
                        # MGW
                        5.74, #5.34 ],
                        # ProT#
                        -8.6, # -9.01 ],
                        # Roll#
                        -2.69,# 4.14 ],
                    ],
            ]),
            "peak_00002": np.array([
                    # pos1
                    [
                        # EP
                        -6.99,# -6.0 ],
                        # HelT
                        31.08,# 34.84 ],
                        # MGW
                        5.47, #4.18 ],
                        # ProT
                        -5.31,# -5.67 ],
                        # Roll
                        -2.96,# -3.71 ],
                    ],

                    # pos2
                    [
                        # EP
                        -5.23,# -7.65 ],
                        # HelT#
                        34.22,# 35.34 ],
                        # MGW
                        5.67, #4.94 ],
                        # ProT#
                        -6.43,# -7.96 ],
                        # Roll#
                        4.10, #-1.65 ],
                    ],

                    # pos3
                    [
                        # EP
                        -7.65,# -5.23 ],
                        # HelT
                        35.34,# 34.22 ],
                        # MGW
                        4.94, #5.67 ],
                        # ProT
                        -7.96,# -6.43 ],
                        # Roll
                        -1.65,# 4.10 ],
                    ],

                    # pos4
                    [
                        # EP
                        -6.0, #-6.99 ],
                        # HelT#
                        34.84,# 31.08 ],
                        # MGW
                        4.18, #5.47 ],
                        # ProT#
                        -5.67,# -5.31 ],
                        # Roll#
                        -3.71,# -2.96 ],
                    ],
            ]),
        }

        result = self.records.X
        for rec_name,rec_target in target_shapes.items():
            same = np.all(rec_target == result[rec_name])
            self.assertTrue(same)


if __name__ == "__main__":
    unittest.main()

