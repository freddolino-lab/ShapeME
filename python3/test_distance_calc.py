#!/usr/bin/env python3

import unittest
import inout
import find_motifs as fm
import numpy as np

"""
Unit tests for distance calculations
"""

class Test(unittest.TestCase):

    def setUp(self):
        self.ref_plus = np.zeros((2,5))
        self.ref_minus = np.ones((2,5))
        self.ref_3d = np.stack([self.ref_plus, self.ref_minus], axis=-1)
        self.query_val = 0.5
        print(self.ref_3d.shape)
        self.alpha = 0.1

        # prior to saving, I transposed axes 1 and 2, so I'll reverse that
        X_transposed = np.load('../test_data/shapes.npy')
        X = X_transposed.transpose((0,2,1,3))
        # one record, two shapes, three vals, two strands
        # X shape will be (1,3,2,2)
        X = X[0:1,:3,:2,:]
        y = np.load('../test_data/subset_y_vals.npy')
        self.records = inout.RecordDatabase(
            X = X,
            # subset data has two shapes. these may not be their
            # actual names, but it doesn't matter for these tests
            shape_names = ["EP", "Roll"],
            y = y,
        )
        # we'll just have one window at this window size of 3
        self.records.compute_windows(wsize = 3)
        self.rec_answer = np.array([0.482780318, 0.79238146])
        self.match_threshold = 0.8711171869882366
        
    def test_stranded_dists(self):
        trans_weights = np.full((2,5,1), 0.1)
        total = trans_weights.sum()
        target_weights = trans_weights / total
        answer = np.sum((np.abs(self.ref_3d - self.query_val) * target_weights), axis=(0,1))

        weights = np.log(
            (trans_weights - self.alpha)
            / (1.0 - self.alpha) / (1.0 - (trans_weights - self.alpha) / (1.0 - self.alpha))
        )


        query = np.full((2,5,1), self.query_val)
        dists = inout.constrained_inv_logit_manhattan_distance(
            self.ref_3d,
            query,
            weights,
            self.alpha,
        )
        print(dists)
        print(answer)
        results = []
        for i in range(dists.size):
            print(dists[i])
            print(answer[i])
            self.assertAlmostEqual(dists[i], answer[i])

    def test_record_db_stranded_dists(self):
        trans_weights = np.full((3,2,1), 0.1)
        total = trans_weights.sum()
        target_weights = trans_weights / total
        weights = np.log(
            (target_weights - self.alpha)
            / (1.0 - self.alpha) / (1.0 - (target_weights - self.alpha) / (1.0 - self.alpha))
        )

        query = np.full((3,2,1), self.query_val)
        hits = np.zeros((1,2), dtype='int64')
        distances = np.zeros((self.records.windows.shape[3], 2))
        lt = np.zeros((self.records.windows.shape[3], 2), dtype='bool')
        inout.optim_generate_peak_array(
            ref = self.records.windows,
            query = query,
            weights = weights,
            threshold = self.match_threshold,
            results = hits,
            R = 1,
            W = self.records.windows.shape[3],
            dist = inout.constrained_inv_logit_manhattan_distance,
            max_count = 1,
            alpha = self.alpha,
            dists = distances,
            lt = lt,
        )
        print(distances)
        print(self.rec_answer)
        for i in range(distances.size):
            self.assertAlmostEqual(distances[0,i], self.rec_answer[i])


if __name__ == '__main__':
    unittest.main()

