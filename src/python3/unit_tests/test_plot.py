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
import ShapeME as shapeme

out_dir = "test_data/performance_data/main_output"
fold_direcs = [
    "test_data/performance_data/fold_0_output",
    "test_data/performance_data/fold_1_output",
    "test_data/performance_data/fold_2_output",
    "test_data/performance_data/fold_3_output",
    "test_data/performance_data/fold_4_output",
]
binary_out_dir = "test_data/performance_data/binary_main_output"
binary_fold_direcs = [
    "test_data/performance_data/binary_fold_0_output",
    "test_data/performance_data/binary_fold_1_output",
    "test_data/performance_data/binary_fold_2_output",
    "test_data/performance_data/binary_fold_3_output",
    "test_data/performance_data/binary_fold_4_output",
]

performance = shapeme.Performance(out_dir, fold_direcs)
performance.plot_performance("test_data/performance_data/results/test.png")

performance = shapeme.Performance(binary_out_dir, binary_fold_direcs)
performance.plot_performance("test_data/performance_data/results/binary_test.png")


