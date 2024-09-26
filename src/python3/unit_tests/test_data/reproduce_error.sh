#!/bin/bash

python /corexfs/schroedj/src/ShapeME/src/python3/evaluate_motifs.py --test_shape_files /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa.EP /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa.HelT /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa.MGW /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa.ProT /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa.Roll  --shape_names EP HelT MGW ProT Roll --data_dir /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/ --test_score_file /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp06b2zk0u_fold_2_test.txt \
    --out_dir test_reproduce \
    --nprocs 8 \
    --out_prefix shape_and_seq --test_seq_fasta /corexfs/schroedj/src/ShapeME/src/python3/unit_tests/test_data/SP2_0.95/tmp906dqji8fold_2_test.fa
