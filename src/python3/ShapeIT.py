#!/usr/bin/env python3

"""
The main driver script for identifying instances of the motifs in the meme or dsm file provided.
"""

import subprocess
import argparse
import os
import sys
import numpy as np
from pathlib import Path
import logging
import shlex
import shutil
import json
from jinja2 import Environment,FileSystemLoader
import seaborn as sns
import pandas as pd
import base64
from matplotlib import pyplot as plt
import pickle
import tempfile

this_path = Path(__file__).parent.absolute()
sys.path.insert(0, this_path)

import evaluate_motifs as evm
import inout as io

from convert_narrowpeak_to_fire import make_kfold_datasets
import inout

jinja_env = Environment(loader=FileSystemLoader(os.path.join(this_path, "templates/")))

def write_report(environ, temp_base, info, out_name):
    print("writing report")
    print(f"base template: {temp_base}")
    template = environ.get_template(temp_base)
    print(f"out_name: {out_name}")
    content = template.render(**info)
    with open(out_name, "w", encoding="utf-8") as report:
        report.write(content)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seq_fasta", type=str, default=None,
        help=f"Fasta file in which to search for motifs")
    parser.add_argument("--motifs_file", type=str, default=None,
        help=f"Meme or dsm file with motif definitions. The instances of the motifs in this file "\
        "present in --seq_fasta will be returned in a format similar to fimo output.")
    parser.add_argument("--log_level", type=str, default="INFO",
        help=f"Sets log level for logging module. Valid values are DEBUG, "\
                f"INFO, WARNING, ERROR, CRITICAL.")

    args = parser.parse_args()
    return args

def parse_fasta(fasta_file):
    with open(fasta_file, 'r') as f:
        sequences = []
        sequence_id = None
        sequence = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id is not None:
                    sequences.append((sequence_id, sequence))
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        if sequence_id is not None:
            sequences.append((sequence_id, sequence))
    return sequences
    
def identify(args):

    seq_fasta = args.seq_fasta
    motifs_file = args.motifs_file

    motifs = inout.Motifs()
    motifs.read_file( motifs_file )
    seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()

    # if there are shape motifs in the motifs file, find them
    if shape_motifs:

        seqs = parse_fasta(seq_fasta)

        shape_names = ["EP", "HelT", "MGW", "ProT", "Roll"]
        full_shape_fnames = ""
        for shape_name in shape_names:
            full_shape_fnames += f"{seq_fasta}.{shape_name} "

        convert = f"Rscript {this_path}/utils/calc_shape.R {seq_fasta}"
        convert_result = subprocess.run(
            convert,
            shell=True,
            capture_output=True,
            #check=True,
        )
        
        shape_fname_dict = {
            n:fname for n,fname
            in zip(shape_names, full_shape_fnames.split(" "))
        }
        records = io.RecordDatabase(
            shape_dict = shape_fname_dict,
            shift_params = ["Roll", "HelT"],
            exclude_na = True,
            y = np.zeros(len(seqs)),
            record_names = [_[0] for _ in seqs],
        )
        records.determine_center_spread()
        records.normalize_shape_values()
        print(records)

        ##################################################################################
        ##################################################################################
        ##################################################################################
        ## I need to get new way to build recorddb in motifer and motifs in motifer for this use
        ##################################################################################
        ##################################################################################
        ##################################################################################
        with open(shape_fname, 'wb') as shape_f:
            np.save(shape_fname, records.X.transpose((0,2,1,3)))
        FIND_CMD = f"{identify_bin} {shape_fname} {shape_motif_fname}"
        result = subprocess.run(FIND_CMD, shell=True, env=my_env, capture_output=True)

    # if there are sequence motifs in the file, just run fimo and collect results
    if seq_motifs:

        with tempfile.TemporaryDirectory() as td:
            evm.run_fimo(
                seq_motifs,
                seq_fasta,
                "seq_motifs.meme",
                td,
                this_path,
                recs=None,
            )
            with open(f"{td}/fimo.tsv", "r") as ff:
                header = ff.readline().strip()
                fimo_res = []
                for line in ff:
                    if not line.startswith("#"):
                        if not line == "\n":
                            fimo_res.append(line.strip())
        print(fimo_res)
        


def main():

    args = parse_args()
    identify(args)


if __name__ == '__main__':
    main()

