import sys
import os
import argparse
import numpy as np
import logging
import random
import pathlib
import pickle

this_path = pathlib.Path(__file__).parent.absolute()
utils_path = os.path.join(this_path, 'utils')
sys.path.insert(0, utils_path)

import fasta as fa
import peak as pk

BASES = ['A','C','T','G']

class FIREfile(object):

    def __init__(self):
        self.data = {}
        self.names = []

    def __iter__(self):
        for name in self.names:
            yield (name, self.data[name])

    def __len__(self):
        return len(self.names)

    def __add__(self, other):
        newfile = FIREfile()
        for name, score in self:
            newfile.add_entry(name, score)
        for name, score in other:
            newfile.add_entry(name, score)
        return newfile

    def add_entry(self, name, score):
        self.data[name] = score
        self.names.append(name)

    def pull_value(self, name):
        return self.data[name]

    def discretize_quant(self, nbins=10):
        # first pull all the values
        all_vals = [val for name, val in self]
        all_vals = np.array(all_vals)
        quants = np.arange(0,100, 100.0/nbins)
        bins = []
        for quant in quants:
            bins.append(np.percentile(all_vals, quant))
        all_vals = np.digitize(all_vals, bins)
        for new_val, (name, old_val) in zip(all_vals, self):
            self.data[name] = new_val

    def shuffle(self):
        size = len(self)
        subset = np.random.permutation(size)
        shuffled_names = [self.names[val] for val in subset]
        self.names = shuffled_names

    def write(self, fname):
        with open(fname, mode="w") as outf:
            outf.write("name\tscore\n")
            for name in self.names:
                outf.write("%s\t%s\n"%(name, self.data[name]))


def make_kfold_datasets(k, fastafile, firefile, outpre):
    logging.warning("Making %s fold datasets"%k)
    # shuffle all the fasta entries in place 
    np.random.shuffle(fastafile.names)
    # create k folds out of the names as a list of lists
    size = len(fastafile.names)
    folds = [firefile.names[i::k] for i in range(k)]
    # loop through each fold
    for test_idx in range(len(folds)):
        # make a seperate fasta file and fire file for the test fold
        this_test_fasta = fa.FastaFile()
        this_test_fire = FIREfile()
        for name in folds[test_idx]:
            this_test_fasta.add_entry(fastafile.pull_entry(name))
            this_test_fire.add_entry(name, firefile.pull_value(name))
        with open(outpre+"_test_%i.fa"%test_idx, mode="w") as outf:
            this_test_fasta.write(outf)
        this_test_fire.write(outpre+"_test_%i.txt"%test_idx)
        
        # make a sperate fasta and fire file for the train folds
        this_train_fasta = fa.FastaFile()
        this_train_fire = FIREfile()
        
        for train_idx in range(len(folds)):
            if train_idx != test_idx:
                for name in folds[train_idx]:
                    this_train_fasta.add_entry(fastafile.pull_entry(name))
                    this_train_fire.add_entry(name, firefile.pull_value(name))

        logging.warning("Writing fold %i"%test_idx)
        with open(outpre+"_train_%i.fa"%test_idx, mode="w") as outf:
            this_train_fasta.write(outf)
        this_train_fire.write(outpre+"_train_%i.txt"%test_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take a narrowPeak file and pull seqs for FIRE format")
    parser.add_argument('npfile', type=str, help="input narrowPeak file")
    parser.add_argument('fasta', type=str, help="input fasta file")
    parser.add_argument('outpre', type=str, help="output file prefix")
    parser.add_argument('--wsize', type=int, default=60,
                        help="total window size around peak center")
    parser.add_argument('--nrand', type=int, default=3, 
            help="multiplier for number of random seqs to include")
    parser.add_argument('--max_peaks', type=int, default=0, 
            help="maximum number of peaks to include. Default is 0, which indicates no limit. If set to a different value, peaks will be sorted in order of descending signal and truncated at the selected number of highest-signal peaks.")
    parser.add_argument('--percentile_thresh', type=float, default=None, 
            action="store",
            help="Percentile cutoff (by signalValue) for peak inclusion"
            )
    parser.add_argument('--seed', type=int, default=1234, 
            help="random seed for reproducibility")
    parser.add_argument('--rmchr', action="store_true", default=False, 
            help="rm chr string from peak chromosomes")
    parser.add_argument('--continuous', default=False, action="store_true",
            help="Include at command line to keep value field continuous")
    parser.add_argument('--center_metric', type=str, 
            help="geom or height, geom gives geometric center of the peak (default). \
                    height gives narrowpeak defined peak summit.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    genome = fa.FastaFile()
    logging.warning("reading in full genome")
    with open(args.fasta) as inf:
        genome.read_whole_file(inf)
    #print("Genome fasta chrom names: {}".format(genome.names))
    #with open("debug.pkl", "wb") as f:
    #    pickle.dump(genome, f)

    peaks = pk.PeakList()
    logging.warning("reading in narrowPeaks")
    peaks.from_narrowPeak_file(args.npfile)
    if args.percentile_thresh is not None:
        peaks.filter_above_percentile(args.percentile_thresh)
    if args.max_peaks != 0:
        peaks.filter_max_n(args.max_peaks)
    if args.rmchr:
        for peak in peaks.generator():
            peak.chrm = peak.chrm.replace("chr", "")
    outfasta = fa.FastaFile()
    outfire = FIREfile()
    #realfire = FIREfile()
    #fakefire = FIREfile()
    for i, peak in enumerate(peaks.generator()):
        this_entry = fa.FastaEntry()
        if args.center_metric == "height":
            peak_center = peak.find_height_center()
        else:
            peak_center = peak.find_geometric_center()

        #print("This peak chrom name: {}".format(peak.chrm))
        this_chrm = genome.pull_entry(peak.chrm)
        this_seq = this_chrm.pull_seq(
            max(peak_center - args.wsize // 2,0), 
            min(peak_center + args.wsize // 2, len(this_chrm)),
        )
        if not set(list(this_seq)).issubset(BASES):
            print("WARNING: skipping peak_{} because it contains at least one character not in {}".format(i,BASES))
            continue
        
        this_entry.set_seq(this_seq)
        this_entry.set_header(">"+"peak_%i"%(i))
        outfasta.add_entry(this_entry)
        if args.continuous:
            #realfire.add_entry(this_entry.chrm_name(), peak.signalval)
            outfire.add_entry(this_entry.chrm_name(), peak.signalval)
        else:
            #realfire.add_entry(this_entry.chrm_name(), 1)
            outfire.add_entry(this_entry.chrm_name(), 1)

        for j in range(args.nrand):
            this_rand = fa.FastaEntry()
            this_seq = "N"*(args.wsize + 1)
            while this_seq.count("N") > args.wsize // 2:
                this_loc = np.random.randint(0+args.wsize // 2 , len(this_chrm)-args.wsize//2)
                this_seq = this_chrm.pull_seq(this_loc-args.wsize//2, this_loc+args.wsize//2)
            this_rand.set_seq(this_seq)
            this_rand.set_header(">"+"peak_%i_%i"%(i,j))
            outfasta.add_entry(this_rand)
            outfire.add_entry(this_rand.chrm_name(),0)
            #fakefire.add_entry(this_rand.chrm_name(),0)
    #finalfire = realfire + fakefire
    #finalfire.shuffle()
    #finalfire.write(args.outpre+".txt")
    outfire.write(args.outpre+".txt")
    with open(args.outpre+".fa", mode="w") as outf:
        outfasta.write(outf)
        #outfire.write(args.outpre+"_fire.txt")
