import inout
import dnashapeparams as dsp
import logging
import argparse
import numpy as np
import scipy.optimize as opt
np.random.seed(1234)

def make_initial_seeds(cats, wsize,wstart,wend):
    seeds = []
    for param in cats:
        for window in param.sliding_windows(wsize, start=wstart, end=wend):
            seeds.append(window)
    return seeds

def optimize_mi(param_vec, data):
    threshold = param_vec[-1]
    this_discrete = []
    this_data = data.random_subset(0.1)
    for this_seq in this_data.vectors:
        seq_pass = 0
        for motif in this_seq:
            distance = motif.distance(np.array(param_vec[:-1]), vec=True, cache=True)
            if distance < threshold:
                seq_pass = 1
                break
        this_discrete.append(seq_pass)
    this_mi = this_data.mutual_information(this_discrete)
    return this_mi

def find_initial_threshold(cats, wsize, wstart, wend):
    seeds = []
    for param in cats:
        for window in param.sliding_windows(wsize, start=wstart, end=wend):
            seeds.append(window)
    distances = []
    for seed1 in seeds:
        for seed2 in seeds:
            distances.append(seed1.distance(seed2.as_vector(cache=True), vec=True, cache=True))
    distances = np.array(distances)
    distances = distances[distances > 0]
    distances = np.log(distances)

    mean = np.mean(distances)
    stdev = np.std(distances)
    return np.exp(mean-3*stdev)


def read_parameter_file(infile):
    fastadata = inout.FastaFile()
    with open(infile) as f:
        fastadata.read_whole_datafile(f)
    return fastadata

class MotifMatch(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return "Distance is %f"%self.value

def greedy_search(cats, threshold = 10, number=1000):
    seeds = []
    cats_shuffled = cats.shuffle()
    for seq in cats_shuffled.category_subset(1).iterate_through_precompute():
        if(len(seeds) >= number):
            break
        for motif in seq:
            try:
                for motif2 in seeds:
                    distance = motif2.distance(motif.as_vector(), vec=True, cache=True)
                    if distance < threshold:
                        raise MotifMatch(distance)
                seeds.append(motif)             
            except MotifMatch as e:
                continue
    return seeds
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', action='store', type=str,
                         help='inputfile with names and scores')
    parser.add_argument('--params', nargs="+", type=str,
                         help='inputfile with mgw scores')
    parser.add_argument('--param_names', nargs="+", type=str,
                         help='parameter names')

    
    args = parser.parse_args()
    
    logging.warning("Reading in files")
    all_params = [read_parameter_file(x) for x in args.params]
    cats = inout.SeqDatabase(names=[])
    cats.read(args.infile)

    for name, param in zip(cats.names, cats):
        for this_param, this_param_name in zip(all_params, args.param_names):
            param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))

    logging.warning("Normalizing parameters")
    cats.normalize_params()

    logging.warning("Determining inital threshold")
    threshold = find_initial_threshold(cats.random_subset(0.05), 15, wstart=2, wend=498)
    logging.warning("Using %f as an initial threshold"%(threshold))

    #all_15mers = make_initial_seeds(cats.category_subset(1), 15, 2, 498)

    logging.warning("Precomputing all windows")
    cats.pre_compute_windows(15, wstart=2, wend=498)
    all_mi = []
    #threshold = 10
    logging.warning("Greedy search for possible motifs")
    possible_motifs = greedy_search(cats, threshold, 100)
    logging.warning("%s possible motifs"%(len(possible_motifs)))
    logging.warning("Finding seeds")
#    for motif1 in [possible_motifs[x] for x in np.random.randint(0,high=len(possible_motifs), size=100)]:
    for motif1 in possible_motifs:
        this_discrete = []
        for this_seq in cats.iterate_through_precompute():
            seq_pass = 0
            this_mi = 0
            for motif2 in this_seq:
                distance = motif1.distance(motif2.as_vector(cache=True), vec=True,cache=True)
                if distance < threshold:
                    seq_pass = 1
                    break
            this_discrete.append(seq_pass)
        this_mi = cats.mutual_information(this_discrete)
        all_mi.append((this_mi, motif1.as_vector()))
    logging.warning("Sorting seeds")
    all_mi = sorted(all_mi, key=lambda x: x[0])
    logging.warning(all_mi[-5:])
    logging.warning("Taking top seed to optimize")
    motif_to_optimize = list(all_mi[-1][-1])
    motif_to_optimize.append(threshold)
    logging.warning(optimize_mi(motif_to_optimize, cats))
    logging.warning("Optimizing top seed using Nelder Mead")
    final = opt.minimize(lambda x: -optimize_mi(x, data=cats), motif_to_optimize, method="nelder-mead", options={'disp':True})
    logging.warning(final)
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=cats), motif_to_optimize)
    #logging.warning(final)
