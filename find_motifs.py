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
    this_data = data.random_subset(args.optimize_perc)
    this_discrete = generate_peak_vector(this_data, param_vec[:-1], threshold)
#    for this_seq in this_data.vectors:
#        seq_pass = 0
#        for motif in this_seq:
#            distance = motif.distance(np.array(param_vec[:-1]), vec=True, cache=True)
#            if distance < threshold:
#                seq_pass = 1
#                break
#        this_discrete.append(seq_pass)
    this_mi = this_data.mutual_information(this_discrete)
    return this_mi

def generate_peak_vector(data, motif_vec, threshold):
    this_discrete = []
    for this_seq in data.iterate_through_precompute():
        seq_pass = 0
        for this_motif in this_seq:
            distance = this_motif.distance(motif_vec, vec=True, cache=True)
            if distance < threshold:
                seq_pass = 1
                break
        this_discrete.append(seq_pass)
    return this_discrete

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
    return np.exp(mean-2*stdev)


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
    for seq in cats_shuffled.iterate_through_precompute():
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
    parser.add_argument('--windowsize', type=int,
                         help='window_size to use', default=15)
    parser.add_argument('--windowstart', type=int,
                         help='window_start to use', default=2)
    parser.add_argument('--windowend', type=int,
                         help='window_start to use', default=None)
    parser.add_argument('--num_seeds', type=int,
                         help='number of seeds to start', default=100)
    parser.add_argument('--threshold_perc', type=float, default=0.05)
    parser.add_argument('--optimize_perc', type=float, default=0.1)
    parser.add_argument('--seed_perc', type=float, default=1)
    parser.add_argument('--continuous', action="store_true")

    
    args = parser.parse_args()
    
    logging.warning("Reading in files")
    all_params = [read_parameter_file(x) for x in args.params]
    cats = inout.SeqDatabase(names=[])
    cats.read(args.infile,float)
    if args.continuous:
        logging.warning("Discretizing data")
        cats.discretize()

    for name, param in zip(cats.names, cats):
        for this_param, this_param_name in zip(all_params, args.param_names):
            param.add_shape_param(dsp.ShapeParamSeq(this_param_name, this_param.pull_entry(name).seq))

    logging.warning("Normalizing parameters")
    cats.normalize_params()

    logging.warning("Determining inital threshold")
    threshold = find_initial_threshold(cats.random_subset(args.threshold_perc), args.windowsize, wstart=args.windowstart, wend=args.windowend)
    logging.warning("Using %f as an initial threshold"%(threshold))

    #all_15mers = make_initial_seeds(cats.category_subset(1), 15, 2, 498)

    logging.warning("Precomputing all windows")
    cats.pre_compute_windows(args.windowsize, wstart=args.windowstart, wend=args.windowend)
    all_mi = []
    #threshold = 10
    logging.warning("Greedy search for possible motifs")
    possible_motifs = greedy_search(cats, 2*threshold, args.num_seeds)
    logging.warning("%s possible motifs"%(len(possible_motifs)))
    logging.warning("Finding seeds")
#    for motif1 in [possible_motifs[x] for x in np.random.randint(0,high=len(possible_motifs), size=100)]:
    for i,motif1 in enumerate(possible_motifs):
        if not (i % 10):
            logging.warning("Computing motif %s"%i)
        this_mi = 0
        this_discrete = []
        if args.optimize_perc != 1:
            this_cats = cats.random_subset(args.seed_perc)
        else:
            this_cats = cats
        this_discrete = generate_peak_vector(this_cats, motif1.as_vector(cache=True), threshold)
        this_mi = this_cats.mutual_information(this_discrete)
        all_mi.append((this_mi, motif1.as_vector(cache=True)))
    logging.warning("Sorting seeds")
    all_mi = sorted(all_mi, key=lambda x: x[0])
    logging.warning(all_mi[-1:])
    logging.warning("Taking top seed to optimize")
    motif_to_optimize = list(all_mi[-1][-1])
    enriched = cats.calculate_enrichment(np.array(generate_peak_vector(cats, motif_to_optimize, threshold)))
    for key in sorted(enriched.keys()):
        logging.warning("Enrichment for Cat %s is %s"%(key, enriched[key]))
    motif_to_optimize.append(threshold)
    logging.warning(optimize_mi(motif_to_optimize, cats))
    logging.warning("Optimizing top seed using Nelder Mead")
    final = opt.minimize(lambda x: -optimize_mi(x, data=cats), motif_to_optimize, method="nelder-mead", options={'disp':True})
    logging.warning(final['x'])
    #final = opt.basinhopping(lambda x: -optimize_mi(x, data=cats), motif_to_optimize)
    #logging.warning(final)
