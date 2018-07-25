import sys
import argparse
import inout
import fimopytools as fpt


class InStream(object):

    def __init__(self, instream):
        self.instream = instream
        self.curr_line = ""

    def update_line(self):
        try:
            line = next(self.instream)
            self.curr_line = line.rstrip().split("\t")
        except StopIteration:
            self.curr_line = None

    def __getitem__(self, val):
        return self.curr_line[val]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infimo', type=str)
    parser.add_argument('--infasta', type=str)
    parser.add_argument('--metric', type=str, default="max", help="max or min score")
    parser.add_argument('--score_field', type=int, default=5, help="what field is score in?")
    parser.add_argument('--motif_name', type=int, default=0, help="what field is motif name in?")
    parser.add_argument('--seq_name', type=int, default=1, help="what field is seq name in?")
    parser.add_argument('outpre', type=str)

    args = parser.parse_args()

    metric = {"max": max, "min": min}
    this_metric = metric[args.metric]
    if args.infimo is not "-":
        seqs = inout.FastaFile()
        with open(args.infasta, mode="r") as inf:
            seqs.read_whole_file(inf)

        fimo_matches = fpt.FimoFile()
        fimo_matches.parse(args.infimo)
        with open(args.outpre+"fimo_counts.txt", mode="w") as outf:

            for entry in seqs:
                try:
                    these_matches = fimo_matches.pull_entry(entry.chrm_name())
                    num_matches = len(these_matches)
                except KeyError:
                    num_matches = 0
                outf.write("%s\t%s\t%i\n"%(entry.chrm_name(), "fimo", num_matches))
    else:
        if args.outpre == "-":
            outf = sys.stdout
        else:
            outf = open(args.outpre+args.metric+"_scores.txt", mode="w")
        instream = InStream(sys.stdin)
        # read the first line
        instream.update_line()
        # iter through the header
        while instream[0][0] == "#":
            instream.update_line()
        # iter through the entire file
        while instream.curr_line is not None:
            # initialize all the data needed for a seq motif-pair
            this_name = instream[args.seq_name]
            this_motif = instream[args.motif_name]
            these_matches = []
            # while still on this seq and not at end of file add the score
            # of the motif to a list and iter
            while instream.curr_line is not None and this_name == instream[args.seq_name]:
                these_matches.append(float(instream[args.score_field]))
                instream.update_line()
            # when no longer matching names, write out to the outfile
            outf.write("%s\t%s\t%0.4f\n"%(this_name, this_motif, this_metric(these_matches)))
        outf.close()







