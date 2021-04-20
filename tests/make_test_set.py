import sys
sys.path.append("../")
import inout
import dnashapeparams as dsp
import numpy as np

# FIRE format file with shape param for each position?
infile1 = sys.argv[1] 
# fasta sequence 
infile2 = sys.argv[2]
outfile = sys.argv[3]

indata = inout.FastaFile()
cats = inout.SeqDatabase(names=[])
cats.read(infile1)
with open(infile2) as f:
    indata.read_whole_datafile(f)
for name, param in zip(cats.names, cats):
    param.add_shape_param(dsp.ShapeParamSeq(name="MGW", params=indata.pull_entry(name).seq))

cats.normalize_params()
center, spread = cats.center_spread["MGW"]
correct_motif = dsp.ShapeParamSeq(name="MGW", params = list(5.0*np.ones(15)))
#correct_motif.normalize_values(center, spread)
for i, param in enumerate(cats):
    if cats.values[i] == 1 or i == 10:
        param.data["MGW"].params[300:315] = correct_motif.params * np.random.normal(1.0, spread, 15)
    else:
        for motif in param.sliding_windows(15, start=2, end=498):
            distance = motif.distance(np.array(correct_motif), vec=True)
            if distance < 10:
                print("Warning motif found in negative set")

new_fasta = inout.FastaFile()
outf = open(outfile, mode="w")
for name, param in zip(cats.names, cats):
    param.data["MGW"].unnormalize_values(center, spread)
    outstr = param.as_vector()
    outstr = ["NA" if np.isnan(x) else "%f"%x for x in outstr]
    outf.write(">"+name+"\n"+",".join(outstr)+"\n")
outf.close()





