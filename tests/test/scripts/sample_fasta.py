import sys
sys.path.append("/home/mbwolfe/src/tinker_motif_finder/DNAshape_motif_finder/")
import inout
from pathlib import Path

if __name__ == "__main__":
    infire = sys.argv[1]
    outprefix = sys.argv[2]
    infiles = sys.argv[3:]
    
    genomes = [inout.FastaFile() for inf in infiles]
    outgenomes = [inout.FastaFile() for inf in infiles]
    # read in data
    for genome, inf in zip(genomes, infiles):
        with open(inf) as inhandle:
            genome.read_whole_datafile(inhandle)
    # read in fire file
    fire = inout.SeqDatabase(names = [])
    fire.read(infire)

    # sample
    subset = fire.random_subset_by_class(0.1)
    names = subset.get_names()

    subset.write(outprefix + Path(infire).resolve().name)

    # update new genomes with new names
    for name in names:
        for genome, outgenome in zip(genomes, outgenomes):
            new_entry = genome.pull_entry(name)
            outgenome.add_entry(new_entry)
    outfiles = [outprefix + Path(inf).resolve().name for inf in infiles]
    for outgenome, outf in zip(outgenomes, outfiles):
        with open(outf, mode = "w") as outf_handle:
            outgenome.write(outf_handle, wrap = 30, delim = ",")



