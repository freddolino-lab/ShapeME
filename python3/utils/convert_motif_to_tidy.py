import inout
import sys

if __name__ == "__main__":
    infile = sys.argv[1]
    unnormalized = int(sys.argv[2])

    motif = inout.ShapeMotifFile()
    motif.read_file(infile)
    if unnormalized == 1:
        motif.unnormalize()

    motif.to_tidy(1)
    
