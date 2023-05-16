library(DNAshapeR)

args = commandArgs(trailingOnly=TRUE)

fn <- args[1]
print(fn)
pred <- getShape(fn)
