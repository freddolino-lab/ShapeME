library(DNAshapeR)

args = commandArgs(trailingOnly=TRUE)

fn <- args[1]
pred <- getShape(fn)
