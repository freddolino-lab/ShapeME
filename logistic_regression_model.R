library(tidyr)
library(ggplot2)
library(optparse)
source("/home/mbwolfe/src/mike_rtools/modelvistools.R")
source("/home/mbwolfe/src/mike_rtools/myggthemes.R")

option_list <- list(
    make_option(c("--trainstatus"), help="categories of training seqs"),
    make_option(c("--trainmotifs"), help="structural motif match for training data"),
    make_option(c("--trainfimo"), default=NULL, help="fimo matches for training data"),
    make_option(c("--teststatus"), default=NULL, help="categories of testing seqs"),
    make_option(c("--testmotifs"), default=NULL, help="structural motif match for testing data"),
    make_option(c("--outpre"), default="linearmodel", help="out prefix"),
    make_option(c("--testfimo"), default=NULL, help="fimo matches for testing data"))

opt <- parse_args(OptionParser(option_list=option_list))

train.all.data <- read.table(opt$trainstatus, stringsAsFactors=FALSE, header=TRUE, sep="\t")
train.all.data$bound <- as.numeric(train.all.data$score > 0)

# Add training data from motifs
if (!is.null(opt$trainmotifs)){
    train.motifs.data <- read.table(opt$trainmotifs, stringsAsFactors=FALSE, header=FALSE, sep="\t")
    names(train.motifs.data) <- c("name","motifname", "num_matches")
    train.motifs.data <- spread(train.motifs.data, motifname, num_matches)
    train.all.data <- merge(train.all.data, train.motifs.data, by="name")
}

# add training data from fimo searches (seq motifs)
if(!is.null(opt$trainfimo)){
    train.fimo.data <- read.table(opt$trainfimo, stringsAsFactors=FALSE, header=FALSE, sep="\t")
    names(train.fimo.data) <- c("name","motifname", "num_matches")
    train.fimo.data <- spread(train.fimo.data, motifname, num_matches)
    train.all.data <- merge(train.all.data, train.fimo.data, by="name")
}

# find the best model by adding variables in one at a time
models <- add.one.in(train.all.data, "bound", "1", names(train.all.data[c(-1,-2,-3)]))
best_model <- find.best.model.up(glm(formula="bound ~ 1", family=binomial(link="logit"), data=train.all.data), 
                                 "bound", models, train.all.data)
print(summary(best_model))

# get the correct cutoffs based on the training data for the roc and prc curves
cutoffs <- find.roc.cutoffs(train.all.data$bound, predict(best_model, type="response"))
train.roc <- calc.roc(train.all.data$bound, predict(best_model, type="response"), cutoffs)
cutoffs <- get.convex.hull.cutoffs(train.roc)

# if we have testing data test everything on the test data
if(!is.null(opt$teststatus)){
    test.all.data <- read.table(opt$teststatus, stringsAsFactors=FALSE, header=TRUE, sep="\t")
    test.all.data$bound <- as.numeric(test.all.data$score > 0)
    if(!is.null(opt$testmotifs)){
        test.motifs.data <- read.table(opt$testmotifs, stringsAsFactors=FALSE, header=FALSE, sep="\t") 
        names(test.motifs.data) <- c("name","motifname", "num_matches") 
        test.motifs.data <- spread(test.motifs.data, motifname, num_matches) 
        test.all.data <- merge(test.all.data, test.motifs.data, by="name")
    }
    
    if(!is.null(opt$testfimo)){
        test.fimo.data <- read.table(opt$testfimo, stringsAsFactors=FALSE, header=FALSE, sep="\t")
        names(test.fimo.data) <- c("name","motifname", "num_matches")
        test.fimo.data <- spread(test.fimo.data, motifname, num_matches)
        test.all.data <- merge(test.all.data, test.fimo.data, by="name")
    }
    # calculate the prc and roc with new data
    roc <- calc.roc(test.all.data$bound, predict(best_model, type="response", newdata=test.all.data), cutoffs)
    prc <- calc.prc(test.all.data$bound, predict(best_model, type="response", newdata=test.all.data), cutoffs)
}else{
    # if no new data then just calculate it on the training data
    roc <- calc.roc(train.all.data$bound, predict(best_model, type="response"), cutoffs)
    prc <- calc.prc(train.all.data$bound, predict(best_model, type="response"), cutoffs)
}

auc_roc <- calc.auc(1-roc$specificity, roc$recall)
auc_prc <- calc.auc(prc$recall, prc$precision)
out.stats <- data.frame(fname=opt$trainstatus, num_motifs=length(variable.names(best_model))-1, auc_roc=auc_roc, auc_prc =auc_prc)
write.table(out.stats, paste0(opt$outpre, "out_stats.txt"), col.names=FALSE, row.names=FALSE, quote=FALSE)
models[[length(models)+1]] <- best_model
if(!is.null(opt$teststatus)){
    fig <- logit_regress_perform(models, newdata=test.all.data)
    fig <- fig + theme_pub()
    ggsave(paste0(opt$outpre, "roc.png"))
    fig <- logit_regress_perform(models, metric="prc", newdata=test.all.data)
    fig <- fig + theme_pub()
    ggsave(paste0(opt$outpre, "prc.png"))
}else{
    fig <- logit_regress_perform(models)
    fig <- fig + theme_pub()
    ggsave(paste0(opt$outpre, "roc.png"))
    fig <- logit_regress_perform(models, metric="prc")
    fig <- fig + theme_pub()
    ggsave(paste0(opt$outpre, "prc.png"))
}

warnings()
