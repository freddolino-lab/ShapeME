library(tidyr)
library(dplyr)
library(optparse)
library(PRROC)

curve.points.for.model <- function(data, test.data, this.formula) {
    model <- glm(this.formula, family=binomial(link="logit"), data=data)
    roc <- roc.curve(weights.class0=test.data$bound, scores.class0=predict(model, type="response", newdata=test.data), curve =TRUE)
    prc <- pr.curve(weights.class0=test.data$bound, scores.class0=predict(model, type="response", newdata=test.data), curve=TRUE)
    out.prc <- data.frame(prc$curve)
    out.roc <- data.frame(roc$curve)
    out.prc$name <- paste(variable.names(model)[-1], collapse="+")
    out.roc$name <- paste(variable.names(model)[-1], collapse="+")
    list(roc = out.roc, prc = out.prc)
}


option_list <- list(
    make_option(c("--trainstatus"), help="categories of training seqs"),
    make_option(c("--trainmotifs"), help="structural motif match for training data"),
    make_option(c("--trainfimo"), default=NULL, help="fimo matches for training data"),
    make_option(c("--teststatus"), default=NULL, help="categories of testing seqs"),
    make_option(c("--testmotifs"), default=NULL, help="structural motif match for testing data"),
    make_option(c("--outpre"), default="linearmodel", help="out prefix"),
    make_option(c("--textonly"), action="store_true", default=FALSE, help="don't output any plots"),
    make_option(c("--testfimo"), default=NULL, help="fimo matches for testing data"))

opt <- parse_args(OptionParser(option_list=option_list))

train.all.data <- read.table(opt$trainstatus, stringsAsFactors=FALSE, header=TRUE, sep="\t")
train.all.data$bound <- as.numeric(train.all.data$score > 0)

# Add training data from motifs
if (!is.null(opt$trainmotifs)){
    train.motifs.data <- read.table(opt$trainmotifs, stringsAsFactors=FALSE, header=FALSE, sep="\t")
    names(train.motifs.data) <- c("name","motifname", "num_matches")
    train.motifs.data <- spread(train.motifs.data, motifname, num_matches)
    train.all.data <- inner_join(train.all.data, train.motifs.data, by="name")
}

# add training data from fimo searches (seq motifs)
if(!is.null(opt$trainfimo)){
    train.fimo.data <- read.table(opt$trainfimo, stringsAsFactors=FALSE, header=FALSE, sep="\t")
    names(train.fimo.data) <- c("name","motifname", "num_matches")
    train.fimo.data <- spread(train.fimo.data, motifname, num_matches)
    train.all.data <- inner_join(train.all.data, train.fimo.data, by="name")
}
# find the best model by adding variables in one at a time
all_model <- glm(bound ~ ., family=binomial(link="logit"), data=subset(train.all.data, select=-c(name, score)))
min_model <- glm(bound ~ 1, family=binomial(link="logit"), data=subset(train.all.data, select=-c(name, score)))
best_model <- step(min_model, k=log(nrow(train.all.data)), scope=list(lower=formula(min_model), upper=formula(all_model)), direction="forward", trace=0)

# if we have testing data test everything on the test data
if(!is.null(opt$teststatus)){
    test.all.data <- read.table(opt$teststatus, stringsAsFactors=FALSE, header=TRUE, sep="\t")
    test.all.data$bound <- as.numeric(test.all.data$score > 0)
    if(!is.null(opt$testmotifs)){
        test.motifs.data <- read.table(opt$testmotifs, stringsAsFactors=FALSE, header=FALSE, sep="\t") 
        names(test.motifs.data) <- c("name","motifname", "num_matches") 
        test.motifs.data <- spread(test.motifs.data, motifname, num_matches) 
        test.all.data <- inner_join(test.all.data, test.motifs.data, by="name")
    }
    
    if(!is.null(opt$testfimo)){
        test.fimo.data <- read.table(opt$testfimo, stringsAsFactors=FALSE, header=FALSE, sep="\t")
        names(test.fimo.data) <- c("name","motifname", "num_matches")
        test.fimo.data <- spread(test.fimo.data, motifname, num_matches)
        test.all.data <- inner_join(test.all.data, test.fimo.data, by="name")
    }
    # calculate the prc and roc with new data
    roc <- roc.curve(weights.class0=test.all.data$bound, scores.class0=predict(best_model, type="response", newdata=test.all.data))
    prc <- pr.curve(weights.class0=test.all.data$bound, scores.class0=predict(best_model, type="response", newdata=test.all.data))
}else{
    # if no new data then just calculate it on the training data
    #
    roc <- roc.curve(weights.class0=train.all.data$bound, scores.class0=predict(best_model, type="response", newdata=train.all.data))
    prc <- pr.curve(weights.class0=train.all.data$bound, scores.class0=predict(best_model, type="response", newdata=train.all.data))
}

auc_roc <- roc$auc
auc_prc <- prc$auc.davis.goadrich
out.stats <- data.frame(fname=opt$trainstatus, num_motifs=length(variable.names(best_model))-1, auc_roc=auc_roc, auc_prc =auc_prc)
write.table(out.stats, paste0(opt$outpre, "out_stats.txt"), col.names=FALSE, row.names=FALSE, quote=FALSE)
if(!opt$textonly){
    library(ggplot2)
    if(exists("test.all.data")){
        test.data <- test.all.data
    }else{
        test.data <- train.all.data
    }
    all_prc <- list()
    all_roc <- list()
    i <- 1
    for(var in names(train.all.data)[c(-1,-2,-3)]){
        out <- curve.points.for.model(subset(train.all.data, select=-c(name, score)), test.data, as.formula(paste("bound ~", var)))
        all_prc[[i]] <- out$prc
        all_roc[[i]] <- out$roc
        i <- i + 1
    }
    # add in best model to curves
    out <- curve.points.for.model(subset(train.all.data, select=-c(name, score)), test.data, formula(best_model))
    all_prc[[i+1]] <- out$prc
    all_roc[[i+1]] <- out$roc
    all_prc <- do.call('rbind', all_prc)
    all_roc <- do.call('rbind', all_roc)

    ggplot(all_roc, aes(x=X1, y=X2, color=name)) + geom_line() + theme_classic() +
    labs(x="FPR", y="TPR", legend="Model")
    ggsave(paste0(opt$outpre, "roc.png"))
    ggplot(all_prc, aes(x=X1, y=X2, color=name)) + geom_line() + theme_classic() +
    labs(x="Recall", y="Precision", legend="Model")
    ggsave(paste0(opt$outpre, "prc.png"))
}

