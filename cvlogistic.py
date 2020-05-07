import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem
import logging


def plot_score(clf, outf):
    """ Plot scores as a function of the C parameter.
    Assumes multinomial model
    """
    first_class = list(clf.scores_.keys())[0]
    all_scores = clf.scores_[first_class]
    # make an empty array of size Cs and intercept + coefficients
    total_num = np.zeros((len(clf.Cs_), len(clf.coef_[0])+1))
    # find how many non-zero coefficients there are
    for key in clf.coefs_paths_.keys():
        these_coefs = clf.coefs_paths_[key]
        # determine the average over CV reps
        CV_avg = np.mean(these_coefs, axis=0)
        # determine the number of non-zero coefficients based on the tolerance
        num_nonzero = abs(CV_avg) > 0
        total_num = np.logical_or(total_num, num_nonzero)
    num_nonzero = np.sum(total_num, axis=1)
    CV_avg = np.mean(all_scores,axis=0)
    CV_std = sem(all_scores, axis=0)
    best_c = find_best_c(clf)
    plt.figure()
    plt.errorbar(-np.log10(1.0/clf.Cs_), CV_avg, yerr=CV_std)
    plt.axvline(-np.log10(1.0/clf.C_[0]), ls="--", color='k')
    plt.axvline(-np.log10(1.0/best_c), ls="--", color='r')
    plt.xlabel("-log10(1/C)")
    plt.ylabel("Accuracy")
    ax = plt.gca()
    ax.margins(x=0)
    ax2 = ax.twiny()
    ax2.set_xlabel("Number of non-zero coef (+ intercept)")
    ax2.set_xticks(-np.log10(1.0/clf.Cs_))
    ax2.set_xticklabels(num_nonzero)
    plt.savefig(outf)


def plot_coef_paths(clf, outf, cls=1, tol=0):
    """ Plots the average path of each coefficient for a class
    """
    # pull the paths for the coefficients for a class
    these_coefs = clf.coefs_paths_[cls]
    # determine the average over CV reps
    CV_avg = np.mean(these_coefs, axis=0)
    # determine the number of non-zero coefficients based on the tolerance
    num_nonzero = np.sum(abs(CV_avg) > tol, axis=1)
    # determine the std of the CV reps
    CV_std = sem(these_coefs, axis=0)
    # plot each coefficient over the different regularization parameters
    # Have to transpose the arrays to iterate over the correct values
    plt.figure()
    for i, (coef, stderr) in enumerate(zip(CV_avg.T, CV_std.T)):
        plt.errorbar(-np.log10(1.0/clf.Cs_), coef, yerr=stderr, label=i)
    # since this is multinomial, C is the same for each class
    plt.axvline(-np.log10(1.0/clf.C_[0]), ls="--", color='k')
    plt.legend()
    plt.axhline(0, ls="--", color='k')
    plt.xlabel("-log10(1/C)")
    plt.ylabel("Coefficient value")
    # add an axis telling how many coefficients are > 0
    ax = plt.gca()
    ax.margins(x=0)
    ax2 = ax.twiny()
    ax2.set_xlabel("Number of non-zero coef (+ intercept)")
    ax2.set_xticks(-np.log10(1.0/clf.Cs_))
    ax2.set_xticklabels(num_nonzero)
    plt.savefig(outf)

def plot_coef_per_class(clf, outf):
    # deals with edge case where there are only two classes. Pulls the only
    # class the model is fit for
    clses = clf.classes_
    if len(clses) <= 2:
        clses = [clses[-1]]
    c_ind = choose_features(clf, tol=0)
    num_classes = len(clses)
    max_x = np.max(clf.coef_)
    min_x = np.min(clf.coef_)
    if num_classes > 1:
        fig, axes = plt.subplots(1,num_classes)
        for cls, ax, coef in zip(clses, axes, clf.coef_):
            ax.barh(np.arange(0,len(coef[c_ind])), coef[c_ind])
            ax.axvline(0, ls="--", color='k')
            ax.set_xlim((min_x, max_x))
            ax.set_title("%s"%cls)
    else:
        plt.figure()
        plt.barh(np.arange(0,len(clf.coef_[0][c_ind])), clf.coef_[0][c_ind])
        plt.axvline(0, ls="--", color='k')
        plt.xlim((min_x, max_x))
        plt.title("%s"%clses[0])
    plt.savefig(outf)

def write_coef_per_class(clf, outf):
    outfmt = "%s\t%s\t%s\n"
    c_ind = choose_features(clf, tol=0)
    # deals with edge case where there are only two classes. Pulls the only
    # class the model is fit for
    clses = clf.classes_
    if len(clses) <= 2:
        clses = [clses[-1]]
    with open(outf, mode="w") as out:
        out.write("coef\tval\tclass\n")
        for cls, coef, inter in zip(clses, clf.coef_, clf.intercept_):
            out.write(outfmt%("intercept", inter, cls))
            for i, chosen_coef in enumerate(coef[c_ind]):
                out.write(outfmt%(i, chosen_coef, cls))

def choose_features(clf, tol=0):
    coefs = clf.coef_
    return(np.argwhere(np.max(abs(coefs), axis=0) > tol).flatten())

def find_best_c(clf):
    first_class = list(clf.scores_.keys())[0]
    all_scores = clf.scores_[first_class]
    CV_avg = np.mean(all_scores, axis=0)
    CV_sem = sem(all_scores, axis=0)
    #print(CV_sem)
    best_score = np.argmax(CV_avg)
    logging.info("best score\n mean: %s, sem: %s"%(CV_avg[best_score],CV_sem[best_score]))
    score_1se = CV_avg[best_score] - 2*CV_sem[best_score]
    logging.info("Score within 2 se\n %s"%score_1se)
    best_c = clf.Cs_[np.argmax(CV_avg >= score_1se)]
    return best_c
    

if __name__ == "__main__": 
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets

#    X, y = datasets.load_iris(return_X_y=True)
#    X = np.c_[X, np.random.normal(0,2, 150)]
#
#    X = StandardScaler().fit_transform(X)
#    index = y <= 1
#    y = y[index]
#    X = X[index,:]
#    clf = LogisticRegressionCV(cv=5, random_state=42, multi_class='multinomial', penalty='l1', solver='saga', max_iter=4000).fit(X, y)
#    plot_coef_per_class(clf, "coef_per_class.png")
#    print(clf.C_)
#    for cls in clf.scores_.keys():
#        plot_coef_paths(clf, "coef_paths_class%s.png"%cls, cls=cls)
#    plot_score(clf, "score_over_c.png")
#    indices = choose_features(clf, tol=0)
#    print(indices)
#    print([clf.coef_[0][index] for index in indices])
#    write_coef_per_class(clf, "coef_per_class.txt")

    # lets make a gross data set of 30 random variables and 3 that have
    # a decent relationship with y

    y = np.random.randint(0, 4, 1000)
    X1 = y * np.random.normal(10, 3, 1000)
    X2 = np.array([ np.random.normal(5,0.5) if val == 2 else np.random.normal(0,0.5) for val in y])
    X3 = np.array([ np.random.normal(5,0.5) if val == 3 else np.random.normal(0,0.5) for val in y])
    X4 = np.array([ np.random.normal(5,0.5) if val == 1 else np.random.normal(0,0.5) for val in y])
    X5 = np.array([ np.random.normal(5,0.5) if val == 0 else np.random.normal(0,0.5) for val in y])
    X = np.c_[X1, X2, X3, X4, X5]
    for val in np.arange(0,30):
        this_X = np.random.normal(0, val, 1000)
        X = np.c_[X, this_X]

    X = StandardScaler().fit_transform(X)
 
    clf = LogisticRegressionCV(cv=5, random_state=42, multi_class='multinomial', penalty='l1', solver='saga', max_iter=4000, Cs=100).fit(X, y)
    best_c = find_best_c(clf)
    clf_f = LogisticRegression(C=best_c, multi_class='multinomial', penalty='l1', solver='saga', max_iter=4000).fit(X, y)
    plot_coef_per_class(clf_f, "coef_per_class.png")
    for cls in clf.scores_.keys():
        plot_coef_paths(clf, "coef_paths_class%s.png"%cls, cls=cls)
    plot_score(clf, "score_over_c.png")
    indices = choose_features(clf_f, tol=0)
    for i,val in enumerate(indices):
        print("%i is variable %i"%(i, val))
    write_coef_per_class(clf_f, "coef_per_class.txt")

