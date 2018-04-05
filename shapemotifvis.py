import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
import find_motifs as fm
import numpy as np

class EnrichmentHeatmap(object):

    def __init__(self, motifs=None):
        self.motifs = motifs
    
    def plot_optimization(self, outfile):
        f, axes = plt.subplots(len(self.motifs))
        if len(self.motifs) < 2:
            axes = [axes]
        for i, motif in enumerate(sorted(self.motifs, key = lambda x:x['mi'], reverse=True)):
            axes[i].scatter(motif['opt_info']['eval'], motif['opt_info']['value'])
            #axes[i].set_title("Motif %i"%i)
        plt.savefig(outfile)

    def find_max_min(self):
        maxs = []
        mins = []
        for motif in self.motifs:
            maxs.append(np.amax(motif['seed'].matrix()))
            mins.append(np.amin(motif['seed'].matrix()))

        return (np.min(mins), np.max(maxs))

    def display_enrichment(self, outfile, *args):
        out_mat = self.convert_to_enrichment_mat()
        plt.figure()
        ax = plt.gca()
        fig = ax.imshow(out_mat, interpolation='nearest', cmap="bwr", vmax = np.max(out_mat), vmin= -np.max(out_mat), *args)
        plt.colorbar(fig)
        plt.title(self.get_title())
        xlabels, ylabels = self.get_labels()
        ax.set_yticks(np.arange(0, out_mat.shape[0]))
        ax.set_yticklabels(ylabels)
        ax.set_xticks(np.arange(0, out_mat.shape[1]))
        ax.set_xticklabels(xlabels)
        plt.tight_layout()
        plt.savefig(outfile)

    def display_motifs(self, outfile, *args):
        f, axes = plt.subplots(len(self.motifs),1)
        g = gs.GridSpec(len(self.motifs), 2, width_ratios=(20,1))

        this_min, this_max = self.find_max_min()
        if len(self.motifs) < 2:
            axes = [axes]
        for i, motif in enumerate(sorted(self.motifs, key=lambda x: x['mi'], reverse=True)):
            this_seed = motif['seed']
            this_matrix = this_seed.matrix()
            axes = plt.subplot(g[i,0])
            this_fig = plt.imshow(this_matrix, interpolation='nearest', cmap='PRGn', vmin=this_min, vmax=this_max, *args)
            axes.set_yticks(np.arange(0,this_matrix.shape[0]))
            axes.set_yticklabels(this_seed.names)
            if i == len(self.motifs)-1:
                axes.set_xticks(np.arange(0,this_matrix.shape[1]))
                axes.set_xticklabels(np.arange(0, this_matrix.shape[1]))
            else:
                axes.set_xticks([])
        axes = plt.subplot(g[:,1])
        plt.colorbar(this_fig, cax=axes)

        plt.tight_layout()
        plt.savefig(outfile)

    def convert_to_enrichment_mat(self):
        all_vals = []
        for motif in sorted(self.motifs, key= lambda x : x['mi'], reverse=True):
            motif_vals = []
            for val in sorted(motif["enrichment"].keys()):
                this_logodds = fm.two_way_to_log_odds(motif["enrichment"][val])
                motif_vals.append(this_logodds)
            all_vals.append(motif_vals)
        return np.array(all_vals)

    def get_labels(self):
        ylabels = []
        for motif in sorted(self.motifs, key=lambda x : x['mi'], reverse=True):
            ylabels.append("MI: %f Entropy: %f"%(motif['mi'], motif['motif_entropy']))
        xlabels = []
        for val in sorted(motif["enrichment"].keys()):
            xlabels.append(val)
        return (xlabels, ylabels)

    def get_title(self):
        return "Category Entropy: %s"%(self.motifs[0]['category_entropy'])
