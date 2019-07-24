import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec as gs
import find_motifs as fm
import numpy as np
import inout
import sys

plt.rc('figure', titlesize=10)

class MotifVis(inout.ShapeMotifFile):

    def plot_helical(self, motif, baseparams, lineparams, grid=None, vmin=None, vmax=None):
        import helical_wheel as hw
        fig = plt.gcf()
        if grid is None:
           total_grid = gs.GridSpec(1,1) 
           grid= total_grid[0]
        nested_gs = gs.GridSpecFromSubplotSpec(len(baseparams), 1, subplot_spec=grid, hspace=0.05)
        basepairs = len(motif['seed'])
        for motif_part in xrange(len(baseparams)):
            circs = []
            circ_vals=[]
            line_vals=[]
            for base in xrange(basepairs):
                val_at_base = motif['seed'][base]
                circ_vals.append(val_at_base.data[baseparams[motif_part]].params)
                line_vals.append(val_at_base.data[lineparams[motif_part]].params)
                circs.append(hw.BasePair(rad=2))
            line_vals = line_vals[:-1]
            helical = hw.HelicalWheel()
            helical.add_circs(circs)
            helical.arrange_circs()
            helical.connect_circs()
            helical.update_circ_colors(circ_vals)
            helical.update_line_colors(line_vals)
            this_ax = plt.Subplot(fig, nested_gs[motif_part])
            helical.plot(cmap_lines=plt.cm.get_cmap("coolwarm"), cmap_circs=plt.cm.get_cmap("BrBG"), ax=this_ax, vmin=vmin, vmax=vmax)
            this_ax.axis('off')
            fig.add_subplot(this_ax)

        return fig

    def plot_motif(self, motif, consolidate=False, plot_type='line', grid = None, ylim= None, ylabs=False):
        param_names = motif['seed'].names
        mat = motif['seed'].matrix()
        fig = plt.gcf()
        if grid is None:
           total_grid = gs.GridSpec(1,1) 
           grid= total_grid[0]
        nested_gs = gs.GridSpecFromSubplotSpec(len(param_names), 1, subplot_spec=grid, hspace=0.05)
        for i, name in enumerate(param_names):
            this_ax = plt.Subplot(fig, nested_gs[i])
            this_ax.plot(range(len(mat[i,:])), mat[i,:])
            this_ax.set_xticks(range(len(mat[i,:])))
            if i == 0:
                this_ax.set_title("%s\n MI:%0.3f"%(motif['name'], motif['mi']))
            if ylim:
                this_ax.set_ylim(ylim)
            if ylabs:
                this_ax.set_ylabel(name)
            else:
                this_ax.set_yticklabels([])
            this_ax.set_xticklabels([])
            fig.add_subplot(this_ax)
        this_ax.set_xticklabels(range(len(mat[i,:])))
        return fig

    def plot_motifs(self, grid = None, ylim=None, name= ""):
        fig = plt.gcf()
        if grid is None:
            total_grid = gs.GridSpec(1,1)
            grid = total_grid[0]
        nested_gs = gs.GridSpecFromSubplotSpec(1, len(self.motifs), subplot_spec=grid, wspace=0.1)
        for i, motif in enumerate(self.motifs):
            if i == 0:
                ylabs = True
            else:
                ylabs = False
            self.plot_motif(motif, grid=nested_gs[i], ylim=ylim, ylabs=ylabs)
        return fig

    def plot_helical_motifs(self, motif, baseparams, lineparams, grid=None, vmin=None, vmax=None):
        fig = plt.gcf()
        if grid is None:
            total_grid = gs.GridSpec(1,1)
            grid = total_grid[0]
        nested_gs = gs.GridSpecFromSubplotSpec(1, len(self.motifs), subplot_spec=grid, wspace=0.1)
        for i, motif in enumerate(self.motifs):
            self.plot_helical(motif, baseparams, lineparams, nested_gs[i], vmin, vmax)
        return fig

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
        # add 5 for the text
        width = out_mat.shape[1] + out_mat.shape[0]/5.0
        height = out_mat.shape[0]
        if height < 10:
            fig_height = 5
        else:
            fig_height = height/2.0
        aspect_ratio = float(width)/float(height)
        fig=plt.figure(figsize=(fig_height*aspect_ratio, fig_height))
        g = gs.GridSpec(1, 2, height_ratios=(1,1), width_ratios=(20,1), wspace=0.0, hspace=0.0)
        ax = plt.subplot(g[0,0])
        imfig = plt.imshow(out_mat, aspect='equal',interpolation='nearest', cmap="bwr", 
                           vmax = np.nanmax(out_mat[np.isfinite(out_mat)]), vmin= -np.nanmax(out_mat[np.isfinite(out_mat)]), *args)
        ax.set_title(self.get_title(), fontsize=12, horizontalalignment='right')
        xlabels, ylabels = self.get_labels()
        ax.set_yticks(np.arange(0, out_mat.shape[0]))
        ax.set_yticklabels(ylabels)
        ax.set_xticks(np.arange(0, out_mat.shape[1]))
        ax.set_xticklabels(xlabels)
        ax_cbar = plt.subplot(g[0,1])
        plt.colorbar(imfig, cax=ax_cbar)
        g.tight_layout(fig)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0)

    def display_motifs(self, outfile, *args):
        width = len(self.motifs[0]['seed'])
        height = len(self.motifs[0]['seed'].names)*len(self.motifs)
        if height < 10:
            fig_height = 5
        else:
            fig_height = height/2.0
        aspect_ratio = float(width)/float(height)
        fig = plt.figure(figsize=(fig_height * aspect_ratio, fig_height))
        g = gs.GridSpec(len(self.motifs)+1, 1, width_ratios=[1],
                        height_ratios=[3]*len(self.motifs)+[1], hspace=0.01, wspace=0.0)

        this_min, this_max = self.find_max_min()
        
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
        axes = plt.subplot(g[-1,0])
        plt.colorbar(this_fig,orientation="horizontal", cax=axes)

        g.tight_layout(fig)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0)

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
            ylabels.append("$I(M;C)$: %.3f $H(M)$: %.3f"%(motif['mi'], motif['motif_entropy']))
        xlabels = []
        for val in sorted(motif["enrichment"].keys()):
            xlabels.append(val)
        return (xlabels, ylabels)

    def get_title(self):
        return "$H(C)$: %.3f"%(self.motifs[0]['category_entropy'])

    def enrichment_heatmap_txt(self, outfile):
        all_lines=[]
        for i, motif in enumerate(sorted(self.motifs,key=lambda x:x['mi'], reverse=True)):
            motif_vals = ["%.3f"%(motif['mi']), "%.3f"%(motif['motif_entropy'])]
            for val in sorted(motif["enrichment"].keys()):
                this_logodds = fm.two_way_to_log_odds(motif["enrichment"][val])
                motif_vals.append("%.4e"%this_logodds)
            all_lines.append(motif_vals)
        with open(outfile, mode="w") as outf:
            outf.write("#%s\n"%(self.get_title()))
            outf.write("mi\tentropy\t"+"\t".join([str(val) for val in sorted(motif["enrichment"].keys())]) +"\n")
            for line in all_lines:
                outf.write("\t".join(line)+"\n")

         
