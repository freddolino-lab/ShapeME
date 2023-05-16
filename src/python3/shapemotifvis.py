import matplotlib as mpl
import os
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import infer_motifs as im
import numpy as np
import inout
import sys
from pathlib import Path
#from svgpathtools import svg2paths
#from svgpath2mpl import parse_path

this_path = Path(__file__).parent.absolute()

#plt.rc('figure', titlesize=10)

def get_image(path):
    img_arr = plt.imread(path, format="png")
    return img_arr

def scale_image(img_arr, scale=1):
    # scale the alpha layer
    img_cp = img_arr.copy()
    img_cp[:,:,3] *= scale
    #img = OffsetImage(img_arr, zoom=scale*0.1)
    img = OffsetImage(img_cp, zoom=0.125)
    return img

def plot_logo(
        motifs,
        file_name,
        shape_lut,
        top_n = 30,
        opacity=1,
        legend_loc="upper left",
):
    
    ##############################################################
    ##############################################################
    ## below some lower size, don't plot #########################
    ##############################################################
    ##############################################################

    motif_list = motifs.motifs
    motif_list,top_n = set_up(motif_list, top_n)
    
    fig,ax = plt.subplots(ncols=1,nrows=top_n,figsize=(8.5,top_n*2),sharex=True)

    # pre-load images
    img_dict = {}
    offset_dict = {}
    just_a_motif = motif_list[0].motif
    shape_param_num = just_a_motif.shape[0]
    offsets = np.linspace(-0.35, 0.35, shape_param_num)
    for j in range(shape_param_num):
        shape_name = shape_lut[j]
        mark_fname = os.path.join(this_path,"img",shape_name+".png")
        img_arr = get_image(mark_fname)
        img_dict[shape_name] = img_arr
        offset_dict[shape_name] = offsets[j]

    max_weights = []
    uppers = []
    lowers = []
    for res in motif_list[:top_n]:
        max_weights.append(res.weights.max())
        lowers.append(res.motif.min())
        uppers.append(res.motif.max())
    w_max = np.max(max_weights)
    upper = np.max(uppers) + 0.75
    lower = np.max(lowers) - 0.75
    ylims = np.max(np.abs([upper, lower]))

    for i,res in enumerate(motif_list[:top_n]):

        if top_n == 1:
            this_ax = ax
        else:
            this_ax = ax[i]

        this_ax.axhline(y=0.0, color="black", linestyle="solid")
        this_ax.axhline(y=2.0, color="gray", linestyle="dashed")
        this_ax.axhline(y=4.0, color="gray", linestyle="dashed")
        this_ax.axhline(y=-2.0, color="gray", linestyle="dashed")
        this_ax.axhline(y=-4.0, color="gray", linestyle="dashed")
        mi = round(res.mi, 2)
        opt_y = res.motif
        norm_weights = res.weights / w_max
        
        x_vals = [i+1 for i in range(opt_y.shape[1])]
        
        for j in range(opt_y.shape[0]):

            shape_name = shape_lut[j]
            img_arr = img_dict[shape_name]
            j_offset = offset_dict[shape_name]
            j_opt = opt_y[j,:]
            j_w = norm_weights[j,:]

            for k in range(opt_y.shape[1]):

                x_pos = x_vals[k]
                weight = j_w[k]

                #if weight > 0.2:
                img = scale_image( img_arr, scale=weight )
                img.image.axes = this_ax
                ab = AnnotationBbox(
                    offsetbox = img,
                    xy = (x_pos,j_opt[k]),
                    # xybox and boxcoords together shift relative to xy
                    xybox = (j_offset*50, 0.0),
                    xycoords = "data",
                    boxcoords = "offset points",
                    frameon=False,
                )
                #print(f"x: {x_vals[k]}")
                #print(f"y: {opt_y[j,k]}")
                #print(f"dir-ab: {dir(ab)}")
                #print(f"ab xycoords: {ab.xycoords}")
                #print(f"dir-ab xycoords: {dir(ab.xycoords)}")
                #print(f"ab xycoords.center: {ab.xycoords.center()}")
                #print(f"ab boxcoords: {ab.boxcoords}")
                this_ax.add_artist( ab )
                #wind_ext = ab.get_window_extent()
                #tight_box = ab.get_tightbbox()
                #print(f"window_ext: {wind_ext}")
                #print(f"tight_bbox: {tight_box}")
                if j == 0:
                    if k % 2 == 0:
                        this_ax.axvspan(
                            x_vals[k]-0.5,
                            x_vals[k]+0.5,
                            facecolor = "0.2",
                            alpha=0.5,
                        )

        this_ax.set_ylim(bottom=-ylims, top=ylims)
        this_ax.text(1, 3, f"MI: {mi}")
        this_ax.set_ylabel(f"Shape value (z-score)")
        if i == 0:
            this_ax.set_title("Shape logo")
            this_ax.set_xticks(x_vals)
            this_ax.set_xlim(left=x_vals[0]-1, right=x_vals[-1]+1)

    
    if top_n == 1:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = ax[0].get_legend_handles_labels()
    this_ax.set_xlabel(f"Position (bp)")
    fig.legend(handles, labels, loc=legend_loc)
    #fig.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_shapes(rec_db, rec_idx, file_name, take_complement=False):

    shape_arr = rec_db.X[rec_idx,...]
    idx_shape_lut = {v:k for k,v in rec_db.shape_name_lut.items()}
    fig,ax = plt.subplots(nrows=2)
    for i in range(shape_arr.shape[1]):
        ax[0].plot(
            [j+1 for j in range(shape_arr.shape[0])],
            shape_arr[:,i,0],
            label = idx_shape_lut[i],
        )
        if take_complement:
            rev_shapes = shape_arr[::-1,i,1]
        else:
            rev_shapes = shape_arr[:,i,1]
        ax[1].plot(
            [j+1 for j in range(shape_arr.shape[0])],
            rev_shapes,
            label = idx_shape_lut[i],
        )
    plt.legend()
    plt.savefig(file_name)
    plt.close()
    

def set_up(motif_list, top_n):

    motif_list = sorted(motif_list, key=lambda x: x.mi, reverse=True)
    motif_num = len(motif_list)
    if top_n is None:
        top_n = len(motif_list)
    else:
        if motif_num < top_n:
            top_n = motif_num
    return(motif_list, top_n)


def plot_optim_trajectory(motif_list, file_name, top_n=20, opacity=1):

    motif_list,top_n = set_up(motif_list, top_n)

    fig,ax = plt.subplots()

    for i in range(top_n):
        this_info = motif_list[i]['opt_info']
        ax.plot(this_info['eval'], this_info['value'], alpha=opacity)

    plt.savefig(file_name)
    plt.close()


def heatmap(data, pvals, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Copied and pasted from https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    pvals
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    min_p = 1e-4
    clipped_pvals = np.clip(pvals, min_p, 1.0)
    alpha = (-np.log10(clipped_pvals) + 1) / (-np.log10(min_p) + 1)

    # Plot the heatmap
    im = ax.imshow(X=data, alpha=alpha, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Copied and pasted from https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_motif_enrichment(
        motifs,
        file_name,
        records,
        top_n = 30,
):
    '''Plot a heatmap of enrichments for each motif within each category.

    Args:
    -----
    motifs : inout.Motifs
    file_name : str
        Name of output file containing the heatmap
    records : inout.RecrodsDatabase
    max_count : int
        Maximum number of hits per strand
    top_n : int
        Limits number of motifs reported in heatmap to top_n
    '''

    distinct_cats = np.unique(records.y)
    cat_num = len(distinct_cats)

    var_lut = motifs.var_lut
    motif_covar_num = len(var_lut)
    
    hm_data = np.zeros((motif_covar_num,cat_num))
    hm_pvals = np.zeros((motif_covar_num,cat_num))
    hm_teststats = np.zeros((motif_covar_num,cat_num))
    row_labs = []
    for i,(covar_idx,covar_info) in enumerate(var_lut.items()):
        hits = covar_info["hits"]
        motif = motifs[covar_info["motif_idx"]]
        enrich = motif.enrichments
        for table_row_idx,row in enumerate(enrich["row_hit_vals"]):
            if np.all(row == hits):
                break
        row_labs.append(f"Motif: {motif.identifier}, Hit: {hits}")

        for j,category in enumerate(distinct_cats):
            table_col_idx = np.where(enrich["col_cat_vals"] == category)[0][0]
            # I need to map contingency table values back to heatmap row/col
            # I have already mapped hits and categories to the index of the
            # contingency table. Now I need to map them to indices of the heatmap.
            hm_data[i,j] = enrich["log2_ratio"][table_row_idx,table_col_idx]
            hm_pvals[i,j] = enrich["pvals"][table_row_idx,table_col_idx]
            hm_teststats[i,j] = enrich["test_stats"][table_row_idx,table_col_idx]
    col_labs = [f"Category: {int(records.category_lut[category]):d}" for category in distinct_cats]

    abs_max = np.abs(hm_data.max())
    abs_min = np.abs(hm_data.min())
    lim = np.array([abs_min, abs_max]).max()

    nrow = len(row_labs)
    ncol = len(col_labs)
    fig, ax = plt.subplots(figsize=(ncol*1.0, nrow*1.0))
    im,cbar = heatmap(
        hm_data,
        hm_pvals,
        row_labs,
        col_labs,
        ax=ax,
        cmap="bwr",
        cbarlabel="log2-fold-enrichment",
        vmin = -lim,
        vmax = lim,
    )
    texts = annotate_heatmap(im, valfmt="{x:.2f}", textcolors=("white","black"))

    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_optim_shapes_and_weights(
        motifs,
        file_name,
        records,
        top_n = 30,
        opacity=1,
        legend_loc="upper left",
):
    
    shape_lut = {v:k for k,v in records.shape_name_lut.items()}
    motif_list = motifs.motifs
    
    motif_list,top_n = set_up(motif_list, top_n)
    
    fig,ax = plt.subplots(ncols=2,nrows=top_n,figsize=(9,top_n*2),sharex=True)
    if top_n == 1:
        ax = ax[None,:]

    for i,res in enumerate(motif_list[:top_n]):

        mi = round(res.mi, 2)
        opt_y = res.motif
        weights = res.weights
        
        x_vals = [i+1 for i in range(opt_y.shape[1])]
        
        for j in range(opt_y.shape[0]):

            ax[i,0].plot(
                x_vals,
                opt_y[j,:],
                alpha = opacity,
                label = shape_lut[j],
            )
            ax[i,1].plot(
                x_vals,
                weights[j,:],
                alpha = opacity,
                label = shape_lut[j],
            )
        ax[i,0].text(1, 3, f"MI: {mi}")
        ax[i,0].set_ylabel(f"Index: {i}")
        if i == 0:
            ax[i,0].set_title("Optimized shapes")
            ax[i,1].set_title("Optimized weights")
            for j in range(2):
                ax[i,j].set_xticks(x_vals)
    
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc)
    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_shapes_and_weights(motif_list, file_name, records, alpha, top_n = 30, opacity=1, legend_loc="upper left"):
    
    shape_lut = {v:k for k,v in records.shape_name_lut.items()}
    
    motif_list,top_n = set_up(motif_list, top_n)
    
    fig,ax = plt.subplots(ncols=4,nrows=top_n,figsize=(16,top_n*2),sharex=True)
    if top_n == 1:
        ax = ax[None,:]

    for i,res in enumerate(motif_list[:top_n]):

        mi = round(res['mi'], 2)
        start_mi = round(res['mi_orig'], 2)
        opt_y = res['motif']
        orig_y = res['orig_shapes']
        weights = res['weights']
        orig_weights = res['orig_weights']
        weights = apply_weights_normalization(weights, float(alpha))
        orig_weights = apply_weights_normalization(orig_weights, float(alpha))
        
        x_vals = [i+1 for i in range(orig_y.shape[0])]
        
        for j in range(orig_y.shape[1]):

            ax[i,0].plot(
                x_vals,
                orig_y[:,j],
                alpha = opacity,
                label = shape_lut[j],
            )
            ax[i,1].plot(
                x_vals,
                opt_y[:,j],
                alpha = opacity,
                label = shape_lut[j],
            )
            ax[i,2].plot(
                x_vals,
                orig_weights[:,j],
                alpha = opacity,
                label = shape_lut[j],
            )
            ax[i,3].plot(
                x_vals,
                weights[:,j],
                alpha = opacity,
                label = shape_lut[j],
            )
        ax[i,0].text(1, 3, "MI: {}".format(mi))
        ax[i,0].text(1, 2, "Starting MI: {}".format(start_mi))
        ax[i,0].set_ylabel("Index: {}".format(i))
        if i == 0:
            ax[i,0].set_title("Original shapes")
            ax[i,1].set_title("Optimized shapes")
            ax[i,2].set_title("Original weights")
            ax[i,3].set_title("Optimized weights")
            for j in range(4):
                ax[i,j].set_xticks(x_vals)
    
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc)
    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()

def plot_motifs(motif_list, alpha, file_name, top_n=20):

    motif_list,top_n = set_up(motif_list, top_n)

    fig,ax = plt.subplots(
        ncols = 2,
        nrows = top_n,
        figsize = (8,top_n*4),
    )

    if top_n == 1:
        ax = ax[None,:]

    for i in range(top_n):
        weight_vals = motif_list[i]['weights'][...,0]
        norm_weights = apply_weights_normalization(weight_vals, alpha)
        ax[i,0].plot([i+1 for i in range(15)], motif_list[i]['motif'][...,0])
        ax[i,1].plot([i+1 for i in range(15)], norm_weights)

    plt.savefig(file_name)
    plt.close()

def apply_weights_normalization(weight_vals, alpha):
    trans_weights = alpha + (1-alpha) * inout.inv_logit(weight_vals)
    norm_weights = trans_weights / np.sum(trans_weights)
    return norm_weights
    

class MotifVis(inout.ShapeMotifFile):

    def plot_helical(self, motif, baseparams, lineparams, grid=None, vmin=None, vmax=None):
        import helical_wheel as hw
        fig = plt.gcf()
        if grid is None:
           total_grid = gs.GridSpec(1,1) 
           grid= total_grid[0]
        nested_gs = gs.GridSpecFromSubplotSpec(len(baseparams), 1, subplot_spec=grid, hspace=0.05)
        basepairs = len(motif['seed'])
        for motif_part in range(len(baseparams)):
            circs = []
            circ_vals=[]
            line_vals=[]
            for base in range(basepairs):
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
            this_ax.plot(list(range(len(mat[i,:]))), mat[i,:])
            this_ax.set_xticks(list(range(len(mat[i,:]))))
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
        this_ax.set_xticklabels(list(range(len(mat[i,:]))))
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
                this_logodds = im.two_way_to_log_odds(motif["enrichment"][val])
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
                this_logodds = im.two_way_to_log_odds(motif["enrichment"][val])
                motif_vals.append("%.4e"%this_logodds)
            all_lines.append(motif_vals)
        with open(outfile, mode="w") as outf:
            outf.write("#%s\n"%(self.get_title()))
            outf.write("mi\tentropy\t"+"\t".join([str(val) for val in sorted(motif["enrichment"].keys())]) +"\n")
            for line in all_lines:
                outf.write("\t".join(line)+"\n")

         
