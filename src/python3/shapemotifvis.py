import matplotlib as mpl
import os
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
import seaborn as sns
import infer_motifs as im
import numpy as np
import inout
import sys
from pathlib import Path
#from svgpathtools import svg2paths,parse_path
#from svgpath2mpl import parse_path

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import rpy2.robjects.lib.ggplot2 as ggplot
from rpy2.robjects.conversion import localconverter

#numpy2ri.activate()
base = importr("base")
ggseqlogo = importr("ggseqlogo")

this_path = Path(__file__).parent.absolute()

#plt.rc('figure', titlesize=10)

class ImageHandler(HandlerBase):
    def __init__(self, img, image_stretch = (0,0)):
        self.image_data = img
        self.image_stretch = image_stretch

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):

        sx, sy = self.image_stretch 

        # create a bounding box to house the image
        bb = Bbox.from_bounds(
            0,
            #xdescent - sx,
            0,
            #ydescent - sy,
            self.image_data.shape[1],
            #width + sx,
            self.image_data.shape[0],
            #height + sy,
        )

        #tbb = TransformedBbox(bb, trans)
        #image = BboxImage(tbb)
        image = BboxImage(bb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]



def get_image(path):
    #print(f"reading image in file {path}")
    img_arr = plt.imread(path, format="png")
    #print(f"read image in file {path}")
    return img_arr

#def get_svg(fname):
#    print(f"reading image in file {fname}")
#    paths,atts = svg2paths(fname)
#    #print(f"converting svg path: {paths}\nto mpl path")
#    #print("=====================================")
#    print(f"attributes: {atts}")
#    #sys.exit()
#    #img = parse_path(atts[0]["d"])
#    #print(f"read image in file {path}")
#    #return img
#    return paths[0]

def scale_image(img_arr, scale=1, frameon=False):
    # scale the alpha layer
    img_cp = img_arr.copy()
    img_cp[:,:,3] *= scale
    #img = OffsetImage(img_arr, zoom=scale*0.1)
    if frameon:
        img_cp[:,0:20,0:3] = 0
        img_cp[:,0:20,3] = 1
        img_cp[0:20,:,0:3] = 0
        img_cp[0:20,:,3] = 1
        img_cp[:,-20:-1,0:3] = 0
        img_cp[:,-20:-1,3] = 1
        img_cp[-20:-1,:,0:3] = 0
        img_cp[-20:-1,:,3] = 1
    img = OffsetImage(img_cp, zoom=1/30)
    return img

def set_plot_fname(motif, suffix):
    print(motif.alt_name)
    print(motif.identifier)
    if motif.alt_name is None:
        prefix = motif.identifier
    elif motif.alt_name == "None":
        prefix = motif.identifier
    else:
        prefix = motif.alt_name
    file_name = suffix.format(prefix)
    return file_name

def plot_seq_logo(motif, suffix):

    ro.r("""
        logo = function(mat, fname) {
            gp = ggseqlogo(mat) + lims(y=c(0,2)) + theme(axis.line=element_line(color="black"))
            ggsave(fname, gp, width=3.5, height=1)
        }
    """)
    logo = ro.globalenv["logo"]

    if motif.motif_type == "sequence":
        ppm = motif.ppm()
    file_name = set_plot_fname(motif, suffix)
    flat = list(ppm.flatten())
    with localconverter(ro.default_converter):
        ppm_r = ro.r.matrix(
            ro.FloatVector(flat),
            nrow=ppm.shape[1],
            ncol=ppm.shape[0],
        )
    ppm_r.rownames = ro.StrVector(["A","C","G","T"])
    logo(ppm_r, file_name)
    return file_name


def plot_seq_logos(motifs, suffix):

    seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()
    if len(seq_motifs) == 0:
        return []
    fnames = []
    for i,motif in enumerate(seq_motifs):
        fnames.append((plot_seq_logo(motif, suffix),motif))
    return fnames


def set_icons(motif, shape_lut):
    #print(f"Setting up to plot")
    img_dict = {}
    offset_dict = {}
    shape_param_num = motif.motif.shape[0]
    #print(f"Now at line 55")
    offsets = np.linspace(-0.35, 0.35, shape_param_num)
    #print(f"Now at line 57")
    #print("==============================================")
    #print(f"shape_lut: {shape_lut}")
    #print("==============================================")
    idx_shape_lut = {v:k for k,v in shape_lut.items()}
    for j in range(shape_param_num):
        #print(f"Now at line 59")
        #print(f"shape index {j}")
        shape_name = idx_shape_lut[j]
        #print(f"Setting up img_dict and offset_dict for {shape_name}")
        #mark_fname = os.path.join(this_path,"img",shape_name+".svg")
        mark_fname = os.path.join(this_path,"img",shape_name+".png")
        #img_path = get_svg(mark_fname)
        # center the icon
        #img_path.vertices -= img_path.vertices.mean(axis=0)
        img_arr = get_image(mark_fname)
        img_dict[shape_name] = img_arr
        #img_dict[shape_name] = img_path
        offset_dict[shape_name] = offsets[j]
    return (img_dict, offset_dict, idx_shape_lut)

def set_limits(motifs, top_n):
    max_weights = []
    uppers = []
    lowers = []
    for motif in motifs[:top_n]:
        max_weights.append(motif.weights.max())
        lowers.append(motif.motif.min())
        uppers.append(motif.motif.max())
    w_max = np.max(max_weights)
    upper = np.max(uppers) + 0.75
    lower = np.max(lowers) - 0.75
    ylims = np.max(np.abs([upper, lower]))
    return (w_max,upper,lower,ylims)

def plot_shape_logo(
        motif,
        suffix,
        idx_shape_lut,
        img_dict,
        offset_dict,
        w_max,
        upper,
        lower,
        ylims,
):
    file_name = set_plot_fname(motif, suffix)
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(8.5,2))
    ax.axhline(y=0.0, color="black", linestyle="solid", linewidth=1)
    ax.axhline(y=2.0, color="gray", linestyle="dashed", linewidth=1)
    ax.axhline(y=4.0, color="gray", linestyle="dashed", linewidth=1)
    ax.axhline(y=-2.0, color="gray", linestyle="dashed", linewidth=1)
    ax.axhline(y=-4.0, color="gray", linestyle="dashed", linewidth=1)
    mi = round(motif.mi, 2)
    opt_y = motif.motif
    norm_weights = motif.weights / w_max
    
    x_vals = [i+1 for i in range(opt_y.shape[1])]

    #legend_artists = []
    #legend_key = []

    for j in range(opt_y.shape[0]):

        shape_name = idx_shape_lut[j]
        #print(f"Working on shape {shape_name} for motif {i}")
        img_arr = img_dict[shape_name]
        j_offset = offset_dict[shape_name]
        j_opt = opt_y[j,:]
        j_w = norm_weights[j,:]

        for k in range(opt_y.shape[1]):

            x_pos = x_vals[k]
            weight = j_w[k]

            #ax.plot(x_pos, j_opt[k], marker=img_arr, markersize=40)

            #if weight > 0.2:
            #print(f"Setting image opacity")
            img = scale_image( img_arr, scale=weight, frameon=True )
            img.image.axes = ax
            ab = AnnotationBbox(
                offsetbox = img,
                xy = (x_pos,j_opt[k]),
                # xybox and boxcoords together shift relative to xy
                xybox = (j_offset*50, 0.0),
                xycoords = "data",
                boxcoords = "offset points",
                frameon=False,
            )
            ax.add_artist( ab )
            if j == 0:
                if k % 2 == 0:
                    ax.axvspan(
                        x_vals[k]-0.5,
                        x_vals[k]+0.5,
                        facecolor = "0.2",
                        alpha=0.2,
                    )
        #legend_artists.append(ab)
        #legend_key.append(shape_name)

    ax.set_ylim(bottom=-ylims, top=ylims)
    #ax.text(1, 3, f"MI: {mi}")
    ax.set_ylabel("Shape value (z-score)")
    ax.set_xticks(x_vals)
    ax.set_xlim(left=x_vals[0]-0.5, right=x_vals[-1]+0.5)
    ax.set_xlabel("Motif position")
    ###############################################################################
    ###############################################################################
    ## Here I need to figure out how to actually get a proxy artist for these images placed in the report
    ## Another possibility is to place them on the web page to the plots' right, not in the rendered plot per se.
    ###############################################################################
    ###############################################################################
    for i,(k,img_arr) in enumerate(img_dict.items()):
        img = scale_image( img_arr, scale=1.0, frameon=True )
        img.image.axes = ax
        ab = AnnotationBbox(
            offsetbox = img,
            xy = (1, 0),
            xycoords = "data",
            # xybox and boxcoords together shift relative to xy
            xybox = (1.03, (1.0-0.2*i)-0.08),
            boxcoords = "axes fraction",
            frameon=False,
        )
        ax.add_artist( ab )
        ax.annotate(k,
            xy = (1.05, (1.0-0.2*i)-0.08),
            xycoords='axes fraction',
            verticalalignment='center',
        )

    ##legend_keys = [k for k in img_dict.keys()]
    ##legend_keys = [k for k in img_dict.keys()]
    #a = plt.scatter([],[])
    #b = plt.scatter([],[])
    #c = plt.scatter([],[])
    #d = plt.scatter([],[])
    #e = plt.scatter([],[])
    #legend_map = {k:ImageHandler(v) for k,v in img_dict.items()}

    ##plt.legend([s, s2],
    ##       ['Scatters 1', 'Scatters 2'],
    ##       handler_map={s: custom_handler, s2: custom_handler},
    ##       labelspacing=2,
    ##       frameon=False)

    #ax.legend(
    #    [a,b,c,d,e], #[v for v in legend_map.values()],
    #    [k for k in legend_map.keys()],
    #    handler_map = {
    #        a: ImageHandler(img_dict["EP"]),
    #        b: ImageHandler(img_dict["HelT"]),
    #        c: ImageHandler(img_dict["MGW"]),
    #        d: ImageHandler(img_dict["ProT"]),
    #        e: ImageHandler(img_dict["Roll"]),
    #    },
    #    loc = "upper right",
    #    #bbox_to_anchor=(1.05, 1),
    #)
    #ax.legend(legend_vals, legend_keys, loc="upper left", bbox_to_anchor=(1.05, 1))
    #if i == 0:
    #    ax.set_title("Shape logo")
    #    ax.set_xticks(x_vals)
    #    ax.set_xlim(left=x_vals[0]-1, right=x_vals[-1]+1)

    #if top_n == 1:
    #    handles, labels = ax.get_legend_handles_labels()
    #else:
    #    handles, labels = ax[0].get_legend_handles_labels()
    ax.set_xlabel(f"Position (bp)")
    #fig.legend(handles, labels, loc=legend_loc)
    print(f"Writing shape logo to {file_name}")
    #fig.tight_layout()
    plt.savefig(file_name, dpi=600)
    plt.close()
    return file_name


def plot_shape_logos(motifs, suffix, shape_lut, top_n = 30, opacity=1, legend_loc="upper left"):

    #import ipdb; ipdb.set_trace()
    #print(motifs)
    seq_motifs,shape_motifs = motifs.split_seq_and_shape_motifs()
    #print(f"seq_motifs:\n{seq_motifs}")
    #print(f"shape_motifs:\n{shape_motifs}")
    #print(f"len(shape_motifs):\n{len(shape_motifs)}")
    if shape_motifs is None:
        return []
    if len(shape_motifs) == 0:
        return []

    # pre-load images
    (img_dict,offset_dict,idx_shape_lut) = set_icons(shape_motifs[0], shape_lut)
    (w_max,upper,lower,ylims) = set_limits(shape_motifs, top_n)

    fnames = []
    for i,motif in enumerate(shape_motifs):

        fnames.append((plot_shape_logo(
            motif,
            suffix,
            idx_shape_lut,
            img_dict,
            offset_dict,
            w_max,
            upper,
            lower,
            ylims,
        ),motif))

    return fnames


def plot_logos(
        motifs,
        suffix,
        shape_lut,
        top_n = 30,
        opacity=1,
        legend_loc="upper left",
):
    # plot the sequence logos into pngs
    fnames = plot_seq_logos(motifs, suffix)
    #print(f"file names after seq logos: {fnames}")
    #import ipdb; ipdb.set_trace()
    fnames.extend(plot_shape_logos(motifs, suffix, shape_lut, top_n, opacity, legend_loc))
    #print(f"file names after shape logos: {fnames}")
    return fnames

def plot_each_shape(rec_db, shape_names, rec_idx, file_name, take_complement=False):

    shape_arr = rec_db.X[rec_idx,...]
    idx_shape_lut = {v:k for k,v in rec_db.shape_name_lut.items()}
    fig,ax = plt.subplots(nrows=shape_arr.shape[1], sharex=True)
    for i in range(shape_arr.shape[1]):
        ax[i].plot(
            [j+1 for j in range(shape_arr.shape[0])],
            shape_arr[:,i,0],
            label = idx_shape_lut[i],
        )
        ax[i].set_ylabel(shape_names[i])
    plt.tight_layout()
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
            cbar_kw={}, cbarlabel="", robust=True, **kwargs):
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
    robust
        If true (default), set max and min vals to upper and lower 2-nd percentile
        found in data.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    min_p = 1e-4
    clipped_pvals = np.clip(pvals, min_p, 1.0)
    #alpha = (-np.log10(clipped_pvals) + 1) / (-np.log10(min_p) + 1)
    alpha = 1 - clipped_pvals

    if robust:
        minval = np.nanpercentile(data, 2)
        maxval = np.nanpercentile(data, 98)
        data[data < minval] = minval
        data[data > maxval] = maxval

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

def plot_motif_enrichment_seaborn(
        motifs,
        file_name,
        records=None,
        distinct_cats = None,
        top_n = 30,
        robust = True,
):
    if records is not None:
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
    if records is not None:
        col_labs = [f"Bin: {int(records.category_lut[category]):d}" for category in distinct_cats]
    else:
        col_labs = [f"Bin: {category:d}" for category in distinct_cats]

    abs_max = np.abs(hm_data.max())
    abs_min = np.abs(hm_data.min())
    lim = np.array([abs_min, abs_max]).max()

    if lim > 6:
        vmax = 6
        vmin = -6
    else:
        vmax = lim
        vmin = -lim

    nrow = len(row_labs)
    ncol = len(col_labs)
    fig, ax = plt.subplots(figsize=(ncol+5, nrow+1))

    sns.heatmap(
        hm_data,
        cmap="bwr",
        center=0.0,
        vmin = vmin,
        vmax = vmax,
        robust=robust,
        annot=True,
        fmt='.1f',
        annot_kws=None,
        linewidths=0,
        linecolor='white',
        cbar=True,
        cbar_kws={"label": "log2(motif enrichment)"},
        cbar_ax=None,
        square=False,
        xticklabels=col_labs,
        yticklabels=row_labs,
        mask=None,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_motif_enrichment(
        motifs,
        file_name,
        records,
        top_n = 30,
        robust = True,
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
        robust = robust,
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

         
