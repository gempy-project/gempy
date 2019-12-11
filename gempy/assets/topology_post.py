from typing import Iterable, Tuple
from nptyping import Array
import numpy as np
import matplotlib.pyplot as plt


def clean_unconf_edges(
        unconf_labels:Iterable[int], 
        edges:Array[int, ..., 2], 
        centroids:dict) -> Tuple[Array[int, ..., 2], dict]:
    """Clean unconformity topology edges and centroids. Needs to be run for
    each unconformity separately.
    
    Args:
        unconf_labels (Iterable[int]): Iterable of unconformity labels to be
            cleaned.
        edges (Array[int, ..., 2]): Edges [..., [n1, n2]]
        centroids (dict): Centroids dictionary.
    
    Returns:
        Tuple[Array[int, ..., 2], dict]: 
            [0] Cleaned topology edges array.
            [1] Cleaned topology centroids array with averaged centroid for
                resulting single unconformity label.
    """
    edges_clean = []
    centroids_clean = {}

    for k, v in centroids.items():
        if k in centroids_clean.keys():
            continue
        if k in unconf_labels[1:]:
            continue
            
        if k == unconf_labels[0]:
            centroids_clean[k] = np.average(
                [centroids[k] for k in unconf_labels], axis=0
            )
        else:
            centroids_clean[k] = v
    
    for e1, e2 in edges.astype(int):
        if e1 not in unconf_labels and e2 not in unconf_labels:
            edges_clean.append([e1, e2])
            continue
        if e1 in unconf_labels:
            e1 = unconf_labels[0]
        if e2 in unconf_labels:
            e2 = unconf_labels[0]
        if e1 == e2:
            continue
        edges_clean.append([e1, e2])
        
    return edges_clean, centroids_clean


def _plot_adj_matrix(
        geo_model, 
        adj_matrix, 
        adj_matrix_labels,
        adj_matrix_lith_labels, 
        n_liths,
        n_faults
    ):
    # ///////////////////////////////////////////////////////
    n = len(adj_matrix_labels)
    fig, ax = plt.subplots(figsize=(n // 2.5, n // 2.5))

    ax.imshow(adj_matrix, cmap="Greys", alpha=1)
    ax.set_xlim(-.5, n_liths * n_faults * 2 - 0.5)
    ax.set_ylim(-.5, n_liths * n_faults * 2 - 0.5)

    ax.set_title("Topology Adjacency Matrix")

    # ///////////////////////////////////////////////////////
    # lith tick labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(adj_matrix_lith_labels, rotation=0)
    ax.set_yticklabels(adj_matrix_lith_labels, rotation=0)
    
    # ///////////////////////////////////////////////////////
    # lith tick labels colors
    colors = list(geo_model.surfaces.colors.colordict.values())
    bboxkwargs = dict(
        edgecolor='none',
    )
    for xticklabel, yticklabel, l in zip(ax.xaxis.get_ticklabels(), 
                                         ax.yaxis.get_ticklabels(), 
                                         adj_matrix_labels):
        color = colors[l[0] - 1]
        xticklabel.set_bbox(
            dict(facecolor=color, **bboxkwargs)
        )
        xticklabel.set_color("white")
        yticklabel.set_bbox(
            dict(facecolor=color, **bboxkwargs)
        )
        yticklabel.set_color("white")

    # ///////////////////////////////////////////////////////
    # fault block tick labeling
    newax = fig.add_axes(ax.get_position())
    newax.patch.set_visible(False)

    newax.spines['bottom'].set_position(('outward', 29))
    newax.set_xlim(0, n_faults * 2)
    newax.set_xticks(np.arange(1, n_faults * 2 + 1) - 0.5)
    newax.set_xticklabels(["FB " + str(i + 1) for i in range(4)])

    newax.spines['left'].set_position(('outward', 25))
    newax.set_ylim(0, n_faults * 2)
    newax.set_yticks(np.arange(1, n_faults * 2 + 1) - 0.5)
    newax.set_yticklabels(["FB "+str(i + 1) for i in range(n_faults*2)][::-1])
    
    # ///////////////////////////////////////////////////////
    # (dotted) lines for fb's
    dlinekwargs = dict(
        color="black",
        linestyle="dashed",
        alpha=0.75,
        linewidth=1
    )
    linekwargs = dict(
        color="black", 
        linewidth=1
    )
    for i in range(0, n_faults * 2 + 1):
        pos = i * n_liths - .5

        if i != 0 and i != n_faults * 2:
            ax.axvline(pos, **dlinekwargs)
            ax.axhline(pos, **dlinekwargs)

        # solid spines outside to separate fbs
        line = ax.plot((-3.3, -.51), (pos, pos), **linekwargs)
        line[0].set_clip_on(False)

        line = ax.plot((pos, pos), (-3, -.51), **linekwargs)
        line[0].set_clip_on(False)
    # ///////////////////////////////////////////////////////
    return

