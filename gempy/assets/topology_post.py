from typing import Iterable, Tuple
from nptyping import Array
import numpy as np


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

