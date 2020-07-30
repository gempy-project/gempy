import numpy as np


def compute_prob(blocks):
    blocks = np.round(blocks)
    lith_id = np.unique(blocks)
    # lith_count = np.zeros_like(lith_blocks[0:len(lith_id)])
    count = np.zeros((len(np.unique(blocks)), blocks.shape[1]))
    for i, l_id in enumerate(lith_id):
        count[i] = np.sum(blocks == l_id, axis=0)
    prob = count / len(blocks)
    return prob


def calculate_ie_masked(prob):
    ie = np.zeros_like(prob[0])
    for l in prob:
        pm = np.ma.masked_equal(l, 0)  # mask where prob is 0
        ie -= (pm * np.ma.log2(pm)).filled(0)
    return ie