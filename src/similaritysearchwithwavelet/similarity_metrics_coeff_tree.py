import numpy as np
from scipy import spatial


def euclidean_dist_coeff_tree (ref, list_coeff_trees, level):  # level corresponds to the level in the tree
    sim_arr = np.zeros((len(list_coeff_trees), 2))  # array with [index, sim]
    for index, coeff_tree in enumerate(list_coeff_trees):
        # if the vector contain only zeros, the sim cannot be calculated (ref[level] has been checked befor calling this function)
        if not np.any(coeff_tree[1][level]):  # coeff_tree[1] is the tree, coeff_tree[0] is the index of the row
            sim_arr[index] = [coeff_tree[0], np.nan]  # set sim to NaN if distance cannot be calculated
        else:
            sim_arr[index] =  [coeff_tree[0], np.linalg.norm(coeff_tree[1][level] - ref[level])]

    #sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    return sim_arr

def chebyshev_dist_coeff_tree (ref, list_coeff_trees, level): # level corresponds to the level in the tree
    sim_arr = np.zeros((len(list_coeff_trees), 2))  # array with [index, sim]
    for index, coeff_tree in enumerate(list_coeff_trees):
        # if the vector contain only zeros, the sim cannot be calculated (ref[level] has been checked befor calling this function)
        if not np.any(coeff_tree[1][level]):  # coeff_tree[1] is the tree, coeff_tree[0] is the index of the row
            sim_arr[index] = [coeff_tree[0], np.nan]  # set sim to NaN if distance cannot be calculated
        else:
            sim_arr[index] = [coeff_tree[0], spatial.distance.chebyshev(coeff_tree[1][level], ref[level])]

    #sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    return sim_arr


def cityblock_dist_coeff_tree (ref, list_coeff_trees, level): # level corresponds to the level in the tree
    sim_arr = np.zeros((len(list_coeff_trees), 2))  # array with [index, sim]
    for index, coeff_tree in enumerate(list_coeff_trees):
        # if the vector contain only zeros, the sim cannot be calculated (ref[level] has been checked befor calling this function)
        if not np.any(coeff_tree[1][level]):  # coeff_tree[1] is the tree, coeff_tree[0] is the index of the row
            sim_arr[index] = [coeff_tree[0], np.nan]  # set sim to NaN if distance cannot be calculated
        else:
            sim_arr[index] =  [coeff_tree[0], spatial.distance.cityblock(coeff_tree[1][level], ref[level])]

    #sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    return sim_arr