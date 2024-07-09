import numpy as np
from scipy import spatial


def euclidean_dist_raw_data(v_arr, v_new):  # returns np array ([0, dist],[1, dist],[2, dist],...)
    sim_arr = np.empty((len(v_arr), 2))
    for index, row in enumerate(v_arr):
        sim_arr[index] = [index, np.linalg.norm(row - v_new)]

    # sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    return sim_arr


def chebyshev_dist_raw_data(v_arr, v_new): # returns np array ([0, dist],[1, dist],[2, dist],...)
    sim_arr = np.empty((len(v_arr), 2))
    for index, row in enumerate(v_arr):
        sim_arr[index] = [index, spatial.distance.chebyshev(row, v_new)]

    # sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    #print(sim_arr)
    return sim_arr


def cityblock_dist_raw_data(v_arr, v_new): # returns np array ([0, dist],[1, dist],[2, dist],...)
    sim_arr = np.empty((len(v_arr), 2))
    for index, row in enumerate(v_arr):
        sim_arr[index] = [index, spatial.distance.cityblock(row, v_new)]

    # sort by 1st col, then by 2nd col, source: https://stackoverflow.com/questions/29352511/numpy-sort-ndarray-on-multiple-columns
    sim_arr = sim_arr[np.lexsort((sim_arr[:, 0], sim_arr[:, 1]))]
    return sim_arr