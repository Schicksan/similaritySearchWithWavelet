import pywt
from load_SportsTables import *
from similarity_metrics_raw_data import *
from similarity_metrics_coeff_tree import *
from mapk import *
from time import time

wavelet = pywt.Wavelet('haar')
metric_func_raw_data = euclidean_dist_raw_data
metric_func_coeff_tree = euclidean_dist_coeff_tree

raw_data = load_num_cols(10)
max_level = pywt.dwt_max_level(len(raw_data[0]), wavelet.name)

# parameters
coeff_tree_search_level = 1
k = 10
x = k
space_budget = round(len(raw_data[0])/2)

t0 = time()

# offline work
list_coeff_trees = []
for i_raw_data, row in enumerate(raw_data):  # generate Haar-tree for each row
    coeffs = pywt.wavedec(row, wavelet.name)  # list with (cA, cD_1, cD_2, ...) cA, cD_1 is the max_level, cD_2 is maxlevel-1
    length = len(coeffs[-1])  # get length of last element of list => cD on max_level+1
    coeff_tree = np.empty([max_level+1, len(coeffs[-1])])  # array that stores Haar-tree

    for j, coeff in enumerate(coeffs):
        coeff_ext = np.pad(coeff, (0, length - len(coeff)), 'constant', constant_values=0)  # fill with zeros
        coeff_tree[j] = coeff_ext

    # find lowest number that can be stored and threshold:
    coeff_tree_abs = np.abs(coeff_tree)
    coeff_tree_flat_ord = np.sort(coeff_tree_abs, axis=None)[::-1]  # order desc
    coeff_tree_thresholded = pywt.threshold(coeff_tree, coeff_tree_flat_ord[space_budget-1], 'hard')

    list_coeff_trees.append([i_raw_data, coeff_tree_thresholded])

t1 = time()

# online work
ref = list_coeff_trees[2][1] # [1] is the tree ([2] is the index)
list_coeff_trees_tpm = list_coeff_trees
indices_coeff_tree_top_k = []

# for each level calculate top k+x, adapt list_coeff_trees_tpm with top k*(1+x) and go to the next level
for level in range(coeff_tree_search_level):
    if level == coeff_tree_search_level-1: # on last level set x to zero
        x = 0
    if not np.any(ref[level]):  # if the vectors contain only zeros, scip this level
        x = x/2
    else:
        sim_indices_coeff_ref = metric_func_coeff_tree(ref, list_coeff_trees_tpm, level) # result is sorted by score
        # extract top k+x
        indices_coeff_tree_top_k = sim_indices_coeff_ref[:, 0][:round(k + x)]
        # build new list_coeff_trees with top k+x
        list_coeff_trees_tpm = [[u,v] for u,v in list_coeff_trees_tpm if u in indices_coeff_tree_top_k]
        x = x/2

print("result of similarity search:")
print(indices_coeff_tree_top_k)

t2 = time()

str_offline = 'offline %f' %(t1-t0)
str_online = ' online %f' %(t2-t1)

print(str_offline)
print(str_online)
