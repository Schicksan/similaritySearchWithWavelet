import pywt
from load_SportsTables import *
from similarity_metrics_raw_data import *
from similarity_metrics_coeff_tree import *
from mapk import *

wavelet = pywt.Wavelet('haar')
metric_funcs = ['euclidean', 'chebyshev', 'cityblock']
metric_funcs_raw_data = [euclidean_dist_raw_data, chebyshev_dist_raw_data, cityblock_dist_raw_data]
metric_funcs_coeff_tree = [euclidean_dist_coeff_tree, chebyshev_dist_coeff_tree, cityblock_dist_coeff_tree]

raw_data = load_num_cols(10)
max_level = pywt.dwt_max_level(len(raw_data[0]), wavelet.name)

# parameters
coeff_tree_search_level = max_level+1  # round((max_level+1)/2)  # max_level+1 is the whole tree (+1 because the last cA has its own level)
k = 10
x_orig = 0
space_budget = round(len(raw_data[0])/2)

list_coeff_trees = []

# offline work
for i_raw_data, row in enumerate(raw_data):  # generate Haar-tree for each row
    coeffs = pywt.wavedec(row, wavelet.name)  # list with (cA, cD_1, cD_2, ...) cA, cD_1 are in maxlevel, cD_2 is maxlevel-1
    length = len(coeffs[-1])  # get length of last element of list => cD on max_level+1
    coeff_tree = np.empty([max_level+1, len(coeffs[-1])])  # array that stores Haar-tree

    for j, coeff in enumerate(coeffs):
        coeff_ext = np.pad(coeff, (0, length - len(coeff)), 'constant', constant_values=0)  # fill with zeros
        coeff_tree[j] = coeff_ext

    # find the lowest number that can be stored and threshold:
    coeff_tree_abs = np.abs(coeff_tree)
    coeff_tree_flat_ord = np.sort(coeff_tree_abs, axis=None)[::-1]  # order desc
    coeff_tree_thresholded = pywt.threshold(coeff_tree, coeff_tree_flat_ord[space_budget-1], 'hard')

    list_coeff_trees.append([i_raw_data, coeff_tree_thresholded])

# online work
mapk_results_all_metrics = np.empty((len(metric_funcs_raw_data), len(raw_data)))  # array that stores mAP scores

for i_metric, metric_func in enumerate(metric_funcs_raw_data):

    sim_indices_raw_data = []
    for row in raw_data:  # calculate metric of row to raw_data
        sim_indices_raw_data.append(metric_func(raw_data, row)[:, 0])  # [:,0] returns only indices

    sim_indices_coeff_top_k = []
    for coeff_tree in list_coeff_trees:  # calculate top k for each coeff tree

        ref = coeff_tree[1]  # [1] is the tree (coeff_tree[0] is the index)
        x = x_orig  # reset x to original value
        list_coeff_trees_tpm = list_coeff_trees
        indices_coeff_tree_top_k = []

        # for each level calculate top k+x, adapt list_coeff_trees_tpm with top k+x and go to the next level
        for level in range(coeff_tree_search_level):
            if level == coeff_tree_search_level-1:  # on last level set x to zero
                x = 0
            if not np.any(ref[level]):  # if the vectors contain only zeros, scip this level
                x = x/2
            else:
                sim_indices_coeff_ref = metric_funcs_coeff_tree[i_metric](ref, list_coeff_trees_tpm, level)
                # extract top (k+x)
                indices_coeff_tree_top_k = sim_indices_coeff_ref[:, 0][:round(k + x)]
                # build new list_coeff_trees with top k+x
                list_coeff_trees_tpm = [[u, v] for u, v in list_coeff_trees_tpm if u in indices_coeff_tree_top_k]
                x = x/2

        sim_indices_coeff_top_k.append(indices_coeff_tree_top_k)

    # perform mapk
    mapk_results = np.empty((len(raw_data)))

    for index, raw in enumerate(sim_indices_raw_data):
        actual = [sim_indices_raw_data[index].tolist()[:k]]  # extract top k, double brackets for mak func
        predicted = [sim_indices_coeff_top_k[index].tolist()[:k]]  # double brackets for mak func
        mapk_results[index] = mapk(actual, predicted, k)

    mapk_results_all_metrics[i_metric] = mapk_results

df = pd.DataFrame(data=mapk_results_all_metrics.transpose(), columns=metric_funcs)
df.to_csv('out.csv', index=False)
