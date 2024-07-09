import pywt
from load_SportsTables import *
from similarity_metrics_raw_data import *
from mapk import *

wavelet = pywt.Wavelet('haar')
metric_funcs = ['euclidean', 'chebyshev', 'cityblock']
metric_funcs_raw_data = [euclidean_dist_raw_data, chebyshev_dist_raw_data, cityblock_dist_raw_data]
raw_data = load_num_cols(10)
max_level = pywt.dwt_max_level(len(raw_data[0]), wavelet.name)

# parameters
k = 10
space_budget = round(len(raw_data[0])/2)

#mapk_results_all_k = np.empty((len(metric_funcs_raw_data)*len(space_budgets)*len(k_arr), len(raw_data)))
#mapk_results_all_k[:] = np.nan
#counter = 0
list_coeffs = []

# offline work
for i_raw_data, row in enumerate(raw_data):  # generate Haar-tree for each row
    coeffs = pywt.wavedec(row, wavelet.name)  # list with (cA, cD_1, cD_2, ...) cA, cD_1 are in maxlevel, cD_2 is maxlevel-1
    length = len(coeffs[-1])  # get length of last element of list => cD on max_level+1
    coeff_tree = np.empty([max_level+1, len(coeffs[-1])])  # array that stores Haar-tree -> here only needed to find lowest no.

    for j, coeff in enumerate(coeffs):
        coeff_ext = np.pad(coeff, (0, length - len(coeff)), 'constant', constant_values=0)  # fill with zeros
        coeff_tree[j] = coeff_ext

    # find lowest number that can be stored:
    coeff_tree_abs = np.abs(coeff_tree)
    coeff_tree_flat_ord = np.sort(coeff_tree_abs, axis=None)[::-1]  # order desc

    for i_coeff, coeff in enumerate(coeffs): # thresholding of the coeffs
        coeffs[i_coeff] = pywt.threshold(coeff, coeff_tree_flat_ord[space_budget-1], 'hard')

    list_coeffs.append(coeffs)

# online work
mapk_results_all_metrics = np.empty((len(metric_funcs_raw_data),len(raw_data))) # array that stores mAP scores

for i_metric, metric_func in enumerate(metric_funcs_raw_data):

    sim_indices_raw_data = []
    for row in raw_data:  # calculate metric of row to raw_data
        sim_indices_raw_data.append(metric_func(raw_data, row)[:, 0])  # [:,0] returns only indices

    reconstructed_data = []
    for data in list_coeffs:  # reconstruct the thresholded coeffs
        reconstructed_data.append(pywt.waverec(data, wavelet.name))

    sim_indices_reconstr_data = []
    for ref in reconstructed_data: #calculate top k for each reconstructed data
        sim_indices_reconstr_data.append(metric_func(reconstructed_data, ref)[:, 0])  # [:,0] returns only indices

    # perform mapk
    mapk_results = np.empty((len(raw_data)))

    for index, raw in enumerate(sim_indices_raw_data):
        actual = [sim_indices_raw_data[index].tolist()[:k]]  # extract top k, double brackets for mapk func
        predicted = [sim_indices_reconstr_data[index].tolist()[:k]]  # double brackets for mapk func
        predicted = [sim_indices_reconstr_data[index].tolist()[:k]]  # double brackets for mapk func
        mapk_results[index] = mapk(actual, predicted, k)

    mapk_results_all_metrics[i_metric] = mapk_results

df = pd.DataFrame(data=mapk_results_all_metrics.transpose(), columns=metric_funcs)
df.to_csv('out.csv', index=False)