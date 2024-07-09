import pywt
from load_SportsTables import *
from similarity_metrics_raw_data import *
from mapk import *
from time import time

wavelet = pywt.Wavelet('haar')
metric_funcs_raw_data = euclidean_dist_raw_data
raw_data = load_num_cols(10)
max_level = pywt.dwt_max_level(len(raw_data[0]), wavelet.name)

# parameters
k = 10
space_budget = round(len(raw_data[0])/2)

t0 = time()

# offline work
list_coeffs = []
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

t1 = time()

# online work
reconstructed_data = []
for data in list_coeffs:  # reconstruct the thresholded coeffs
    reconstructed_data.append(pywt.waverec(data, wavelet.name))

ref = reconstructed_data[2]
sim_indices_reconstr_data = metric_funcs_raw_data(reconstructed_data, ref)[:, 0]  # [:,0] returns only indices, result is sorted by score

print("result of similarity search:")
print(sim_indices_reconstr_data[:k])

t2 = time()

str_offline = 'offline %f' %(t1-t0)
str_online = ' online %f' %(t2-t1)

print(str_offline)
print(str_online)


