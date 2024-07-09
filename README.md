# Similarity Search With Wavelet
This code was developed as part of the bachelor thesis "Investigating the Potential of Wavelet Transforms as a Dataset Summary for Answering Data Discovery Queries" at the TU Berlin. 

## Description
With this code, we tested the effectiveness and efficiency of two different approaches to find similar datasets that have been compressed with the wavelet transform and stored in the Haar-Tree structure.

### Firts approach: Haar-Tree levels
We search for datasets that are similar to a reference dataset by calculating distance metrics between Haar-Tree levels.

### Second approach: reconstructed data
We reconstruct the data from the Haar-Trees and calculate distance metrics between the reconstructed data.

## Getting Started

### Dependencies
- Python 3.8.
- Numpy 1.24.4
- PyWavelets 1.4.1
- Pandas 2.0.3
- Scipy 1.10.0
- opencv-python 4.10.0.84
- scikit-learn 1.3.2

### Executing program
Run the file main_Haar_Tree.py to perform similarity searches with the Haar-Tree levels. The output will be a CSV file containing the mean average precision (mAP) scores. 
The results from similarity searches using the raw data serves as the benchmark for comparison.
The file main_reconstructed_data.py does the same, but with the reconstructed data. 

To measure the runtime of both approaches, use the files execution_time_Haar_Tree.py and execution_time_reconstructed_data.py.

### Version History
- 0.1
  - Initial Release
