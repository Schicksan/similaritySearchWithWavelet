import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import cv2

# Returns a list of all columns with numeric data, a column is returned as sorted np arrays.
def load_all_num_cols():
    list_with_cols = []
    shortest = 0
    largest = 0

    directory = os.path.join('../../test_data')
    for root,dirs,files in os.walk(directory):
        for file in files:
            df = pd.read_csv(os.path.join('../../test_data', file), on_bad_lines='warn', lineterminator='\n', sep=',')
            for col_name in df.columns:
                if is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name].astype(float)
                    df[col_name] = df[col_name].fillna(0)  # replace NaN with Zero
                    col = np.asarray(df[col_name])
                    if not np.any(col):  # if the col contains only zeros, skip this one
                        continue
                    else:
                        col_sorted = np.sort(col)
                        list_with_cols.append(col_sorted)

                        # get largest amount of values per column (for upsampling)
                        if len(col) > largest:
                            largest = len(col)

    data_lake = []

    for col in list_with_cols:  # upsample arrays in data_lake to largest
        col_upsampled = cv2.resize(col, (1, largest), interpolation=cv2.INTER_NEAREST)
        data_lake.append(col_upsampled.flatten())

    return data_lake

# Returns a list of all columns with numeric data, a column is returned as sorted np arrays.
def load_num_cols(num_of_files):
    list_with_cols = []
    shortest = 0
    largest = 0
    counter = 0

    directory = os.path.join('../../test_data')
    for root,dirs,files in os.walk(directory):
        for file in files:
            if counter < num_of_files:
                df = pd.read_csv(os.path.join('../../test_data', file), on_bad_lines='warn', lineterminator='\n', sep=',')
                for col_name in df.columns:
                    if is_numeric_dtype(df[col_name]):
                        df[col_name] = df[col_name].astype(float)
                        df[col_name] = df[col_name].fillna(0)  # replace NaN with Zero
                        col = np.asarray(df[col_name])
                        if not np.any(col):  # if the col contains only zeros, skip this one
                            continue
                        else:
                            col_sorted = np.sort(col)
                            list_with_cols.append(col_sorted)

                            # get largest amount of values per column (for upsampling)
                            if len(col) > largest:
                                largest = len(col)
            counter += 1

    data_lake = []

    for col in list_with_cols:  # upsample arrays in data_lake to largest
        col_upsampled = cv2.resize(col, (1, largest), interpolation=cv2.INTER_NEAREST)
        data_lake.append(col_upsampled.flatten())

    return data_lake