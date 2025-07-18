import numpy as np
import re
from datetime import datetime
from joblib import Parallel, cpu_count, delayed
#import bfast
import utils.filemanager as fm
import utils.custom_bfast as bfast
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


###Some Functions
def _ndi(b1,b2):

    denom = b1 + b2        
    nom = (b1-b2)

    denom[denom==0] = 1e-8
    index = nom/denom  

    index[index>1] = 1
    index[index<-1] = -1
    
    return index

def _bsi(blue, red, nir, swir):

    numerator = (swir + red) - (nir + blue)
    denominator = (swir + red) + (nir + blue) + 1e-6  # Avoid divide-by-zero
    return numerator / denominator 



# Map dates to month numbers
def get_month_numbers(dates):
    return np.array([d.month for d in dates])

# Interpolate data for a single year

def interpolate_for_year(pixel_data, dates):
    month_numbers = get_month_numbers(dates)
    month_dict = defaultdict(list)

    # Fill dictionary: month → [values]
    for val, month in zip(pixel_data, month_numbers):
        if not np.isnan(val):
            month_dict[month].append(val)

    # Initialize output
    monthly_values = np.full(12, np.nan, dtype=np.float32)

    # Average for each month (or keep single value)
    for month, values in month_dict.items():
        monthly_values[month - 1] = np.mean(values)

    # Interpolate missing months
    if np.all(np.isnan(monthly_values)):
        return np.zeros(12, dtype=np.float16)
    valid = ~np.isnan(monthly_values)
    num_valid = np.sum(valid)

    if num_valid > 3:
        monthly_values = np.interp(np.arange(12), np.where(valid)[0], monthly_values[valid])
        return monthly_values.astype(np.float16)
    else:
        return np.zeros(12, dtype=np.float16)
    
        
'''    
def interpolate_for_year(pixel_data, dates):
    month_numbers = get_month_numbers(dates)
    valid_indices = ~np.isnan(pixel_data)
    valid_months = month_numbers[valid_indices]
    valid_data = pixel_data[valid_indices]
    target_months = np.arange(1, 13)
    if len(valid_data) == 0:
        return np.zeros(12)
    return np.interp(target_months, valid_months, valid_data).astype(np.float16)
'''
# Interpolate data for both years
def interpolate_time_series(pixel_data, dates_2018, dates_2019):
    pixel_data_2018 = pixel_data[:len(dates_2018)]
    pixel_data_2019 = pixel_data[len(dates_2018):]
    interpolated_2018 = interpolate_for_year(pixel_data_2018, dates_2018)
    interpolated_2019 = interpolate_for_year(pixel_data_2019, dates_2019)
    return np.concatenate([interpolated_2018, interpolated_2019]).astype(np.float16)

# Fusion of NDVI and BSI
def fuse_features(ndvi, bsi):
    return np.sqrt((ndvi ** 2 + bsi ** 2) / 2).astype(np.float16)
    
    
# Interpolate parallel processing 

def parallel_interpolate(feature_data, dates_2018, dates_2019, chunk_size=12056040, n_jobs=-1):
    height, width, _ = feature_data.shape
    flat_pixels = feature_data.reshape(-1, feature_data.shape[2])
    total = flat_pixels.shape[0]
    all_results = []

    for i in tqdm(range(0, total, chunk_size), desc="Chunked Interpolation"):
        chunk = flat_pixels[i:i+chunk_size]
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(interpolate_time_series)(px, dates_2018, dates_2019)
            for px in chunk
        )
        all_results.extend(results)

    interpolated = np.stack(all_results, axis=0).reshape(height, width, 24).astype(np.float16)
    return interpolated

   

# Parallel BFAST processing
def run_bfast_parallel(par_mngr, ts_2D, dates, freq, verbosity=0):
    step = max(len(ts_2D) // cpu_count(), 100)
    parallel_range = range(0, len(ts_2D), step)
    results = list(
        zip(
            *par_mngr(
                delayed(bfast.bfast_cci)(  # Call BFAST analysis
                    ts_2D[start: start + step].T,
                    dates,
                    h=(freq / 2) / (ts_2D.shape[1]),
                    verbosity=verbosity,
                )
                for start in parallel_range
            )
        )
    )
    return np.concatenate(results[0], axis=0), np.concatenate(results[1], axis=0)
                   