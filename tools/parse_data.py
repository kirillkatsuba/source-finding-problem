import numpy as np
import xarray as xr
from tqdm import tqdm


def parse_data(dataset_paths):
    data = []
    source_coordinates = []
    for p in tqdm(dataset_paths):
        for source in range(10):
            subset = xr.open_dataset(p)['CONC'][:, 0, source, :, :]
            source_MIN_LATS = xr.open_dataset(p)['CONC'].attrs['MIN_LATS'][source]
            source_MIN_LONGS = xr.open_dataset(p)['CONC'].attrs['MIN_LONGS'][source]
            source_MAX_LATS = xr.open_dataset(p)['CONC'].attrs['MAX_LATS'][source]
            source_MAX_LONGS = xr.open_dataset(p)['CONC'].attrs['MAX_LONGS'][source]
            subset = np.array(subset)
            source_coordinates.append([
                source,
                source_MIN_LATS,
                source_MIN_LONGS,
                source_MAX_LATS,
                source_MAX_LONGS
                ])
            data.append(subset)
    return np.array(data), np.array(source_coordinates)
