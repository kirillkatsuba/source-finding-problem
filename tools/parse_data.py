import numpy as np
import xarray as xr
from tqdm import tqdm


def parse_data(dataset_paths, dtype=np.float32):
    data = []
    source_coordinates = []

    for p in tqdm(dataset_paths):
        # открываем один раз
        with xr.open_dataset(p) as ds:
            conc = ds['CONC']
            min_lats = conc.attrs['MIN_LATS']
            min_longs = conc.attrs['MIN_LONGS']
            max_lats = conc.attrs['MAX_LATS']
            max_longs = conc.attrs['MAX_LONGS']

            n_sources = conc.shape[2]  # вместо «10» жёстко

            for source in range(n_sources):
                # берём срез
                subset = conc[:, 0, source, :, :].values  # .values вместо np.array()
                subset = subset.astype(dtype, copy=False)  # можно ужать до float32

                source_coordinates.append([
                    source,
                    min_lats[source],
                    min_longs[source],
                    max_lats[source],
                    max_longs[source],
                ])
                data.append(subset)

    data = np.stack(data)  # если все одного размера
    source_coordinates = np.array(source_coordinates)
    return data, source_coordinates
