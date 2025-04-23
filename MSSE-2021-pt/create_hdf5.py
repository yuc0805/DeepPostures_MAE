import os
import pickle
import h5py
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Compatibility for old NumPy pickles ---
class LegacyNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("_core", "core")
        if module == "numpy.core.multiarray":
            import numpy.core.multiarray
            return getattr(numpy.core.multiarray, name)
        return super().find_class(module, name)

# --- Collect all pickle paths ---
def collect_pickle_paths(root_dir):
    pickle_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.pkl'):
                pickle_paths.append(os.path.join(dirpath, fname))
    return pickle_paths

# --- Load one pickle with legacy compatibility ---
def load_pickle(path):
    with open(path, 'rb') as f:
        data = LegacyNumpyUnpickler(f).load()
    return data['x'], data['y']

# --- Batched conversion ---
def convert_pickles_to_hdf5(pickle_paths, hdf5_path, batch_size=1000):
    num_samples = len(pickle_paths)

    # Estimate shape and dtype
    first_x, first_y = load_pickle(pickle_paths[0])
    x_shape = first_x.shape
    y_shape = first_y.shape
    x_dtype = first_x.dtype
    y_dtype = first_y.dtype

    with h5py.File(hdf5_path, 'w') as h5f:
        x_dataset = h5f.create_dataset('x', shape=(num_samples, *x_shape), dtype=x_dtype)
        y_dataset = h5f.create_dataset('y', shape=(num_samples, *y_shape), dtype=y_dtype)

        print(f"Processing {num_samples} pickle files in batches of {batch_size}...")
        start_time = time.time()

        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Total Progress"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_paths = pickle_paths[batch_start:batch_end]

            with Pool(processes=cpu_count()) as pool:
                results = pool.map(load_pickle, batch_paths)

            for i, (x, y) in enumerate(results):
                x_dataset[batch_start + i] = x
                y_dataset[batch_start + i] = y

        total_elapsed = time.time() - start_time
        print(f"\nCompleted writing HDF5 in {total_elapsed:.2f} seconds.")

# --- Example usage ---
if __name__ == '__main__':
    root_directory = '/niddk-data-central/iWatch/pre_processed_seg/W/train'
    output_hdf5_file = '/niddk-data-central/iWatch/pre_processed_seg/W/train.hdf5'
    pickle_file_paths = collect_pickle_paths(root_directory)
    convert_pickles_to_hdf5(pickle_file_paths, output_hdf5_file, batch_size=1000)
