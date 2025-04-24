import os
from tqdm import tqdm
from commons import input_iterator
import pickle

import os
import h5py
import numpy as np
from tqdm import tqdm

def merge(preprocessed_h, preprocessed_w, subject_ids, output_dir, tol=1e-6):
    """
    Iterate hip and wrist data in parallel, verify timestamps match,
    merge 3-channel windows into 6-channel samples, and save into one HDF5 file
    with datasets: x, y, timestamp, subject_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'merged_data.h5')

    with h5py.File(output_file, 'w') as h5f:
        x_ds = h5f.create_dataset('x', shape=(0,100,6), maxshape=(None,100,6), dtype='float32', chunks=(1,100,6))
        y_ds = h5f.create_dataset('y', shape=(0,),        maxshape=(None,),       dtype='int32',   chunks=(1,))
        ts_ds = h5f.create_dataset('timestamp', shape=(0,),maxshape=(None,),       dtype='float64', chunks=(1,))
        str_dt = h5py.string_dtype(encoding='utf-8')
        sid_ds = h5f.create_dataset('subject_id', shape=(0,), maxshape=(None,),   dtype=str_dt,     chunks=(1,))

        total = 0
        for subject_id in tqdm(subject_ids, desc='subjects'):
            hip_gen   = input_iterator(preprocessed_h, subject_id, train=True)
            wrist_gen = input_iterator(preprocessed_w, subject_id, train=True)

            for (x_h, ts_h, y_h), (x_w, ts_w, y_w) in zip(hip_gen, wrist_gen):
                if not np.isclose(ts_h, ts_w, atol=tol): #TODO: also check if y_h == y_w
                    print(f"Warning: timestamp mismatch for {subject_id}: hip {ts_h}, wrist {ts_w} – skipping")
                    #TODO: output a log to record it.
                    continue

                merged_x = np.concatenate((x_h, x_w), axis=-1)

                x_ds.resize(total+1, axis=0); x_ds[total] = merged_x
                y_ds.resize(total+1, axis=0); y_ds[total] = y_h
                ts_ds.resize(total+1, axis=0); ts_ds[total] = ts_h
                sid_ds.resize(total+1, axis=0); sid_ds[total] = subject_id

                total += 1

    print(f"Merged {total} samples saved to {output_file}")



           

if __name__ == "__main__":
    preprocessed_h = '/niddk-data-central/iWatch/pre_processed_pt/H'
    preprocessed_w = '/niddk-data-central/iWatch/pre_processed_pt/W'
    output_dir = '/niddk-data-central/iWatch/pre_processed_seg/HW'

    os.makedirs(output_dir, exist_ok=True)
    split_data_file = "/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl"
    with open(split_data_file, "rb") as f:
        split_data = pickle.load(f)

        train_subjects = split_data["train"]
        # valid_subjects = split_data["val"]
        # test_subjects = split_data["test"]

    merge(preprocessed_h, preprocessed_w, train_subjects, output_dir)






    pass
