import os
import h5py
import numpy as np
from tqdm import tqdm
from commons import input_iterator
import pickle

import os
import numpy as np
import h5py

def merge(h_f5, d_f5, subject_ids, output_dir, tol=1e-10):
    """
    Iterate over subject_ids, join hip (h_f5) and wrist (d_f5) by exact timestamp,
    merging their 3-channel windows into 6-channel windows.

    - h_f5, d_f5: paths to HDF5 files containing datasets:
        'x': (N, 100, 3), 'y': (N,), 'timestamp': (N,), 'subject_id': (N,)
    - subject_ids: list of subject_id strings to process
    - output_dir: directory where 'merged_data.h5' and 'merge_warnings.log' are created
    - tol: unused for integer timestamps (exact match required)

    Records any dropped windows in 'merge_warnings.log'.

    Returns the path to the merged HDF5 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_h5_path = os.path.join(output_dir, 'merged_data.h5')
    log_path = os.path.join(output_dir, 'merge_warnings.log')

    # Initialize log file
    with open(log_path, 'w') as log_f:
        log_f.write('subject_id\ttimestamp\treason\n')

    # Open input files and preload lightweight arrays
    with h5py.File(h_f5, 'r') as f_hip, h5py.File(d_f5, 'r') as f_wrist:
        hip_subj = f_hip['subject_id'][:]  # array of bytes or str
        wrist_subj = f_wrist['subject_id'][:]
        hip_ts = f_hip['timestamp'][:]
        wrist_ts = f_wrist['timestamp'][:]
        hip_x = f_hip['x']
        hip_y = f_hip['y']
        wrist_x = f_wrist['x']
        wrist_y = f_wrist['y']

        # Set up output HDF5 with extendable datasets
        with h5py.File(out_h5_path, 'w') as f_out:
            # merged x will have shape (None, 100, 6)
            merged_shape = (0, hip_x.shape[1], hip_x.shape[2] * 2)
            chunk_shape = (1000, hip_x.shape[1], hip_x.shape[2] * 2)
            x_out = f_out.create_dataset(
                'x', shape=merged_shape, maxshape=(None, 100, 6),
                chunks=chunk_shape, dtype=hip_x.dtype
            )
            y_out = f_out.create_dataset(
                'y', shape=(0,), maxshape=(None,),
                chunks=(1000,), dtype=hip_y.dtype
            )
            ts_out = f_out.create_dataset(
                'timestamp', shape=(0,), maxshape=(None,),
                chunks=(1000,), dtype=hip_ts.dtype
            )
            subj_out = f_out.create_dataset(
                'subject_id', shape=(0,), maxshape=(None,),
                chunks=(1000,), dtype=h5py.string_dtype(encoding='utf-8')
            )

            total = 0
            # Process each subject separately
            for subject_id in subject_ids:
                # Find indices for this subject
                hip_idx = np.where(hip_subj == subject_id)[0]
                wrist_idx = np.where(wrist_subj == subject_id)[0]
                if hip_idx.size == 0 or wrist_idx.size == 0:
                    with open(log_path, 'a') as log_f:
                        log_f.write(f"{subject_id}\t-\tno windows in one file\n")
                    continue

                # Build quick lookup for hip timestamps
                hip_ts_sub = hip_ts[hip_idx]
                hip_map = {ts: idx for ts, idx in zip(hip_ts_sub, hip_idx)}

                # Iterate wrist windows and merge when timestamp matches
                for w_i in wrist_idx:
                    ts = wrist_ts[w_i]
                    hip_i = hip_map.get(ts)
                    if hip_i is None:
                        with open(log_path, 'a') as log_f:
                            log_f.write(f"{subject_id}\t{ts}\tmissing counterpart\n")
                        continue

                    # Read data and check labels
                    x1 = hip_x[hip_i]
                    x2 = wrist_x[w_i]
                    y1 = hip_y[hip_i]
                    y2 = wrist_y[w_i]
                    if y1 != y2:
                        with open(log_path, 'a') as log_f:
                            log_f.write(f"{subject_id}\t{ts}\tlabel mismatch\n")
                        continue

                    merged_x = np.concatenate([x1, x2], axis=2)

                    # Resize and append to output
                    new_total = total + 1
                    x_out.resize((new_total, 100, 6))
                    y_out.resize((new_total,))
                    ts_out.resize((new_total,))
                    subj_out.resize((new_total,))

                    x_out[total] = merged_x
                    y_out[total] = y1
                    ts_out[total] = ts
                    subj_out[total] = subject_id

                    total = new_total
                    
    print('Done merging:', total, 'windows')
    return out_h5_path



if __name__ == "__main__":
    preprocessed_h = '/niddk-data-central/iWatch/pre_processed_pt/H'
    preprocessed_w = '/niddk-data-central/iWatch/pre_processed_pt/W'
    output_dir     = '/niddk-data-central/iWatch/pre_processed_seg/HW'

    with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
        split_data = pickle.load(f)
    train_subjects = split_data["test"]

    merge(preprocessed_h, preprocessed_w, train_subjects, output_dir)
    print('Done!')
