import os
import h5py
import numpy as np
from tqdm import tqdm
from commons import input_iterator

def merge(preprocessed_h,
          preprocessed_w,
          subject_ids,
          output_dir,
          tol=1e-10,
          batch_size=10000):
    """
    Stream hip/wrist windows in parallel, check timestamps and labels,
    merge into 6-channel windows, and write in batches to one HDF5.

    Outputs:
      merged_data.h5  with datasets:
        x           float32, shape (N,100,6)
        y           int32,   shape (N,)
        timestamp   float64, shape (N,)
        subject_id  utf-8 str, shape (N,)
      merge_warnings.log  listing any skipped pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    out_h5 = os.path.join(output_dir, 'merged_data.h5')
    log_fp = os.path.join(output_dir, 'merge_warnings.log')

    # open warning log
    log_f = open(log_fp, 'w')
    log_f.write("subject_id\treason\n")

    with h5py.File(out_h5, 'w') as h5f:
        # create extendable datasets with chunk size = batch_size
        x_ds = h5f.create_dataset(
            'x',
            shape=(0,100,6),
            maxshape=(None,100,6),
            dtype='float32',
            chunks=(batch_size,100,6)
        )
        y_ds = h5f.create_dataset(
            'y',
            shape=(0,),
            maxshape=(None,),
            dtype='int32',
            chunks=(batch_size,)
        )
        ts_ds = h5f.create_dataset(
            'timestamp',
            shape=(0,),
            maxshape=(None,),
            dtype='float64',
            chunks=(batch_size,)
        )
        str_dt = h5py.string_dtype(encoding='utf-8')
        sid_ds = h5f.create_dataset(
            'subject_id',
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=(batch_size,)
        )

        total = 0
        # in-memory buffers
        x_buf, y_buf, ts_buf, sid_buf = [], [], [], []

        def flush():
            nonlocal total
            n = len(x_buf)
            if n == 0:
                return
            # resize all at once
            x_ds.resize(total+n, axis=0)
            y_ds.resize(total+n, axis=0)
            ts_ds.resize(total+n, axis=0)
            sid_ds.resize(total+n, axis=0)

            # write buffers
            x_ds[total:total+n] = np.stack(x_buf)
            y_ds[total:total+n] = np.array(y_buf, dtype='int32')
            ts_ds[total:total+n] = np.array(ts_buf, dtype='float64')
            sid_ds[total:total+n] = np.array(sid_buf, dtype=str_dt)

            total += n
            x_buf.clear(); y_buf.clear(); ts_buf.clear(); sid_buf.clear()

        # main loop: subjects → parallel windows
        for subject_id in tqdm(subject_ids, desc='subjects'):
            hip_gen   = input_iterator(preprocessed_h, subject_id, train=True)
            wrist_gen = input_iterator(preprocessed_w, subject_id, train=True)

            for (x_h, ts_h, y_h), (x_w, ts_w, y_w) in zip(hip_gen, wrist_gen):
                reasons = []
                if not np.isclose(ts_h, ts_w, atol=tol):
                    reasons.append(f"timestamp mismatch (hip={ts_h}, wrist={ts_w})")
                if y_h != y_w:
                    reasons.append(f"label mismatch (hip={y_h}, wrist={y_w})")
                if reasons:
                    log_f.write(f"{subject_id}\t{'; '.join(reasons)}\n")
                    continue

                # merge channels
                merged = np.concatenate((x_h, x_w), axis=-1)
                x_buf.append(merged)
                y_buf.append(y_h)
                ts_buf.append(ts_h)
                sid_buf.append(subject_id)

                # flush once buffer is full
                if len(x_buf) >= batch_size:
                    flush()

        # final flush of leftovers
        flush()

    log_f.close()
    print(f"Merged {total} samples into {out_h5}")
    print(f"Any mismatches logged in {log_fp}")

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
