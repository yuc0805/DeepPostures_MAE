import os
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from commons import input_iterator

def save_samples_from_iter(preprocessed_dir,
                           out_dir,
                           subject_ids,
                           window_size=42,
                           flush_threshold=1000):
    """
    Stream fixed-length windows from per-subject HDF5 files into one HDF5
    without overloading memory. We buffer up to `flush_threshold` windows in RAM,
    then write them at once.

    Outputs in out_dir/10s_train.h5:
      x           float32, shape (N_time, 100, 3)
      y           int32,   shape (N_time,)
      timestamp   float64, shape (N_time,)
      subject_id  utf-8 str, shape (N_time,)
    """
    #os.makedirs(out_dir, exist_ok=True)
    out_h5_path = out_dir#os.path.join(out_dir, '10s_val.h5')

    x_buf, y_buf, ts_buf, subj_buf = [], [], [], []
    first_write = True

    with h5py.File(out_h5_path, 'w') as f_out:
        for subject_id in tqdm(subject_ids, desc='Subjects'):
            subject_dir = os.path.join(preprocessed_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue

            for x_seq, ts_seq, y_seq in input_iterator(preprocessed_dir,
                                                       subject_id,
                                                       train=True):
                x_win, y_win, ts_win = [], [], []
                for x, ts, y in zip(x_seq, ts_seq, y_seq):
                    x_win.append(x)
                    y_win.append(y)
                    ts_win.append(ts)

                    if len(y_win) == window_size:
                        # buffer one window of shape (window_size, 100, 3)
                        x_buf.append(np.stack(x_win, axis=0).astype(np.float32))
                        y_buf.append(np.array(y_win, dtype=np.int32))
                        ts_buf.append(np.array(ts_win, dtype=np.float64))
                        subj_buf.append(subject_id)

                        x_win.clear()
                        y_win.clear()
                        ts_win.clear()

                    if len(y_buf) >= flush_threshold:
                        _flush_to_h5(f_out, x_buf, y_buf, ts_buf, subj_buf, first_write)
                        first_write = False
                        x_buf.clear(); y_buf.clear(); ts_buf.clear(); subj_buf.clear()

        # final flush
        if y_buf:
            _flush_to_h5(f_out, x_buf, y_buf, ts_buf, subj_buf, first_write)


def _flush_to_h5(f_out, x_list, y_list, ts_list, subj_list, first_write):
    """
    Write data as (BS, window_size, 100, 3) without flattening.
    """
    x_arr = np.stack(x_list, axis=0)   # (BS, window_size, 100, 3)
    y_arr = np.stack(y_list, axis=0)   # (BS, window_size)
    ts_arr = np.stack(ts_list, axis=0) # (BS, window_size)
    subj_arr = np.array(subj_list, dtype=h5py.string_dtype(encoding='utf-8'))  # (BS,)

    if first_write:
        f_out.create_dataset(
            'x',
            data=x_arr,
            maxshape=(None,) + x_arr.shape[1:],
            chunks=(min(100, x_arr.shape[0]),) + x_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'y',
            data=y_arr,
            maxshape=(None,) + y_arr.shape[1:],
            chunks=(min(100, y_arr.shape[0]),) + y_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'timestamp',
            data=ts_arr,
            maxshape=(None,) + ts_arr.shape[1:],
            chunks=(min(100, ts_arr.shape[0]),) + ts_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'subject_id',
            data=subj_arr,
            maxshape=(None,),
            chunks=(min(100, subj_arr.shape[0]),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            compression='gzip'
        )
    else:
        for name, arr in zip(['x', 'y', 'timestamp', 'subject_id'],
                             [x_arr, y_arr, ts_arr, subj_arr]):
            ds = f_out[name]
            old = ds.shape[0]
            new = old + arr.shape[0]
            ds.resize((new,) + ds.shape[1:])
            ds[old:new] = arr


# def _flush_to_h5(f_out, x_list, y_list, ts_list, subj_list, first_write):
#     """
#     Flatten windows along the time axis and write to HDF5 so that
#     """
#     # concatenate along the time axis
#     x_arr = np.concatenate(x_list, axis=0)   # (sum(window), 100, 3)
#     y_arr = np.concatenate(y_list, axis=0)   # (sum(window),)
#     ts_arr = np.concatenate(ts_list, axis=0) # (sum(window),)
#     subj_arr = np.array(
#         [sid for sid, arr in zip(subj_list, x_list) for _ in range(arr.shape[0])],
#         dtype=h5py.string_dtype(encoding='utf-8')
#     )

#     if first_write:
#         f_out.create_dataset(
#             'x',
#             data=x_arr,
#             maxshape=(None,) + x_arr.shape[1:],
#             chunks=(min(1000, x_arr.shape[0]),) + x_arr.shape[1:],
#             compression='gzip'
#         )
#         f_out.create_dataset(
#             'y',
#             data=y_arr,
#             maxshape=(None,),
#             chunks=(min(1000, y_arr.shape[0]),),
#             compression='gzip'
#         )
#         f_out.create_dataset(
#             'timestamp',
#             data=ts_arr,
#             maxshape=(None,),
#             chunks=(min(1000, ts_arr.shape[0]),),
#             compression='gzip'
#         )
#         f_out.create_dataset(
#             'subject_id',
#             data=subj_arr,
#             maxshape=(None,),
#             chunks=(min(1000, subj_arr.shape[0]),),
#             dtype=h5py.string_dtype(encoding='utf-8'),
#             compression='gzip'
#         )
#     else:
#         for name, arr in zip(['x','y','timestamp','subject_id'],
#                              [x_arr, y_arr, ts_arr, subj_arr]):
#             ds = f_out[name]
#             old = ds.shape[0]
#             new = old + arr.shape[0]
#             ds.resize(new, axis=0)
#             ds[old:new] = arr


if __name__ == "__main__":
    split_data_file = "/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl"
    pre_processed_dir = '/niddk-data-central/iWatch/pre_processed_pt/W'

    with open(split_data_file, "rb") as f:
        split_data = pickle.load(f)
        train_subjects = split_data["train"]
        valid_subjects = split_data["val"]
        test_subjects  = split_data["test"]

    # write out one HDF5 per split, flattened along the time axis
    save_samples_from_iter(pre_processed_dir,
                           "/niddk-data-central/iWatch/pre_processed_long_seg/W/10s_val.h5",
                           valid_subjects,
                           window_size=42,
                           flush_threshold=1000)

    save_samples_from_iter(pre_processed_dir,
                           "/niddk-data-central/iWatch/pre_processed_long_seg/W/10s_train.h5",
                           train_subjects,
                           window_size=42,
                           flush_threshold=1000)

    save_samples_from_iter(pre_processed_dir,
                           "/niddk-data-central/iWatch/pre_processed_long_seg/W/10s_test.h5",
                           test_subjects,
                           window_size=42,
                           flush_threshold=1000)

    print("Done!")
