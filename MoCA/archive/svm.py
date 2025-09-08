import argparse
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from util.datasets import iWatch
from tqdm import tqdm

def process_batch(x, y):
    x = x.numpy()
    y = y.numpy()
    Xb, yb = [], []
    for i in range(x.shape[0]):
        xi = x[i].reshape(x.shape[1], -1)
        Xb.append(xi.mean(axis=0))
        yb.append(np.bincount(y[i]).argmax())
    return np.stack(Xb), np.array(yb)

def stream_dataset(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for x, y, _ in loader:
        yield process_batch(x, y)

def evaluate(model, dataset, batch_size):
    y_true, y_pred = [], []
    for Xb, yb in stream_dataset(dataset, batch_size):
        yp = model.predict(Xb)
        y_true.append(yb)
        y_pred.append(yp)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def main(args):
    train_ds = iWatch(set_type='train', root=args.data_path, subset_ratio=args.subset_ratio)
    val_ds = iWatch(set_type='val', root=args.data_path)
    test_ds = iWatch(set_type='test_complete', root=args.data_path)

    model = SGDClassifier(loss='huber',  # correct loss name
                          learning_rate='optimal',
                          eta0=0.01,
                          tol=None)

    print("Starting training...")
    for epoch in range(args.epochs):
        train_gen = stream_dataset(train_ds, args.batch_size)
        with tqdm(total=len(train_ds) // args.batch_size, desc=f"Epoch {epoch+1}") as pbar:
            for Xb, yb in train_gen:
                model.partial_fit(Xb, yb, classes=np.array([0,1]))
                pbar.update(1)

        val_acc, val_f1 = evaluate(model, val_ds, args.batch_size)
        print(f"[Epoch {epoch+1}] Val Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")

    print("Final evaluation on test set:")
    test_acc, test_f1 = evaluate(model, test_ds, args.batch_size)
    print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    main(args)

'''
python svm.py --data_path "/niddk-data-central/iWatch/pre_processed_long_seg/H"
'''