"""
Minimal test: verify the .mat file can be opened and print its contents.

Usage:
    python test_load.py --mat_path /path/to/data.mat
"""

import argparse
import h5py
import scipy.io

def test_load(mat_path):
    print(f"Trying scipy.io.loadmat ...")
    try:
        mat = scipy.io.loadmat(mat_path)
        keys = [k for k in mat.keys() if not k.startswith("_")]
        print(f"  OK — keys: {keys}")
        for k in keys:
            print(f"    {k}: shape={mat[k].shape}, dtype={mat[k].dtype}")
        return
    except NotImplementedError:
        print("  -> HDF5 format, trying h5py ...")

    with h5py.File(mat_path, "r") as f:
        print(f"  OK — keys: {list(f.keys())}")
        for k in f.keys():
            ds = f[k]
            if hasattr(ds, "shape"):
                print(f"    {k}: shape={ds.shape}, dtype={ds.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True)
    args = parser.parse_args()
    test_load(args.mat_path)
