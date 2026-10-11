"""Download a per-sign subset of the Kaggle asl-signs (PopSign) landmark files.

The full set is about 57 GB, so this reads the index (train.csv) and fetches
only the parquet files for the requested signs, spread evenly across signers.
Needs `kaggle auth login` and the competition rules accepted on kaggle.com.

Run (from the repo root, inside .venv):
    python fetch_popsign.py hello yes no thankyou [--per-sign 100]
"""

import argparse
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

OUT_DIR = "data/popsign"
KAGGLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv/bin/kaggle")


def pick(index, sign, n, seed=0):
    """Up to n rows for sign, taken round-robin across signers so none dominates."""
    rows = index[index.sign == sign].sample(frac=1, random_state=seed)
    rows = rows.assign(rank=rows.groupby("participant_id").cumcount())
    return rows.sort_values(["rank", "participant_id"]).head(n)


def fetch(path):
    dest = os.path.join(OUT_DIR, path)
    if os.path.exists(dest):
        return "cached"
    folder = os.path.dirname(dest)
    os.makedirs(folder, exist_ok=True)
    subprocess.run([KAGGLE, "competitions", "download", "-c", "asl-signs", "-f", path, "-p", folder, "-q"],
                   check=True, capture_output=True)
    zipped = dest + ".zip"
    if os.path.exists(zipped):
        with zipfile.ZipFile(zipped) as z:
            z.extract(os.path.basename(path), folder)
        os.remove(zipped)
    return "downloaded"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("signs", nargs="+", help="PopSign sign names, lower case (e.g. hello thankyou)")
    p.add_argument("--per-sign", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    index = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
    chosen = []
    for sign in args.signs:
        rows = pick(index, sign, args.per_sign)
        if rows.empty:
            print(f"{sign}: not in PopSign, skipped")
            continue
        print(f"{sign}: {len(rows)} sequences from {rows.participant_id.nunique()} signers")
        chosen.append(rows)
    chosen = pd.concat(chosen)

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(fetch, chosen.path))
    print(f"{results.count('downloaded')} downloaded, {results.count('cached')} already present")
    chosen.drop(columns="rank").to_csv(os.path.join(OUT_DIR, "subset.csv"), index=False)


if __name__ == "__main__":
    main()
