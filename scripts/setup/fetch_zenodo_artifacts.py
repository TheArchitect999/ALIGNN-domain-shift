#!/usr/bin/env python3
"""Fetch archival artifacts from the project's Zenodo record.

Record 21398774 (DOI 10.5281/zenodo.21398774). The record's files are read from
the Zenodo API (so this script always matches what is actually deposited) and
each download is verified against the published MD5 checksum. Currently:
  - checkpoint_300.pt         pretrained ALIGNN checkpoint mirror
  - dataset_structures.tar.gz  dataset structure bundle

The fine-tuned and from-scratch checkpoints are intentionally NOT on Zenodo: they
are regenerable from the committed configs, fixed seeds, and training scripts
(see docs/REPRODUCING.md).

Usage:  python scripts/setup/fetch_zenodo_artifacts.py [--list]
Env:    ZENODO_ARTIFACT_DIR (destination dir; default zenodo_artifacts)
"""
import hashlib
import json
import os
import sys
import urllib.request

RECORD = "21398774"
DOI = "10.5281/zenodo.21398774"
API = f"https://zenodo.org/api/records/{RECORD}"
DEST = os.environ.get("ZENODO_ARTIFACT_DIR", "zenodo_artifacts")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_files():
    try:
        with urllib.request.urlopen(API, timeout=60) as r:
            rec = json.load(r)
    except Exception as e:
        sys.exit(f"cannot reach Zenodo record {RECORD} (DOI {DOI}): {e}\n"
                 "The record may not be published yet.")
    files = rec.get("files", [])
    if not files:
        sys.exit(f"record {RECORD} lists no files (not published yet?)")
    return files


def main():
    files = record_files()
    if "--list" in sys.argv[1:]:
        for f in files:
            print(f"  {f['key']:34s} {f.get('size', 0)/1e6:>10,.1f} MB")
        return
    os.makedirs(DEST, exist_ok=True)
    for f in files:
        key, url = f["key"], f["links"]["self"]
        want = f["checksum"].split(":")[-1]
        out = os.path.join(DEST, key)
        if os.path.exists(out) and md5(out) == want:
            print(f"ok (cached): {key}")
            continue
        print(f"downloading {key} ...")
        urllib.request.urlretrieve(url, out)
        if md5(out) != want:
            sys.exit(f"checksum mismatch for {key}")
        print(f"verified: {key}")
    print(f"done -> {DEST}/")


if __name__ == "__main__":
    main()
