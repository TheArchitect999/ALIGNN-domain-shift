#!/usr/bin/env python3
"""Fetch the JARVIS-DFT ALIGNN v1 formation-energy pretrained checkpoint.

Primary download: the official Figshare archive.  The reserved Zenodo record is
used only as a mirror after it becomes public.
Provenance: this checkpoint is a mirror of Kamal Choudhary's public JARVIS-DFT
ALIGNN v1 formation-energy checkpoint (figshare record 17005681,
https://doi.org/10.6084/m9.figshare.17005681.v1); the original model terms apply.
The download is verified against a provenance-pinned SHA-256.

Usage:  python scripts/setup/fetch_pretrained.py
Env:    ALIGNN_CKPT_DIR (destination dir; default models/pretrained)
"""
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

FNAME = "checkpoint_300.pt"
SHA256 = "bce5cdafa06dc26ad8ddb3ceeb2bef7593c218dd66825e7cb5381c156317458f"
OFFICIAL_DOI = "https://doi.org/10.6084/m9.figshare.17005681.v1"
OFFICIAL_ARCHIVE_URL = "https://ndownloader.figshare.com/files/31458679"
ZENODO_MIRROR_URL = "https://zenodo.org/records/21398774/files/checkpoint_300.pt?download=1"
DEST = os.environ.get("ALIGNN_CKPT_DIR", "models/pretrained")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    out = os.path.join(DEST, FNAME)
    if os.path.exists(out) and sha256(out) == SHA256:
        print(f"already present and verified: {out}")
        return
    errors = []
    print(f"downloading official pretrained archive:\n  {OFFICIAL_ARCHIVE_URL}")
    try:
        with tempfile.TemporaryDirectory(prefix="alignn-pretrained-") as tmp:
            archive = os.path.join(tmp, "official_checkpoint.zip")
            urllib.request.urlretrieve(OFFICIAL_ARCHIVE_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                members = [name for name in zf.namelist()
                           if os.path.basename(name) == FNAME]
                if len(members) != 1:
                    raise RuntimeError(
                        f"expected one {FNAME} member, found {len(members)}"
                    )
                with zf.open(members[0]) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as e:
        errors.append(f"official Figshare archive: {e}")
        print("official download failed; trying the reserved Zenodo mirror")
        try:
            urllib.request.urlretrieve(ZENODO_MIRROR_URL, out)
        except Exception as mirror_error:
            errors.append(f"Zenodo mirror: {mirror_error}")
            sys.exit("download failed:\n  " + "\n  ".join(errors) +
                     f"\nOfficial record: {OFFICIAL_DOI}")
    got = sha256(out)
    if got != SHA256:
        sys.exit(f"SHA-256 mismatch!\n  expected {SHA256}\n  got      {got}\n"
                 f"Official source: {OFFICIAL_DOI}")
    print(f"verified SHA-256 OK -> {out}")


if __name__ == "__main__":
    main()
