#!/usr/bin/env python3
"""Splits a stage's per-size NNEF tars into one shared weights tar and small per-size parts.

A stage is exported once per size, with identical weights. Entries the same in every tar go to
<stage>.weights.tar, the rest to <stage>-<size>.tar, written without an end-of-archive marker so
that reading a part then the weights yields one valid tar.

    split_stages.py add <dir> <stage> <base size> <size>   strips <size> against <base size>
    split_stages.py finish <dir> <stage> <base size>       splits the base into weights and part
"""

import hashlib
import json
import sys
import tarfile
from pathlib import Path

BLOCK = 1 << 20


def digests(path):
    found = {}
    with tarfile.open(path) as tar:
        for entry in tar:
            h = hashlib.sha256()
            with tar.extractfile(entry) as f:
                while chunk := f.read(BLOCK):
                    h.update(chunk)
            found[entry.name] = h.hexdigest()
    return found


def write(source, target, keep):
    with tarfile.open(source) as tar, tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as out:
        for entry in tar:
            if keep(entry.name):
                with tar.extractfile(entry) as f:
                    out.addfile(entry, f)


def write_part(source, target, keep):
    write(source, target, keep)
    data = target.read_bytes().rstrip(b"\0")
    target.write_bytes(data + b"\0" * (-len(data) % 512))
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB")


def main():
    command, root, stage, base = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    full = root / f"{stage}-{base}.nnef.tar"
    shared_path = root / f"{stage}.shared.json"
    if command == "add":
        size = sys.argv[5]
        source = root / f"{stage}-{size}.nnef.tar"
        base_digests = digests(full)
        shared = sorted(
            n for n, d in digests(source).items() if base_digests.get(n) == d and n != "graph.nnef"
        )
        if shared_path.exists():
            assert json.loads(shared_path.read_text()) == shared, f"{size} shares other entries"
        shared_path.write_text(json.dumps(shared))
        write_part(source, root / f"{stage}-{size}.tar", lambda n: n not in shared)
        source.unlink()
    else:
        shared = set(json.loads(shared_path.read_text()))
        write(full, root / f"{stage}.weights.tar", lambda n: n in shared)
        write_part(full, root / f"{stage}-{base}.tar", lambda n: n not in shared)
        full.unlink()
        shared_path.unlink()
        print(f"{stage}.weights.tar: {(root / f'{stage}.weights.tar').stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
