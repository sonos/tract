"""Rewrites torch_to_nnef's NNEF archives of nvidia/nemotron-3.5-asr-streaming-0.6b so their
weights are references to the tensors of the Hugging Face model.safetensors:

    python3 export.py <model.safetensors> <out dir> <archive.nnef.tgz>...

A variable whose bytes are a whole safetensors tensor becomes a tract_core_named_weight of that
tensor; one whose bytes are a run of rows of a tensor (the LSTM's per-gate matrices) slices them
out of it. The other variables keep their data in the archive."""

import hashlib
import io
import json
import os
import re
import struct
import sys
import tarfile

st_path, out_dir, archives = sys.argv[1], sys.argv[2], sys.argv[3:]

with open(st_path, 'rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
    header.pop('__metadata__', None)
    data = f.read()
hf = {}
for name, entry in header.items():
    assert entry['dtype'] == 'F32', (name, entry['dtype'])
    begin, end = entry['data_offsets']
    hf[name] = (entry['shape'], data[begin:end])
by_digest = {hashlib.sha1(b).hexdigest(): name for name, (_, b) in hf.items()}


def find(blob):
    name = by_digest.get(hashlib.sha1(blob).hexdigest())
    if name is not None:
        return name, None
    for name, (shape, b) in hf.items():
        row = len(b) // shape[0]
        at = b.find(blob)
        if at >= 0 and at % row == 0 and len(blob) % row == 0:
            return name, (at // row, (at + len(blob)) // row)
    return None, None


def named_weight(name):
    shape = ', '.join(str(d) for d in hf[name][0])
    return f'tract_core_named_weight(label = "{name}", shape = [{shape}], datum_type = "F32")'


VARIABLE = re.compile(r"^(\s*)(\w+) = variable<scalar>\(label = '([^']+)', shape = \[([^\]]*)\]\);$")

os.makedirs(out_dir, exist_ok=True)
for archive in archives:
    with tarfile.open(archive) as t:
        files = {m.name: t.extractfile(m).read() for m in t if m.isfile()}
    graph = files.pop('graph.nnef').decode().split('\n')
    kept, linked, sliced = {}, 0, 0
    for i, line in enumerate(graph):
        m = VARIABLE.match(line)
        if not m:
            continue
        indent, ident, label, shape = m.groups()
        dat = files[label + '.dat']
        bits, code = struct.unpack('<II', dat[44:52])
        name, rows = find(dat[128:]) if (bits, code) == (32, 0) else (None, None)
        if name is None:
            kept[label + '.dat'] = dat
        elif rows is None:
            assert [int(d) for d in shape.split(',')] == hf[name][0], (label, name)
            graph[i] = f'{indent}{ident} = {named_weight(name)};'
            linked += 1
        else:
            dims = [int(d) for d in shape.split(',')]
            assert dims[0] == 1 and dims[1] == rows[1] - rows[0], (label, name, rows)
            graph[i] = (
                f'{indent}{ident}_full = {named_weight(name)};\n'
                f'{indent}{ident} = unsqueeze(slice({ident}_full, axes = [0], begin = [{rows[0]}], '
                f'end = [{rows[1]}]), axes = [0]);'
            )
            sliced += 1
    graph = '\n'.join(graph)
    if 'extension tract_registry tract_core;' not in graph:
        graph = graph.replace('version 1.0;', 'version 1.0;\n\nextension tract_registry tract_core;', 1)
    out = os.path.join(out_dir, os.path.basename(archive).replace('.nnef.tgz', '.nnef.tar'))
    with tarfile.open(out, 'w') as t:
        for name, content in [('graph.nnef', graph.encode())] + sorted(kept.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            t.addfile(info, io.BytesIO(content))
    carried = sum(len(d) for d in kept.values())
    print(f'{out}: {linked} named, {sliced} sliced, {len(kept)} carried ({carried / 1e6:.1f} MB)')
