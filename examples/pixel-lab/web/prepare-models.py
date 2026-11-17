#!/usr/bin/env python3
"""Download the webcam MODNet and PP-HumanSegV2 ONNX files and write the RVM fp16 file the page runs.

The fp16 file is the official v1.0.0 graph with two Casts: the camera input is
float32 and the matte output is float32. Recurrent state stays float16.
"""

import hashlib
import pathlib
import sys
import urllib.request

from onnx import TensorProto, helper, load, save

WEBCAM_URL = "https://github.com/yakhyo/modnet/releases/download/weights/modnet_webcam.onnx"
WEBCAM_SHA = "de03cc16f3c91f25b7c2f0b42ea1a8d34f40a752234f3887572655e744e55306"
PP_URL = (
    "https://tract-test-assets.tract.rs/segmentation/pp-humanseg-v2/"
    "pp_humanseg_v2_lite_256x144_person.onnx"
)
PP_SHA = "6734ac175ca3c3d94bf1dfb7e253573c84a602b8b49d27499d256b73faa476f7"
RVM_URL = (
    "https://huggingface.co/eafish/web-onnx/resolve/"
    "576473dca1b5792d8c30753b21325edc4063440b/rvm_mobilenetv3_fp16.onnx"
)
OFFICIAL_SHA = "6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64"
RVM_SHA = "84c01698359a946d2167152f378d27e59eaebc86666190ffd9b5642aba7e4e98"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "tract-pixel-lab"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def ensure(url, path, digest):
    if path.is_file() and sha256(path) == digest:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    fetch(url, temporary)
    if sha256(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"{path.name} sha256 does not match {digest}")
    temporary.replace(path)


def cast_rvm(source, destination):
    model = load(source)
    graph = model.graph
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == "src":
                node.input[index] = "src_f16"
        for index, name in enumerate(node.output):
            if name == "pha":
                node.output[index] = "pha_f16"
    for value in graph.input:
        if value.name == "src":
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    for value in graph.output:
        if value.name == "pha":
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    graph.node.insert(0, helper.make_node("Cast", ["src"], ["src_f16"], name="src_f32_to_f16", to=TensorProto.FLOAT16))
    graph.node.append(helper.make_node("Cast", ["pha_f16"], ["pha"], name="pha_f16_to_f32", to=TensorProto.FLOAT))
    save(model, destination)
    if sha256(destination) != RVM_SHA:
        raise SystemExit(f"RVM fp16 sha256 does not match {RVM_SHA}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} dist/models")
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    webcam = out / "modnet_webcam.onnx"
    ensure(WEBCAM_URL, webcam, WEBCAM_SHA)
    ensure(PP_URL, out / "pp_humanseg_v2.onnx", PP_SHA)
    official = out / "rvm_mobilenetv3_fp16.official.onnx"
    page = out / "rvm_mobilenetv3_fp16.onnx"
    if not (page.is_file() and sha256(page) == RVM_SHA):
        fetch(RVM_URL, official)
        try:
            if sha256(official) != OFFICIAL_SHA:
                raise SystemExit(f"official RVM fp16 sha256 does not match {OFFICIAL_SHA}")
            cast_rvm(official, page)
        finally:
            official.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
