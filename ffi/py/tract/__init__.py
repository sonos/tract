"""
`tract` Python bindings library

`tract` is a neural network inference engine.
Its main purpose is to *run* a neural network on production premises after it has been trained.
It is a native library written in Rust, and specific attention has been given to its performance
in sound streaming applications in embedded context (typically ARM Cortex-A CPUs), but it is meant
to be a generic engine, performing more than decently with various loads on most architectures. You
can use it to run image categorization on a PC.

```python
import tract

# load MobileNet version 2, an image categorization model
model = (
    tract.onnx()
    .model_for_path("./mobilenetv2-7.onnx")
    .into_optimized()
    .into_runnable()
)

# load, as a numpy array, a picture of Grace Hopper, wearing her military uniform
grace_hopper_1x3x224x244 = numpy.load("grace_hopper_1x3x224x244.npy")

# run the image through Mobilenet in tract
result = model.run([grace_hopper_1x3x224x244])

# output is an array of confidence for each class of the ImageNet challenge
confidences = result[0].to_numpy()

# class 652 is "military uniform"
assert numpy.argmax(confidences) == 652
```

`tract` can also be used as a "model cooking" toolbox: once a model has been trained, it is
sometimes useful to perform some transformations, simplifications and optimizations before shipping
it. These bindings offer access to some of `tract` cooking facilities.
"""

import numpy
from ctypes import *
from pathlib import Path
from typing import Dict, List, Union

from .bindings import check, lib, TractError
from .value import Value
from .fact import Fact, InferenceFact
from .model import Model
from .inference_model import InferenceModel
from .runnable import Runnable
from .nnef import Nnef
from .onnx import Onnx

def version() -> str:
    """Return the version string of `tract` native library"""
    return str(lib.tract_version(), "utf-8")

def nnef() -> Nnef:
    """Return a newly-created NNEF context for loading and saving models"""
    return Nnef()

def onnx() -> Onnx:
    """Return a newly-created ONNX context for loading models"""
    return Onnx()

