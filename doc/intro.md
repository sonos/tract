# Tract

tract is a neural network inference library. It takes trained networks from higher-level
frameworks (Tensorflow, PyTorch, etc.), converts them to an intermediate representation
and runs them on the end-user data. It is designed to be very portable and embedding
friendly. We believe in running Neural Network Inference on the Edge, on a
browser or a small embeddable CPU.

## How to use tract ?

* tract-onnx is a Rust library that can load and run an ONNX network. About 85%
    of ONNX operators are supported.
* tract-tensorflow is a Rust library that can load and run a TensorFlow 1
    network. Because of the huge size of TensorFlow, a smaller portion of the
    operator set is supported.
* tract-nnef is a Rust lbrary that can load and run NNEF networks. Most of
    NNEF is supported (missing deconv, ROI operations and quantization).
* tract is the main command line interface (can be installed with "cargo install").
    It can load network in any of the previously listed formats, dump them in a
    user friendly form, bench and profile a network.
    Additionaly, the tract command line can be used to convert a network to
    NNEF (with some extensions). tract-nnef is significanly smaller and
    lighter to start than tract-onnx or tract-tensorflow, so this conversion
    is useful for embedded situations.

## Crates

### tract-data

Contains the Tensor struct, DatumType enum, and TDim (symbolic dimension
value type).

### tract-linalg

It is bit of a misnomer: this crate contains the low-level optimised
routines for fast computation (actually not restricted to LINear ALGebra).
Beyond Intel, we payed specific attention to the ARM 6, 7 and 8 use
platforms. It is not meant to be used directly.

### tract-core

The heart of tract. It contains

    * the network graph representation manipulation (Graph, Node)
    * the "core" operator set of tract
    * most of the network optimisation logic.

tract-core depends on tract-linalg only, and is usually not used directly.

### tract-nnef

It support for NNEF format and maps its operator set to tract-core operators,
It also contains some tract-core proprietary extension to NNEF.
This crate depends on tract-core (thus tract-linalg transitively).
It is the entry-point for embedded situations where NNEF is preferred to ONNX
or tensorflow formats (requiring model translation to NNEF before hand).

### tract-hir

Python-based training frameworks (TensorFlow or ONNX) have to support lots of
"python-isms" or "numpy-isms". While they are helpful at model design time,
they can be a burden at inference time. As a consequence, we try to have most
of them translated before getting into tract-core. This allow us to comply with
ONNX or TensorFlow semantics while keeping tract-core complexity more
manageable.

Examples of such patterns are: negative indexing, negative rank indexing, rank
broadcasting.

It features the InferenceModel, InferenceFact (and friends), along with the
"analyser" that can work from the partial types and shapes included in the
training frameworks formats, to the stricter expectations of tract-core.

It also contains translation to tract-core logic for operators which have
close enough semantics between TensorFlow and ONNX.

This crate is not meant to be used directly.

### tract-onnx and tract-onnx-opl

Support for ONNX protobuf format and mapping of ONNX operators to tract-hir,
tract-core or ad-hoc operators.

tract-onnx-opl depends only on tract-core and tract-nnef. It contains
operators implementation from ONNX operators which do not have an equivalent
in tract-core, including dumping to / loading from OPL.

tract-onnx is the library to use to load and run an ONNX network. It uses
tract-hir for type inference and translate ONNX operators to operators from
tract-core and tract-onnx-opl.

### tract-tensorflow

Support for TensorFlow 1 frozen model format, similar to the ONNX crates.

NB: The split between tract-tensorflow (tensorflow parser, tensorflow operators
mapping to core) and tract-tensorflow-opl (ad-hoc implementation of operators)
has not been done yet.

### tract-pulse and tract-pulse-opl

Implements translation of streaming networks to pulsing network (tract-pulse)
including runtime support (ad-hoc operatrs in tract-pulse-opl).

### tract-kaldi

Partial support for kaldi framework model. Consider it very experimental, it
may disappear at any time.

### tract

In the `cli/` sub-directory, implements the command line tool.

## tract-OPL

Tract OPL (for Operation Programming Language) is an intermediate
representation of a Neural Network. It is based on NNEF. NNEF is a
specification aiming to be for *inference* applications what ONNX is
to *training* frameworks. As it turns out, inference implementations and
training frameworks have widely divergent objectives.

Tract can be used as a monolithic library, accepting an ONNX or
TensorFlow model, loading it and optimising it on the fly (using
tract-onnx API).

We have recently added support for (most of) NNEF. As this format is
designed for inference, translating it to tract-core operator set is
very straightforward.

We have built tract OPL on top the tract NNEF support: we have extended
NNEF to support tract operators that are not present in NNEF. The same
extension mechanism can be used to extend NNEF with operators belonging
to ONNX that we chose not to include in tract-core. That way it is
possible to reduce runtime footprint and startup time:
    * tract command line includes tract-onnx. It can be used to translate
an onnx network to a tract-core-plus-extensions model in memory, then dump
this network in NNEF form. This is done once, right after training.
    * At runtime we only to tract-core, tract-nnef (for the
format parser) and optionaly tract-onnx-opl if the network used one of the
handful of ONNX operations that are not supported natively by tract-core.

The split between translation time and runtime have also been done for the
streaming (aka pulse) capabilities. We only need tract-pulse to preprocess
the network (which we can do with the command line) but only ship
`tract-pulse-opl`.

It could (and should) be done with tract-tensorflow too.

Note that tract-OPL format is machine independant. We still need to call
into_optimized() on the loaded NNEF network to get the most efficient network
possible, but this operation is actually much lighter than the "decluttering"
of the network from the training formats to the tract-core/NNEF semantics.

We are playing with the idea of adding another similar split (tract R for
tract Runtime). The machine optimized network form would be stored at this
time, shedding most of the optimisation code from tract-core and making
networks even faster to load.
