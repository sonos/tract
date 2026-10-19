"""
Reading and writing numpy ``.npz`` archives.

Every function takes the archive as a path or a buffer and reopens it, there is no archive
object. Entry names are given without their ``.npy`` suffix.

Unstable: needs libtract built with the ``unstable-npyz`` feature.
"""

from ctypes import *
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, Union

from .bindings import check, lib, take_buffer, unstable_npyz
from .tensor import Tensor


def _collect_names(query: Callable) -> List[str]:
    count = c_size_t()
    check(query(byref(count), None))
    cstrings = (POINTER(c_char) * count.value)()
    check(query(byref(count), cstrings))
    names = []
    for i in range(0, count.value):
        names.append(str(cast(cstrings[i], c_char_p).value, "utf-8"))
        lib.tract_free_cstring(cstrings[i])
    return names


def _named_pointers(tensors: Iterable[Tuple[str, Tensor]]):
    items = list(tensors)
    for _, tensor in items:
        tensor._valid()
    names = (c_char_p * len(items))(*(name.encode("utf-8") for name, _ in items))
    pointers = (c_void_p * len(items))(*(tensor.ptr.value for _, tensor in items))
    return c_size_t(len(items)), names, pointers


def names(path: Union[str, Path]) -> List[str]:
    """Names of the arrays in a ``.npz`` file, in archive order."""
    path = str(path).encode("utf-8")
    query = unstable_npyz("tract_npz_names")
    return _collect_names(lambda count, names: query(path, count, names))


def names_buffer(data: bytes) -> List[str]:
    """Names of the arrays in the content of a ``.npz`` file, in archive order."""
    query = unstable_npyz("tract_npz_names_buffer")
    return _collect_names(lambda count, names: query(data, c_size_t(len(data)), count, names))


def get(path: Union[str, Path], name: str) -> Tensor:
    """Read one array of a ``.npz`` file. Only this entry is decompressed."""
    ptr = c_void_p()
    check(unstable_npyz("tract_npz_get")(str(path).encode("utf-8"), name.encode("utf-8"), byref(ptr)))
    return Tensor(ptr)


def get_buffer(data: bytes, name: str) -> Tensor:
    """Read one array out of the content of a ``.npz`` file."""
    ptr = c_void_p()
    check(unstable_npyz("tract_npz_get_buffer")(data, c_size_t(len(data)), name.encode("utf-8"), byref(ptr)))
    return Tensor(ptr)


def load(path: Union[str, Path]) -> List[Tuple[str, Tensor]]:
    """Read all the arrays of a ``.npz`` file, in archive order.

    The archive is reopened for each array.
    """
    return [(name, get(path, name)) for name in names(path)]


def load_buffer(data: bytes) -> List[Tuple[str, Tensor]]:
    """Read all the arrays out of the content of a ``.npz`` file, in archive order.

    The archive is reopened for each array.
    """
    return [(name, get_buffer(data, name)) for name in names_buffer(data)]


def save(path: Union[str, Path], tensors: Iterable[Tuple[str, Tensor]], compress: bool):
    """Write named tensors as a ``.npz`` file, as ``numpy.savez`` (or
    ``numpy.savez_compressed`` with ``compress``) would."""
    count, names, pointers = _named_pointers(tensors)
    check(unstable_npyz("tract_npz_save")(str(path).encode("utf-8"), count, names, pointers, c_bool(compress)))


def to_buffer(tensors: Iterable[Tuple[str, Tensor]], compress: bool) -> bytes:
    """Serialize named tensors in ``.npz`` format."""
    count, names, pointers = _named_pointers(tensors)
    data = POINTER(c_uint8)()
    length = c_size_t()
    check(unstable_npyz("tract_npz_to_buffer")(count, names, pointers, c_bool(compress), byref(data), byref(length)))
    return take_buffer(data, length)
