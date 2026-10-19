import numpy
import pytest
import tract
from tract.bindings import lib

pytestmark = pytest.mark.skipif(
    not hasattr(lib, "tract_npz_get"), reason="libtract built without unstable-npyz"
)


def sample():
    return {
        "input": numpy.arange(4, dtype=numpy.float32).reshape(2, 2),
        "turn_0/half": numpy.array([0.5, -1.0, 2.0], dtype=numpy.float16),
        "mask": numpy.array([True, False]),
        "scalar": numpy.array(-7, dtype=numpy.int64),
    }


def test_npy_round_trip_with_numpy(tmp_path):
    for name, array in sample().items():
        path = tmp_path / (name.replace("/", "_") + ".npy")
        tract.Tensor.from_numpy(array).save_npy(path)
        assert numpy.array_equal(numpy.load(path), array)
        assert numpy.load(path).dtype == array.dtype
        numpy.save(path, array)
        assert numpy.array_equal(tract.Tensor.load_npy(path).to_numpy(), array)
        buffer = path.read_bytes()
        loaded = tract.Tensor.load_npy_buffer(buffer)
        assert numpy.array_equal(loaded.to_numpy(), array)
        assert loaded.to_npy_buffer() == buffer


def test_npy_reads_big_endian_fortran(tmp_path):
    array = numpy.asfortranarray(numpy.arange(6, dtype=">u2").reshape(2, 3))
    numpy.save(tmp_path / "be.npy", array)
    assert numpy.array_equal(tract.Tensor.load_npy(tmp_path / "be.npy").to_numpy(), array)


def test_npz_reads_numpy_output(tmp_path):
    arrays = sample()
    for compressed, save in [(False, numpy.savez), (True, numpy.savez_compressed)]:
        path = tmp_path / f"numpy-{compressed}.npz"
        save(path, **arrays)
        buffer = path.read_bytes()
        assert tract.npz.names(path) == list(arrays)
        assert tract.npz.names_buffer(buffer) == list(arrays)
        for name, array in arrays.items():
            assert numpy.array_equal(tract.npz.get(path, name).to_numpy(), array)
            assert numpy.array_equal(tract.npz.get_buffer(buffer, name).to_numpy(), array)
        for loaded in [tract.npz.load(path), tract.npz.load_buffer(buffer)]:
            assert [name for name, _ in loaded] == list(arrays)
            for (_, tensor), array in zip(loaded, arrays.values()):
                assert numpy.array_equal(tensor.to_numpy(), array)


def test_npz_written_for_numpy(tmp_path):
    arrays = sample()
    tensors = [(name, tract.Tensor.from_numpy(array)) for name, array in arrays.items()]
    for compress in [False, True]:
        path = tmp_path / f"tract-{compress}.npz"
        tract.npz.save(path, tensors, compress)
        assert path.read_bytes() == tract.npz.to_buffer(tensors, compress)
        with numpy.load(path) as loaded:
            assert list(loaded.files) == list(arrays)
            for name, array in arrays.items():
                assert loaded[name].dtype == array.dtype
                assert numpy.array_equal(loaded[name], array)


def test_errors(tmp_path):
    with pytest.raises(tract.TractError):
        tract.Tensor.load_npy_buffer(b"not a npy file")
    with pytest.raises(tract.TractError):
        tract.npz.names_buffer(b"not a zip")
    tract.npz.save(tmp_path / "x.npz", [], True)
    assert tract.npz.load(tmp_path / "x.npz") == []
    with pytest.raises(tract.TractError):
        tract.npz.get(tmp_path / "x.npz", "missing")
