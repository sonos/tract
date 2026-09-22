import numpy
import tract


def test_scalar_keeps_rank_0():
    tensor = tract.Tensor.from_numpy(numpy.array(3.0, dtype=numpy.float32))
    assert tensor.to_numpy().shape == ()


def test_vector_of_one_stays_rank_1():
    tensor = tract.Tensor.from_numpy(numpy.array([3.0], dtype=numpy.float32))
    assert tensor.to_numpy().shape == (1,)


def test_non_contiguous_array_is_copied_in_order():
    array = numpy.asfortranarray(numpy.arange(6, dtype=numpy.float32).reshape(2, 3))
    tensor = tract.Tensor.from_numpy(array)
    assert numpy.array_equal(tensor.to_numpy(), array)
