#include <stddef.h>

void dot_f16(
	_Float16 *sum,
	size_t count,
	_Float16 *iptr,
	_Float16 *kptr,
	ptrdiff_t *ioffsets,
	ptrdiff_t *koffsets
) {
	#pragma clang loop vectorize(enable) interleave(enable)
	for (size_t c = 0; c < count; ++c) {
		_Float16 k = *(kptr + koffsets[c]);
		_Float16 i = *(iptr + ioffsets[c]);
		*sum += k * i;
	}
}