# Notes about implementing and working with the kernels

Kernels in tract-linalg are built using templated assembly and via `extern "C"` calling conventions.

The templates are stored in linalg/$arch, and in general the file and
main entrypoint share name stem . However, the proc name has a suffix
based on the package version. In order to skip maintaining this the `extern_kernel!` macro
declares the matching function but also strips the suffix.

Kernels work like a VM. When dispatching a kernel there's a list of
instructions from `FusedKerSpec` that's dispatched in a jump
table. For example; as of writing a MatMatmUl is roughly encoded as
`[Clear, AddMatMul, Store, Done]`. The dispatch is called `non_linear_loop`.

When iterating on assembly; building the code and looking at the
generated assembly under
target/debug/build/tract-linalg-***/out/fma_mmm_*.S can be much easier
than tracking the flow through each macro.

```c
// matmatmul where B: inputs, A: weights, k: rows (?)
void packed_packed(float* B, float *A, long k) {
    while k --> 0 {
      	vec8 a1 = load_vec(A);
		vec8 a2 = load_vec(A+8);

	    for col in 0..6 {
           vec8 b = load_scalar(B + col);
           r[2 * col + 0] += r[2 * col + 0] = a1 * b;
		   r[2 * col + 1] += r[2 * col + 1] = a2 * b;
		}

		vec8 weight2(B+1);
	}
}

void store(float* Out, long rs, long cs) {
    float* r[8:14] = [out...];

	for group in 0..4 {
	    for elem in 0..4 {
    	    for col in 0..6 {
	    	    r[8 + col] = r[col * 2][group * 4 * elem];
		    	r[8 + col] += rs;
		    }
		}
	}
}
```
