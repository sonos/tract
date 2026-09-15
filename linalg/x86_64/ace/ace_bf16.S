// ACE BF16 matrix-multiply kernel (16x16 tile, K=2 inner).
// This is the real ACE assembly kernel that will replace the software model
// in src/ace/isa.rs when binutils/LLVM gain ACE support.
//
// ACE outer product: TOP2BF16PS consumes two ZMM registers (16x2 bf16 each)
// and accumulates a 16x16x2 outer product into a tile register (f32 accumulator).
// The kernel iterates over K in steps of 2, calling TOP2BF16PS for each block.
//
// NOTE: This file uses ACE mnemonics from the v1.15 specification.
// It will fail to compile on current toolchains (no ACE support yet).
// The cfg(tract_ace) build probe in build.rs handles this gracefully.
//
// Corresponds to: src/ace/isa::ace_top2bf16ps and src/ace/mmm::ace_matmul_bf16

.intel_syntax noprefix
.text

// Global kernel function
.globl ace_mmm_f32_bf16_16x16
ace_mmm_f32_bf16_16x16:
    // Input: pa, pb, k, ab pointer (passed via standard calling convention)
    // This follows the tract kernel ABI: the FusedKerSpec is passed as a pointer
    // and the kernel walks the fused-op list.
    
    // For now, this is a placeholder that demonstrates the ACE instruction usage.
    // The full kernel will be implemented once ACE assembler support is available.
    // The structure will mirror the existing AMX BF16 kernels in tract.
    
    // ACE kernel structure (conceptual):
    // 1. Zero tile accumulator: TILEZERO tmm0 (f32 interpretation)
    // 2. Loop over K blocks (K/2 iterations):
    //    - Load A panel ZMM: vmovdqa zmm1, [pa + block_offset]
    //    - Load B panel ZMM: vmovdqa zmm2, [pb + block_offset]
    //    - Outer product: TOP2BF16PS tmm0, zmm1, zmm2
    // 3. Store tile to accumulator array
    // 4. Handle fused operations (scalar ops, per-row/col ops, etc.)
    // 5. Store result (with f32->bf16 conversion if needed)
    // 6. TILERELEASE
    
    // Placeholder: return error (unimplemented until ACE support exists)
    mov eax, 1
    ret