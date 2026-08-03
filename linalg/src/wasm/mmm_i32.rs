use crate::Scaler;
use crate::mmm::FusedKerSpec;
use crate::mmm::ImplementationQuality;

// Wasm SIMD int8 -> i32 matmul kernel (4x4). WASM's only integer dot
// (i32x4.relaxed_dot_i8x16_i7x16) is non-deterministic for full i8 (its 2nd
// operand is i7), so for a bit-exact kernel the AddMatMul K-loop uses widening
// i8->i32 + i32x4 mul/add (an extmul/SMLAL-style outer product). The quant
// epilogue + fuse ops reuse the bit-exact scalar path (q_scale/q_shr/q_shl),
// which is O(MR*NR) and negligible vs the O(MR*NR*K) inner loop. Bit-identical
// to generic_i32_4x4; selected for i8 matmul via its ManuallyOptimized quality
// (WASM had no int8 matmul kernel — int8 fell back to the generic scalar one).
#[inline(never)]
unsafe fn kernel_i32_4x4(mut pnl: *const FusedKerSpec<i32>) -> isize {
    use crate::ScaleShiftAndRound;
    use std::arch::wasm32::*;
    unsafe {
        let mut ab = [[0i32; 4]; 4];
        loop {
            if pnl.is_null() {
                break;
            }
            match *pnl {
                FusedKerSpec::Done => break,
                FusedKerSpec::Clear => ab = [[0i32; 4]; 4],
                FusedKerSpec::LoadTile(col_major, _row_major) => {
                    for row in 0..4 {
                        for col in 0..4 {
                            ab[row][col] = *col_major.add(col * 4 + row);
                        }
                    }
                }
                FusedKerSpec::ScalarAdd(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] += a;
                        }
                    }
                }
                FusedKerSpec::ScalarMul(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] *= a;
                        }
                    }
                }
                FusedKerSpec::ScalarMin(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(m);
                        }
                    }
                }
                FusedKerSpec::ScalarMax(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(m);
                        }
                    }
                }
                FusedKerSpec::ScalarSub(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = m - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::ScalarSubF(m) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] -= m;
                        }
                    }
                }
                FusedKerSpec::LeakyRelu(a) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = if ab[i][j] > 0 { ab[i][j] } else { a * ab[i][j] };
                        }
                    }
                }
                FusedKerSpec::PerRowMin(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(v);
                        }
                    }
                }
                FusedKerSpec::PerRowMax(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(v);
                        }
                    }
                }
                FusedKerSpec::PerRowAdd(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] += v;
                        }
                    }
                }
                FusedKerSpec::PerRowMul(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] *= v;
                        }
                    }
                }
                FusedKerSpec::PerRowSub(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] = v - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::PerRowSubF(m) => {
                    for i in 0..4 {
                        let v = *m.add(i);
                        for j in 0..4 {
                            ab[i][j] -= v;
                        }
                    }
                }
                FusedKerSpec::PerColMin(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].min(c[j]);
                        }
                    }
                }
                FusedKerSpec::PerColMax(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].max(c[j]);
                        }
                    }
                }
                FusedKerSpec::PerColAdd(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] += c[j];
                        }
                    }
                }
                FusedKerSpec::PerColMul(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] *= c[j];
                        }
                    }
                }
                FusedKerSpec::PerColSub(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = c[j] - ab[i][j];
                        }
                    }
                }
                FusedKerSpec::PerColSubF(m) => {
                    let c = std::slice::from_raw_parts(m, 4);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] -= c[j];
                        }
                    }
                }
                FusedKerSpec::AddRowColProducts(rows, cols) => {
                    for i in 0..4 {
                        let r = *rows.add(i);
                        for j in 0..4 {
                            ab[i][j] += r * *cols.add(j);
                        }
                    }
                }
                FusedKerSpec::AddUnicast(other) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            let p = other.ptr.offset(
                                other.row_byte_stride * i as isize
                                    + other.col_byte_stride * j as isize,
                            );
                            let v = match other.item_size {
                                1 => *(p as *const i8) as i32,
                                4 => *(p as *const i32),
                                _ => return 1,
                            };
                            ab[i][j] += v;
                        }
                    }
                }
                FusedKerSpec::ShiftLeft(shift) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_shl(shift);
                        }
                    }
                }
                FusedKerSpec::RoundingShiftRight(shift, rp) => {
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_shr(shift, rp);
                        }
                    }
                }
                FusedKerSpec::QScale(shift, rp, mult) => {
                    let s = Scaler::from_fuse_params(shift, rp, mult);
                    for i in 0..4 {
                        for j in 0..4 {
                            ab[i][j] = ab[i][j].q_scale(s);
                        }
                    }
                }
                FusedKerSpec::AddMatMul { k, pa, pb, packing } => {
                    if packing == 1 {
                        let a = pa as *const i8;
                        let b = pb as *const i8;
                        let mut acc = [
                            v128_load(ab[0].as_ptr() as *const v128),
                            v128_load(ab[1].as_ptr() as *const v128),
                            v128_load(ab[2].as_ptr() as *const v128),
                            v128_load(ab[3].as_ptr() as *const v128),
                        ];
                        // PackedI8K4 (K=4-inner): per 4-K block, one B v128 load is
                        // shared across the 4 rows; each row broadcasts its 4 K bytes
                        // (a[kb*16 + m*4 ..]) and issues one relaxed_dot (16 MACs).
                        // b[kb*16 + n*4 + kr]. Tail K (k%4) is zero-padded by the packer.
                        #[cfg(target_feature = "relaxed-simd")]
                        for kb in 0..k.div_ceil(4) {
                            let b_all = v128_load(b.add(kb * 16) as *const v128);
                            for (m, acc_m) in acc.iter_mut().enumerate() {
                                let a4 = (a.add(kb * 16 + m * 4) as *const i32).read_unaligned();
                                *acc_m = i32x4_relaxed_dot_i8x16_i7x16_add(
                                    i32x4_splat(a4),
                                    b_all,
                                    *acc_m,
                                );
                            }
                        }
                        // Deterministic fallback (no relaxed-simd): standard PackedFormat
                        // K-major (4 i8 per k), widening outer-product accumulate.
                        #[cfg(not(target_feature = "relaxed-simd"))]
                        for ik in 0..k {
                            let bw = v128_load32_zero(b.add(4 * ik) as *const u32);
                            let bw = i16x8_extend_low_i8x16(bw);
                            let bw = i32x4_extend_low_i16x8(bw);
                            let ar = a.add(4 * ik);
                            acc[0] =
                                i32x4_add(acc[0], i32x4_mul(i32x4_splat(*ar.add(0) as i32), bw));
                            acc[1] =
                                i32x4_add(acc[1], i32x4_mul(i32x4_splat(*ar.add(1) as i32), bw));
                            acc[2] =
                                i32x4_add(acc[2], i32x4_mul(i32x4_splat(*ar.add(2) as i32), bw));
                            acc[3] =
                                i32x4_add(acc[3], i32x4_mul(i32x4_splat(*ar.add(3) as i32), bw));
                        }
                        v128_store(ab[0].as_mut_ptr() as *mut v128, acc[0]);
                        v128_store(ab[1].as_mut_ptr() as *mut v128, acc[1]);
                        v128_store(ab[2].as_mut_ptr() as *mut v128, acc[2]);
                        v128_store(ab[3].as_mut_ptr() as *mut v128, acc[3]);
                    } else if packing == 0 {
                        // i32 x i32, K-major (scalar; rare path).
                        let a = pa as *const i32;
                        let b = pb as *const i32;
                        for ik in 0..k {
                            for i in 0..4 {
                                let av = *a.add(4 * ik + i);
                                for j in 0..4 {
                                    ab[i][j] += av * *b.add(4 * ik + j);
                                }
                            }
                        }
                    } else {
                        return 1;
                    }
                }
                FusedKerSpec::Store(tile) => match tile.item_size {
                    1 => {
                        for i in 0..4 {
                            for j in 0..4 {
                                let loc = tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as *mut u8;
                                *loc = ab[i][j] as u8;
                            }
                        }
                    }
                    4 => {
                        for i in 0..4 {
                            for j in 0..4 {
                                let loc = tile.ptr.offset(
                                    tile.row_byte_stride * i as isize
                                        + tile.col_byte_stride * j as isize,
                                ) as *mut i32;
                                *loc = ab[i][j];
                            }
                        }
                    }
                    _ => return 1,
                },
            };
            pnl = pnl.add(1);
        }
    }
    0
}

// i8i8 packing for wasm_i32_4x4. Under +relaxed-simd the kernel uses the
// `i32x4_relaxed_dot_i8x16_i7x16_add` SDOT-analog, which wants 4 contiguous K per
// mn-lane → PackedI8K4 (K=4-inner). Without relaxed-simd it uses the widening
// outer-product, which wants K-major → standard PackedFormat. Both are picked at
// compile time so the kernel's AddMatMul and the packer always agree.
#[cfg(target_feature = "relaxed-simd")]
fn wasm_i8_packing() -> impl crate::mmm::MMMInputFormat {
    crate::pack::PackedI8K4::new(4)
}
#[cfg(not(target_feature = "relaxed-simd"))]
fn wasm_i8_packing() -> impl crate::mmm::MMMInputFormat {
    use crate::pack::Packing;
    i8::packing(4)
}

MMMRustKernel!(kernel_i32_4x4 => wasm_i32_4x4<i32>(4,4)
    packing[1] = i8i8 => |k| k.with_packing(wasm_i8_packing(), wasm_i8_packing());
    quality(ImplementationQuality::ManuallyOptimized)
    store(i8)
);
