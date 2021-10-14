#![allow(non_snake_case)]
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "blis")]
extern crate blis_src;
#[cfg(feature = "blis")]
extern crate cblas;

pub fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

pub fn tile_2x2(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for row in 0..m / 2 {
        for col in 0..n / 2 {
            let mut sum00 = 0.0;
            let mut sum01 = 0.0;
            let mut sum10 = 0.0;
            let mut sum11 = 0.0;
            for i in 0..k {
                let a0 = a[row * k + i];
                let a1 = a[(row + 1) * k + i];
                let b0 = b[i * n + col];
                let b1 = b[i * n + col + 1];
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
            }
            c[row * n + col] = sum00;
            c[row * n + col + 1] = sum01;
            c[(row + 1) * n + col] = sum10;
            c[(row + 1) * n + col + 1] = sum11;
        }
    }
}

pub fn tile_4x4(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for row in 0..m / 4 {
        for col in 0..n / 4 {
            let mut sum00 = 0.0;
            let mut sum01 = 0.0;
            let mut sum02 = 0.0;
            let mut sum03 = 0.0;
            let mut sum10 = 0.0;
            let mut sum11 = 0.0;
            let mut sum12 = 0.0;
            let mut sum13 = 0.0;
            let mut sum20 = 0.0;
            let mut sum21 = 0.0;
            let mut sum22 = 0.0;
            let mut sum23 = 0.0;
            let mut sum30 = 0.0;
            let mut sum31 = 0.0;
            let mut sum32 = 0.0;
            let mut sum33 = 0.0;
            for i in 0..k {
                let a0 = a[row * k + i];
                let a1 = a[(row + 1) * k + i];
                let a2 = a[(row + 2) * k + i];
                let a3 = a[(row + 3) * k + i];
                let b0 = b[i * n + col];
                let b1 = b[i * n + col + 1];
                let b2 = b[i * n + col + 2];
                let b3 = b[i * n + col + 3];
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum02 += a0 * b2;
                sum03 += a0 * b3;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                sum12 += a1 * b2;
                sum13 += a1 * b3;
                sum20 += a2 * b0;
                sum21 += a2 * b1;
                sum22 += a2 * b2;
                sum23 += a2 * b3;
                sum30 += a3 * b0;
                sum31 += a3 * b1;
                sum32 += a3 * b2;
                sum33 += a3 * b3;
            }
            c[(row + 0) * n + col] = sum00;
            c[(row + 0) * n + col + 1] = sum01;
            c[(row + 0) * n + col + 2] = sum02;
            c[(row + 0) * n + col + 3] = sum03;
            c[(row + 1) * n + col] = sum10;
            c[(row + 1) * n + col + 1] = sum11;
            c[(row + 1) * n + col + 2] = sum12;
            c[(row + 1) * n + col + 3] = sum13;
            c[(row + 2) * n + col] = sum20;
            c[(row + 2) * n + col + 1] = sum21;
            c[(row + 2) * n + col + 2] = sum22;
            c[(row + 2) * n + col + 3] = sum23;
            c[(row + 3) * n + col] = sum30;
            c[(row + 3) * n + col + 1] = sum31;
            c[(row + 3) * n + col + 2] = sum32;
            c[(row + 3) * n + col + 3] = sum33;
        }
    }
}

pub fn tile_8x8(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for row in 0..m / 8 {
        for col in 0..n / 8 {
            let mut sum00 = 0.0;
            let mut sum01 = 0.0;
            let mut sum02 = 0.0;
            let mut sum03 = 0.0;
            let mut sum04 = 0.0;
            let mut sum05 = 0.0;
            let mut sum06 = 0.0;
            let mut sum07 = 0.0;
            let mut sum10 = 0.0;
            let mut sum11 = 0.0;
            let mut sum12 = 0.0;
            let mut sum13 = 0.0;
            let mut sum14 = 0.0;
            let mut sum15 = 0.0;
            let mut sum16 = 0.0;
            let mut sum17 = 0.0;
            let mut sum20 = 0.0;
            let mut sum21 = 0.0;
            let mut sum22 = 0.0;
            let mut sum23 = 0.0;
            let mut sum24 = 0.0;
            let mut sum25 = 0.0;
            let mut sum26 = 0.0;
            let mut sum27 = 0.0;
            let mut sum30 = 0.0;
            let mut sum31 = 0.0;
            let mut sum32 = 0.0;
            let mut sum33 = 0.0;
            let mut sum34 = 0.0;
            let mut sum35 = 0.0;
            let mut sum36 = 0.0;
            let mut sum37 = 0.0;
            let mut sum40 = 0.0;
            let mut sum41 = 0.0;
            let mut sum42 = 0.0;
            let mut sum43 = 0.0;
            let mut sum44 = 0.0;
            let mut sum45 = 0.0;
            let mut sum46 = 0.0;
            let mut sum47 = 0.0;
            let mut sum50 = 0.0;
            let mut sum51 = 0.0;
            let mut sum52 = 0.0;
            let mut sum53 = 0.0;
            let mut sum54 = 0.0;
            let mut sum55 = 0.0;
            let mut sum56 = 0.0;
            let mut sum57 = 0.0;
            let mut sum60 = 0.0;
            let mut sum61 = 0.0;
            let mut sum62 = 0.0;
            let mut sum63 = 0.0;
            let mut sum64 = 0.0;
            let mut sum65 = 0.0;
            let mut sum66 = 0.0;
            let mut sum67 = 0.0;
            let mut sum70 = 0.0;
            let mut sum71 = 0.0;
            let mut sum72 = 0.0;
            let mut sum73 = 0.0;
            let mut sum74 = 0.0;
            let mut sum75 = 0.0;
            let mut sum76 = 0.0;
            let mut sum77 = 0.0;
            for i in 0..k {
                let a0 = a[row * k + i];
                let a1 = a[(row + 1) * k + i];
                let a2 = a[(row + 2) * k + i];
                let a3 = a[(row + 3) * k + i];
                let a4 = a[(row + 4) * k + i];
                let a5 = a[(row + 5) * k + i];
                let a6 = a[(row + 6) * k + i];
                let a7 = a[(row + 7) * k + i];
                let b0 = b[i * n + col];
                let b1 = b[i * n + col + 1];
                let b2 = b[i * n + col + 2];
                let b3 = b[i * n + col + 3];
                let b4 = b[i * n + col + 4];
                let b5 = b[i * n + col + 5];
                let b6 = b[i * n + col + 6];
                let b7 = b[i * n + col + 7];
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum02 += a0 * b2;
                sum03 += a0 * b3;
                sum04 += a0 * b4;
                sum05 += a0 * b5;
                sum06 += a0 * b6;
                sum07 += a0 * b7;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                sum12 += a1 * b2;
                sum13 += a1 * b3;
                sum14 += a1 * b4;
                sum15 += a1 * b5;
                sum16 += a1 * b6;
                sum17 += a1 * b7;
                sum20 += a2 * b0;
                sum21 += a2 * b1;
                sum22 += a2 * b2;
                sum23 += a2 * b3;
                sum24 += a2 * b4;
                sum25 += a2 * b5;
                sum26 += a2 * b6;
                sum27 += a2 * b7;
                sum30 += a3 * b0;
                sum31 += a3 * b1;
                sum32 += a3 * b2;
                sum33 += a3 * b3;
                sum34 += a3 * b4;
                sum35 += a3 * b5;
                sum36 += a3 * b6;
                sum37 += a3 * b7;
                sum40 += a4 * b0;
                sum41 += a4 * b1;
                sum42 += a4 * b2;
                sum43 += a4 * b3;
                sum44 += a4 * b4;
                sum45 += a4 * b5;
                sum46 += a4 * b6;
                sum47 += a4 * b7;
                sum50 += a5 * b0;
                sum51 += a5 * b1;
                sum52 += a5 * b2;
                sum53 += a5 * b3;
                sum54 += a5 * b4;
                sum55 += a5 * b5;
                sum56 += a5 * b6;
                sum57 += a5 * b7;
                sum60 += a6 * b0;
                sum61 += a6 * b1;
                sum62 += a6 * b2;
                sum63 += a6 * b3;
                sum64 += a6 * b4;
                sum65 += a6 * b5;
                sum66 += a6 * b6;
                sum67 += a6 * b7;
                sum70 += a7 * b0;
                sum71 += a7 * b1;
                sum72 += a7 * b2;
                sum73 += a7 * b3;
                sum74 += a7 * b4;
                sum75 += a7 * b5;
                sum76 += a7 * b6;
                sum77 += a7 * b7;
            }
            c[(row + 0) * n + col] = sum00;
            c[(row + 0) * n + col + 1] = sum01;
            c[(row + 0) * n + col + 2] = sum02;
            c[(row + 0) * n + col + 3] = sum03;
            c[(row + 0) * n + col + 4] = sum04;
            c[(row + 0) * n + col + 5] = sum05;
            c[(row + 0) * n + col + 6] = sum06;
            c[(row + 0) * n + col + 7] = sum07;
            c[(row + 1) * n + col] = sum10;
            c[(row + 1) * n + col + 1] = sum11;
            c[(row + 1) * n + col + 2] = sum12;
            c[(row + 1) * n + col + 3] = sum13;
            c[(row + 1) * n + col + 4] = sum14;
            c[(row + 1) * n + col + 5] = sum15;
            c[(row + 1) * n + col + 6] = sum16;
            c[(row + 1) * n + col + 7] = sum17;
            c[(row + 2) * n + col] = sum20;
            c[(row + 2) * n + col + 1] = sum21;
            c[(row + 2) * n + col + 2] = sum22;
            c[(row + 2) * n + col + 3] = sum23;
            c[(row + 2) * n + col + 4] = sum24;
            c[(row + 2) * n + col + 5] = sum25;
            c[(row + 2) * n + col + 6] = sum26;
            c[(row + 2) * n + col + 7] = sum27;
            c[(row + 3) * n + col] = sum30;
            c[(row + 3) * n + col + 1] = sum31;
            c[(row + 3) * n + col + 2] = sum32;
            c[(row + 3) * n + col + 3] = sum33;
            c[(row + 3) * n + col + 4] = sum34;
            c[(row + 3) * n + col + 5] = sum35;
            c[(row + 3) * n + col + 6] = sum36;
            c[(row + 3) * n + col + 7] = sum37;
            c[(row + 4) * n + col] = sum40;
            c[(row + 4) * n + col + 1] = sum41;
            c[(row + 4) * n + col + 2] = sum42;
            c[(row + 4) * n + col + 3] = sum43;
            c[(row + 4) * n + col + 4] = sum44;
            c[(row + 4) * n + col + 5] = sum45;
            c[(row + 4) * n + col + 6] = sum46;
            c[(row + 4) * n + col + 7] = sum47;
            c[(row + 5) * n + col] = sum50;
            c[(row + 5) * n + col + 1] = sum51;
            c[(row + 5) * n + col + 2] = sum52;
            c[(row + 5) * n + col + 3] = sum53;
            c[(row + 5) * n + col + 4] = sum54;
            c[(row + 5) * n + col + 5] = sum55;
            c[(row + 5) * n + col + 6] = sum56;
            c[(row + 5) * n + col + 7] = sum57;
            c[(row + 6) * n + col] = sum60;
            c[(row + 6) * n + col + 1] = sum61;
            c[(row + 6) * n + col + 2] = sum62;
            c[(row + 6) * n + col + 3] = sum63;
            c[(row + 6) * n + col + 4] = sum64;
            c[(row + 6) * n + col + 5] = sum65;
            c[(row + 6) * n + col + 6] = sum66;
            c[(row + 6) * n + col + 7] = sum67;
            c[(row + 7) * n + col] = sum70;
            c[(row + 7) * n + col + 1] = sum71;
            c[(row + 7) * n + col + 2] = sum72;
            c[(row + 7) * n + col + 3] = sum73;
            c[(row + 7) * n + col + 4] = sum74;
            c[(row + 7) * n + col + 5] = sum75;
            c[(row + 7) * n + col + 6] = sum76;
            c[(row + 7) * n + col + 7] = sum77;
        }
    }
}

pub fn matrixmultiply(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as _,
            1,
            b.as_ptr(),
            n as _,
            1,
            0.0,
            c.as_mut_ptr(),
            n as _,
            1,
        )
    }
}

#[allow(unused_variables, unused_mut)]
pub fn cblas(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    #[cfg(feature = "blas")]
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as _,
            n as _,
            k as _,
            1.0,
            &a,
            k as _,
            &b,
            n as _,
            0.0,
            &mut c,
            n as _,
        )
    }
}

pub fn tract(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use tract_data::internal::*;
    use tract_linalg::frame::mmm::FusedSpec;
    unsafe {
        let mmm = tract_linalg::ops()
            .mmm(DatumType::F32, DatumType::F32, DatumType::F32, Some(m), Some(k), Some(n))
            .unwrap();
        let a_storage = mmm.a_packed(f32::datum_type().size_of(), k);
        let b_storage = mmm.b_packed(f32::datum_type().size_of(), k);
        let c_storage = mmm.c_view_with_axis(1, 0);

        let a = Tensor::from_shape(&[m, k], a).unwrap();
        let b = Tensor::from_shape(&[k, n], b).unwrap();
        let mut tc = Tensor::uninitialized_dt(f32::datum_type(), &[m,n]).unwrap();

        let mut pa = Tensor::uninitialized_aligned_dt(
            DatumType::F32,
            &[mmm.a_pack(k).len(m)],
            mmm.a_pack(k).alignment(),
        )
        .unwrap();
        let mut pb = Tensor::uninitialized_aligned_dt(
            DatumType::F32,
            &[mmm.b_pack(k).len(n)],
            mmm.b_pack(k).alignment(),
        )
        .unwrap();
        mmm.a_pack(k).pack(&mut pa.view_mut(), &a.view(), 1, 0);
        mmm.b_pack(k).pack(&mut pb.view_mut(), &b.view(), 0, 1);

        let mut scratch = mmm.allocate_scratch_space();

        mmm.run_with_scratch_space(
            m,
            n,
            &mut *scratch,
            &[
                FusedSpec::AddMatMul {
                    k,
                    a: a_storage.wrap(&pa.view()),
                    b: b_storage.wrap(&pb.view()),
                },
                FusedSpec::Store(c_storage.wrap(&mut tc.view_mut())),
            ],
        )
        .unwrap();
        c.copy_from_slice(tc.as_slice_unchecked())
    }
}
