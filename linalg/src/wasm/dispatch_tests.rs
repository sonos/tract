#[cfg(test)]
mod dispatch_trace {
    fn trace_one(label: &str, m: Option<usize>, k: Option<usize>, n: Option<usize>) {
        let mut ops = crate::generic();
        crate::wasm::plug(&mut ops);
        let mmm = ops.mmm(tract_data::prelude::DatumType::F32, m, k, n).unwrap();
        eprintln!(
            "DFN3 {} (m={:?} k={:?} n={:?}) => {}  [mr={}, nr={}]",
            label,
            m,
            k,
            n,
            mmm.name(),
            mmm.mr(),
            mmm.nr()
        );
    }

    #[test]
    fn dfn3_shapes() {
        // DFN3 N=1 GEMV ops (the dominant matrix-vector cases)
        trace_one("lsnr_fc-style m=1 k=512", Some(1), Some(512), Some(1));
        trace_one("small m=16 k=96", Some(16), Some(96), Some(1));
        trace_one("medium m=32 k=256", Some(32), Some(256), Some(1));
        trace_one("GRU m=256 k=256", Some(256), Some(256), Some(1));
        trace_one("post-rnn m=256 k=512", Some(256), Some(512), Some(1));
        trace_one("frame-encoder m=64 k=96", Some(64), Some(96), Some(1));
        // N>1 sanity: should hit 8x8
        trace_one("MM m=64 k=64 n=8", Some(64), Some(64), Some(8));
    }

    /// Exercise every M-band edge of mmv_f32 to lock in the dispatch.
    /// Lower edge of each band = perfect-tile size; upper edge = last
    /// M before crossover to the next kernel.
    #[test]
    fn band_edges() {
        // 4x1 band: M ∈ 0..=4
        trace_one("band 4x1 lo m=1", Some(1), Some(64), Some(1));
        trace_one("band 4x1 hi m=4", Some(4), Some(64), Some(1));
        // 8x1 band: M ∈ 5..=8
        trace_one("band 8x1 lo m=5", Some(5), Some(64), Some(1));
        trace_one("band 8x1 hi m=8", Some(8), Some(64), Some(1));
        // 16x1 band: M ∈ 9..=16
        trace_one("band 16x1 lo m=9", Some(9), Some(64), Some(1));
        trace_one("band 16x1 hi m=16", Some(16), Some(64), Some(1));
        // 32x1 band: M ≥ 17
        trace_one("band 32x1 lo m=17", Some(17), Some(64), Some(1));
        trace_one("band 32x1 hi m=512", Some(512), Some(64), Some(1));
    }

    /// Regression guard for the GEMM/GEMV dispatch.
    ///
    /// `kernel_selection::strategize` honours the `mmm_f32` / `mmv_f32`
    /// callback only when the returned kernel is `ManuallyOptimized`;
    /// otherwise it falls through to `list_impls`, whose `retain()` drops
    /// every `TargetOptimized` kernel, and for N>1 then picks `max(nr*mr)`
    /// over the surviving `ManuallyOptimized` GEMV kernels — i.e.
    /// `wasm_f32_32x1`, a matrix×vector kernel, for every GEMM. So every
    /// kernel reachable through the dispatch callbacks must be
    /// `ManuallyOptimized`.
    #[test]
    fn dispatch_kernels_are_manually_optimized() {
        use crate::mmm::ImplementationQuality::ManuallyOptimized;
        let mut ops = crate::generic();
        crate::wasm::plug(&mut ops);
        for (label, m, k, n) in [
            ("GEMM m=64 k=64 n=8", 64, 64, 8),
            ("GEMM m=256 k=256 n=256", 256, 256, 256),
            ("GEMM m=1024 k=576 n=10", 1024, 576, 10),
            ("GEMV m=1 k=512 n=1", 1, 512, 1),
            ("GEMV m=256 k=256 n=1", 256, 256, 1),
        ] {
            let mmm =
                ops.mmm(tract_data::prelude::DatumType::F32, Some(m), Some(k), Some(n)).unwrap();
            assert_eq!(
                mmm.quality(),
                ManuallyOptimized,
                "{label}: dispatch returned {} tagged {:?} — strategize would \
                 discard it and reroute onto a GEMV kernel",
                mmm.name(),
                mmm.quality(),
            );
        }
    }
}
