#[path = "ggml_suite.rs"]
mod ggml_suite;
#[path = "suite.rs"]
mod suite;

/// Whether a test can reach a `PrefixMatMul`, the one op whose Metal lowering depends on
/// the GEMM implementation. Everything else runs the same kernels whichever is picked.
fn reaches_gemm(path: &[String]) -> bool {
    match path[0].as_str() {
        "onnx" => {
            let name = path.last().unwrap().to_ascii_lowercase();
            ["matmul", "gemm", "einsum", "attention", "lstm", "gru", "rnn"]
                .iter()
                .any(|op| name.contains(op))
        }
        "unit" => ["bin_einsum", "sdpa", "moe_ffn", "matmul_q40", "matmul_f16", "matmul_rounding"]
            .contains(&path[1].as_str()),
        "pulse" => path[1] == "einsum",
        _ => true,
    }
}

fn gemm_subset(suite: &infra::TestSuite) -> infra::TestSuite {
    let mut suite = suite.clone();
    suite.skip(&|path| !reaches_gemm(path));
    suite
}

fn main() {
    suite::suite().test_runtime("mlx", "suite::suite()", "runtime()", "Approximation::Approximate");
    // matmul_f16 holds only when its products go through the ggml GEMV.
    let mut none = ggml_suite::suite().clone();
    none.ignore(
        &|t| matches!(t, [suite, section, ..] if suite == "unit" && section == "matmul_f16"),
    );
    none.test_runtime("none", "ggml_suite::suite()", "runtime()", "Approximation::Approximate");

    for gemm in ["mlx", "mfa", "mpp"] {
        gemm_subset(suite::suite()).test_runtime(
            &format!("{gemm}_gemm"),
            "suite::suite()",
            "runtime()",
            "Approximation::Approximate",
        );
    }
    gemm_subset(ggml_suite::suite()).test_runtime(
        "ggml_gemm",
        "ggml_suite::suite()",
        "runtime()",
        "Approximation::Approximate",
    );
}
