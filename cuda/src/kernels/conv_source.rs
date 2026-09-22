use askama::Template;

const PREAMBLE: &str = r#"#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

"#;

/// One instantiation of the generic direct convolution kernel, by geometric rank and
/// datum type. The template is compiled into the binary; the parameters are runtime
/// values, so a rank or a type the build did not enumerate still renders.
#[derive(Template)]
#[template(path = "cnn.cu.j2", escape = "none")]
pub struct ConvSource {
    rank: usize,
    dtype: &'static str,
    ctype: &'static str,
    load: &'static str,
    load_close: &'static str,
    store: &'static str,
    store_close: &'static str,
    axes: Vec<usize>,
    /// Every axis but the last, in the order the linear thread index unwinds.
    axes_rev: Vec<usize>,
}

impl ConvSource {
    pub fn new(rank: usize, f16: bool) -> Self {
        ConvSource {
            rank,
            dtype: if f16 { "f16" } else { "f32" },
            ctype: if f16 { "__half" } else { "float" },
            load: if f16 { "__half2float(" } else { "" },
            load_close: if f16 { ")" } else { "" },
            store: if f16 { "__float2half(" } else { "" },
            store_close: if f16 { ")" } else { "" },
            axes: (1..=rank).collect(),
            axes_rev: (1..rank).rev().collect(),
        }
    }
}

/// Every instantiation the Cnn library carries: the ranks and types `Conv` dispatches to.
pub fn conv_library_source() -> String {
    let mut source = String::from(PREAMBLE);
    for f16 in [false, true] {
        for rank in 1..=4 {
            // Rendering into a String only fails on fmt::Error, which String never returns.
            source.push_str(&ConvSource::new(rank, f16).render().unwrap());
            source.push('\n');
        }
    }
    source
}
