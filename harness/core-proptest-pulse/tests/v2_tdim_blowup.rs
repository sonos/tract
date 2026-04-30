//! Isolate the TDim simplifier blowup that hangs PulseV2 on deep conv chains.
//!
//! Build synthetic chains of dilated 1D convs of increasing depth and report
//! how long `PulseV2Model::new` takes plus the size of the output fact shape
//! expressions. We expect time (or expression size) to grow fast past some N.

use std::time::Instant;

use tract_core::ops::array::Slice;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cnn::pools::PoolSpec;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::ops::math::Add;
use tract_core::ops::nn::DataFormat;
use tract_core::prelude::*;
use tract_pulse::v2::PulseV2Model;

/// Wire a 1D Conv with the given kernel size and dilation, NCHW layout.
fn wire_conv(
    model: &mut TypedModel,
    name: &str,
    input: OutletId,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
) -> OutletId {
    let pool_spec = PoolSpec::new(
        DataFormat::NCHW,
        tvec!(kernel),
        tract_core::ops::cnn::PaddingSpec::Valid,
        Some(tvec!(dilation)),
        Some(tvec!(1)),
        in_channels,
        out_channels,
    );
    let kernel_t = Tensor::zero::<f32>(&[out_channels, in_channels, kernel]).unwrap();
    let kernel_outlet = model.add_const(format!("{name}.k"), kernel_t).unwrap();
    let bias_t = Tensor::zero::<f32>(&[out_channels]).unwrap();
    let bias_outlet = model.add_const(format!("{name}.b"), bias_t).unwrap();

    let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };
    model.wire_node(name, conv, &[input, kernel_outlet, bias_outlet]).unwrap()[0]
}

fn dilated_conv_chain(n_layers: usize) -> (TypedModel, Symbol) {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    // NCHW layout: [1, channels=1, time=S]
    let fact = f32::fact([1.to_dim(), 1.to_dim(), s.to_dim()]);
    let mut wire = model.add_source("input", fact).unwrap();
    for i in 0..n_layers {
        let dilation = 1usize << i;
        wire = wire_conv(&mut model, &format!("conv_{i}"), wire, 1, 1, 3, dilation);
    }
    model.select_output_outlets(&[wire]).unwrap();
    (model, s)
}

/// Build a WaveNet-like residual chain: each block has `conv → conv` and a
/// skip connection from the block input that gets sliced to align with the
/// block output, then added back. Skip-Adds are where parallel paths merge —
/// this is what blows broadcast Max in v2.
fn skip_residual_chain(n_blocks: usize) -> (TypedModel, Symbol) {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let fact = f32::fact([1.to_dim(), 1.to_dim(), s.to_dim()]);
    let mut wire = model.add_source("input", fact).unwrap();
    let mut produced_overlap = 0i64; // cumulative receptive-field consumed so far
    for i in 0..n_blocks {
        let dilation = 1usize << (i % 4);
        let block_in = wire;
        let conv = wire_conv(&mut model, &format!("b{i}_c1"), block_in, 1, 1, 3, dilation);
        let conv2 = wire_conv(&mut model, &format!("b{i}_c2"), conv, 1, 1, 3, dilation);
        produced_overlap += 4 * dilation as i64; // two convs each consume 2*dilation
        // Slice block_in to align with conv2's effective length.
        // block_in has shape on streaming axis = S - produced_overlap_before.
        // conv2 has shape = S - produced_overlap. Skip needs slice of length
        // S - produced_overlap, taking the central window (offset 4*dilation).
        let begin = (4 * dilation) as usize;
        let slice = model
            .wire_node(
                format!("b{i}_skip"),
                Slice { axis: 2, start: TDim::Val(begin as i64), end: s.to_dim() },
                &[block_in],
            )
            .unwrap()[0];
        let _ = produced_overlap;
        wire = model
            .wire_node(format!("b{i}_add"), TypedBinOp(Box::new(Add), None), &[conv2, slice])
            .unwrap()[0];
    }
    model.select_output_outlets(&[wire]).unwrap();
    (model, s)
}

#[test]
fn skip_residual_chain_grows() {
    for n in 1..=8 {
        let (model, s) = skip_residual_chain(n);
        let t0 = Instant::now();
        let result = PulseV2Model::new(&model, s);
        let elapsed = t0.elapsed();
        match result {
            Ok(pv2) => {
                let typed = pv2.into_typed().unwrap();
                let out_fact = typed.output_fact(0).unwrap();
                let shape_str = format!("{:?}", out_fact.shape);
                let len = shape_str.len();
                println!(
                    "n={n}  {}ms  shape_str_len={len}  shape={shape_str}",
                    elapsed.as_millis()
                );
            }
            Err(e) => {
                println!("n={n}  {}ms  ERROR: {:#}", elapsed.as_millis(), e);
                break;
            }
        }
        if elapsed.as_secs() > 10 {
            println!("(bail — >10s)");
            break;
        }
    }
}

#[test]
fn dilated_chain_grows() {
    for n in 1..=30 {
        let (model, s) = dilated_conv_chain(n);
        let t0 = Instant::now();
        let result = PulseV2Model::new(&model, s);
        let elapsed = t0.elapsed();
        match result {
            Ok(pv2) => {
                let typed = pv2.into_typed().unwrap();
                let out_fact = typed.output_fact(0).unwrap();
                let shape_str = format!("{:?}", out_fact.shape);
                let len = shape_str.len();
                println!(
                    "n={n:2}  {:>7}ms  shape_str_len={len:5}  shape={shape_str}",
                    elapsed.as_millis()
                );
            }
            Err(e) => {
                println!("n={n}  {}ms  ERROR: {:#}", elapsed.as_millis(), e);
                break;
            }
        }
        if elapsed.as_secs() > 10 {
            println!("(bail — >10s)");
            break;
        }
    }
}
