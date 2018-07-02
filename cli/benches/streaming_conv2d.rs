#[macro_use]
extern crate criterion;
extern crate tfdeploy;
extern crate ndarray;
extern crate rand;

use ndarray::Axis;
use tfdeploy::*;
use tfdeploy::tfpb::types::DataType;
use criterion::Criterion;

#[path = "../src/utils.rs"]
mod utils;

fn streaming_conv2d(c: &mut Criterion) {
    let datatype = DataType::DT_FLOAT;
    let model = tfdeploy::for_path("../tests/models/conv2d-large.pb").unwrap();
    let output = analyser::detect_output(&model).unwrap().unwrap();

    let data = utils::random_tensor(vec![41, 40], datatype);

    // Streaming execution.
    {
        let streaming_dims = vec![None, Some(40)];
        let streaming_inputs = vec![(1, StreamingInput::Streamed(datatype, streaming_dims))];
        let mut streaming_state = StreamingState::start(model.clone(), streaming_inputs, Some(output)).unwrap();

        let chunks = data.as_f32s().unwrap()
            .axis_iter(Axis(0))
            .map(|v| Tensor::F32(v.insert_axis(Axis(0)).to_owned()))
            .enumerate();

        for (i, chunk) in chunks.take(10) {
            let mut next_state = streaming_state.clone();
            next_state.step(1, chunk.clone()).unwrap();

            c.bench_function(
                format!("Streaming - Step {:?}", i).as_str(),
                move |b| b.iter(|| streaming_state.clone().step(1, chunk.clone()).unwrap())
            );

            streaming_state = next_state;
        }
    }

    // Regular execution.
    {
        let regular_inputs = vec![(1, data)];
        c.bench_function(
            "Regular",
            move |b| b.iter(|| model.run(regular_inputs.clone(), output))
        );
    }
}

criterion_group!(benches, streaming_conv2d);
criterion_main!(benches);