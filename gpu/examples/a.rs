use futures::executor::block_on;
use tract_core::prelude::{DatumType, Tensor};
use tract_data::tvec;
use tract_gpu::GpuAccel;

fn main() {
    let gpu = block_on(GpuAccel::default()).unwrap();

    let inp = gpu.import_tensor(
        "inp".to_string(),
        &Tensor::from_shape(&tvec![2, 2], &vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]).unwrap(),
    );
    let a = gpu.create_storage_tensor("a".to_string(), DatumType::F32, tvec![2, 2]);
    let out = gpu.create_out_tensor("out".to_string(), DatumType::F32, tvec![2, 2]);

    gpu.tanh(&inp, &a);
    gpu.sigmoid(&a, &out);

    println!("{:#?}", block_on(gpu.tensor_move_out(out)));
}
