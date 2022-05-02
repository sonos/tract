use futures::executor::block_on;
use tract_core::prelude::{DatumType, Tensor};
use tract_data::tvec;
use tract_gpu::GpuAccel;

fn main() {
    let gpu = block_on(GpuAccel::default()).unwrap();

    let x = 255;
    let y = 255;
    let mut data = Vec::new();
    for i in 1..(x * y + 1) {
        data.push(i as f32);
    }

    let inp =
        gpu.import_tensor("inp".to_string(), &Tensor::from_shape(&tvec![x, y], &data).unwrap());
    let a = gpu.create_storage_tensor("a".to_string(), DatumType::F32, tvec![x, y]);
    let out = gpu.create_out_tensor("out".to_string(), DatumType::F32, tvec![x, y]);

    gpu.tanh(&inp, &a);
    gpu.sigmoid(&a, &out);

    println!("{:#?}", block_on(gpu.tensor_move_out(out)));
}
