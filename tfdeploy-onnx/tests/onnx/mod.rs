use std::{fs, path};

use tfdeploy::*;
use tfdeploy_onnx::pb::TensorProto;
use tfdeploy_onnx::*;

pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = fs::read_dir(path)
        .map_err(|e| format!("accessing {:?}, {:?}", path, e))
        .unwrap()
        .filter(|d| {
            d.as_ref()
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .starts_with(prefix)
        }).count();
    for i in 0..len {
        let filename = path.join(format!("{}_{}.pb", prefix, i));
        let mut file = fs::File::open(filename)
                .map_err(|e| format!("accessing {:?}, {:?}", path, e))
                    .unwrap();
        let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
        vec.push(tensor.to_tfd().unwrap())
    }
    vec
}

pub fn load_dataset(path: &path::Path) -> (TVec<Tensor>, TVec<Tensor>) {
    (
        load_half_dataset("input", path),
        load_half_dataset("output", path),
    )
}

#[derive(Debug, Serialize, Deserialize)]
struct DataJson {
    model_name: String,
    url: String,
}

pub fn run_one<P:AsRef<path::Path>>(root: P, test: &str) {
    let test_path = root.as_ref().join(test);
    let path = if test_path.join("data.json").exists() {
        use fs2::FileExt;
        let f = fs::File::open(test_path.join("data.json")).unwrap();
        let _lock = f.lock_exclusive();
        let data: DataJson = ::serde_json::from_reader(f).unwrap();
        if !test_path.join(&data.model_name).exists() {
            let (_, body) = ::mio_httpc::CallBuilder::get()
                .url(&data.url)
                .unwrap()
                .max_response(1_000_000_000)
                .timeout_ms(600_000)
                .exec()
                .unwrap();
            let gz = ::flate2::read::GzDecoder::new(&*body);
            let mut tar = ::tar::Archive::new(gz);
            let tmp = test_path.join("tmp");
            let _ = fs::remove_dir_all(&tmp);
            tar.unpack(&tmp).unwrap();
            fs::rename(tmp.join(&data.model_name), test_path.join(&data.model_name)).unwrap();
            let _ = fs::remove_dir_all(&tmp);
        }
        test_path.join(&data.model_name)
    } else {
        test_path
    };
    let path = path.join("model.onnx");
    let model = for_path(&path).unwrap();
    let plan = SimplePlan::for_model(&model).unwrap();
    for d in fs::read_dir(root).unwrap() {
        let d = d.unwrap();
        if d.metadata().unwrap().is_dir() && d
            .file_name()
            .to_str()
            .unwrap()
            .starts_with("test_data_set_")
        {
            let (inputs, expected) = load_dataset(&d.path());
            let computed = plan.run(inputs).unwrap().remove(0);
            for (a, b) in computed.iter().zip(expected.iter()) {
                if !a.close_enough(b, true) {
                    panic!("Different result: got:{:?} expected:{:?}", a, b)
                }
            }
        }
    }
}
