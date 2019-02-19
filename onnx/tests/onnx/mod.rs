use std::{fs, path};

use tract_core::*;
use tract_onnx::pb::TensorProto;
use tract_onnx::*;

#[allow(dead_code)]
fn setup_test_logger() {
    use simplelog::{Config, LevelFilter, TermLogger};
    use std::sync::Once;

    static START: Once = Once::new();
    START.call_once(|| TermLogger::init(LevelFilter::Trace, Config::default()).unwrap());
}

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
        })
        .count();
    for i in 0..len {
        let filename = path.join(format!("{}_{}.pb", prefix, i));
        let mut file = fs::File::open(filename)
            .map_err(|e| format!("accessing {:?}, {:?}", path, e))
            .unwrap();
        let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
        vec.push(tensor.tractify().unwrap())
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

pub fn run_one<P: AsRef<path::Path>>(root: P, test: &str, optim: bool) {
//    setup_test_logger();
    let test_path = root.as_ref().join(test);
    let path = if test_path.join("data.json").exists() {
        use fs2::FileExt;
        let f = fs::File::open(test_path.join("data.json")).unwrap();
        let _lock = f.lock_exclusive();
        info!("Locked {:?}", f);
        let data: DataJson = ::serde_json::from_reader(&f).unwrap();
        if !test_path.join(&data.model_name).exists() {
            let (_, body) = ::mio_httpc::CallBuilder::get()
                .url(&data.url)
                .unwrap()
                .max_response(1_000_000_000)
                .timeout_ms(1_200_000)
                .exec()
                .unwrap();
            info!("Downloaded {:?}", data.url);
            let gz = ::flate2::read::GzDecoder::new(&*body);
            let mut tar = ::tar::Archive::new(gz);
            let tmp = test_path.join("tmp");
            let _ = fs::remove_dir_all(&tmp);
            tar.unpack(&tmp).unwrap();
            fs::rename(tmp.join(&data.model_name), test_path.join(&data.model_name)).unwrap();
            let _ = fs::remove_dir_all(&tmp);
        }
        info!("Done with {:?}", f);
        test_path.join(&data.model_name)
    } else {
        test_path
    };
    let model_file = path.join("model.onnx");
    debug!("Loading {:?}", model_file);
    let mut model = for_path(&model_file).unwrap();
    trace!(
        "Model: {:#?}",
        tract_onnx::model::model_proto_for_path(&model_file)
    );
    trace!("Model: {:#?}", model);
    model.analyse().unwrap();
    debug!("Loaded {:?}", model_file);
    if optim {
        model = model.into_optimized().unwrap();
    }
    if model.missing_type_shape().unwrap().len() != 0 {
        panic!("Incomplete inference {:?}", model.missing_type_shape());
    }
    trace!("Optimized model: {:#?}", model);
    let plan = SimplePlan::new(&model).unwrap();
    for d in fs::read_dir(path).unwrap() {
        let d = d.unwrap();
        if d.metadata().unwrap().is_dir()
            && d.file_name()
                .to_str()
                .unwrap()
                .starts_with("test_data_set_")
        {
            let (inputs, expected) = load_dataset(&d.path());
            // println!("inputs: {:?}", inputs[0].dump(true));
            let computed = plan.run(inputs).unwrap();
            if computed.len() != expected.len() {
                panic!(
                    "Different number of results: got:{} expected:{}",
                    computed.len(),
                    expected.len()
                );
            }
            for (ix, (a, b)) in computed.iter().zip(expected.iter()).enumerate() {
//                println!("computed: {:?}", computed[ix].dump(true));
//                println!("expected: {:?}", expected[ix].dump(true));
                if !a.close_enough(b, true) {
                    panic!(
                        "Different result for output #{}: got:{:?} expected:{:?}",
                        ix, a, b
                    )
                }
            }
        }
    }
}
