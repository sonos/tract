use std::{fs, path};

use log::*;

use tract_core::internal::*;
use tract_onnx::pb::TensorProto;
use tract_onnx::*;

use std::convert::TryInto;

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = fs::read_dir(path)
        .map_err(|e| format!("accessing {:?}, {:?}", path, e))
        .unwrap()
        .filter(|d| d.as_ref().unwrap().file_name().to_str().unwrap().starts_with(prefix))
        .count();
    for i in 0..len {
        let filename = path.join(format!("{}_{}.pb", prefix, i));
        let mut file =
            fs::File::open(filename).map_err(|e| format!("accessing {:?}, {:?}", path, e)).unwrap();
        let tensor: TensorProto = ::protobuf::parse_from_reader(&mut file).unwrap();
        vec.push(tensor.try_into().unwrap())
    }
    debug!("{:?}: {:?}", path, vec);
    vec
}

pub fn load_dataset(path: &path::Path) -> (TVec<Tensor>, TVec<Tensor>) {
    (load_half_dataset("input", path), load_half_dataset("output", path))
}

pub fn run_one<P: AsRef<path::Path>>(root: P, test: &str, optim: bool) {
    setup_test_logger();
    let test_path = root.as_ref().join(test);
    let path = if test_path.join("data.json").exists() {
        use fs2::FileExt;
        let url = fs::read_to_string(test_path.join("data.json"))
            .unwrap()
            .split("\"")
            .find(|s| s.starts_with("https://"))
            .unwrap()
            .to_string();
        let f = fs::File::open(test_path.join("data.json")).unwrap();
        let _lock = f.lock_exclusive();
        let name: String =
            test_path.file_name().unwrap().to_str().unwrap().chars().skip(5).collect();
        info!("Locked {:?}", f);
        if !test_path.join(&name).exists() {
            let tgz_name = test_path.join(format!("{}.tgz", name));
            info!("Downloading {:?}", tgz_name);
            let wget = std::process::Command::new("wget")
                .arg("-q")
                .arg(&url)
                .arg("-O")
                .arg(&tgz_name)
                .status()
                .unwrap();
            if !wget.success() {
                panic!("wget: {:?}", wget);
            }
            let tar = std::process::Command::new("tar").arg("zxf").arg(&tgz_name).status().unwrap();
            if !tar.success() {
                panic!("tar: {:?}", tar);
            }
            fs::rename(&name, test_path.join(&name)).unwrap();
            fs::remove_file(&tgz_name).unwrap();
        }
        info!("Done with {:?}", f);
        test_path.join(&name)
    } else {
        test_path
    };
    let model_file = path.join("model.onnx");
    info!("Loading {:?}", model_file);
    let onnx = onnx();
    trace!("Proto Model:\n{:#?}", onnx.proto_model_for_path(&model_file));
    let mut model = onnx.model_for_path(&model_file).unwrap();
    info!("Analyse");
    trace!("Model:\n{:#?}", model);
    model.analyse(false).unwrap();
    info!("Incorporate");
    let model = model.incorporate().unwrap();
    info!("Check full inference");
    if model.missing_type_shape().unwrap().len() != 0 {
        panic!("Incomplete inference {:?}", model.missing_type_shape());
    }
    info!("Test model (optim: {:?}) {:#?}", optim, path);
    if optim {
        info!("Into type");
        let model = model.into_typed().unwrap();
        let optimized = model.into_optimized().unwrap();
        trace!("Run optimized model:\n{:#?}", optimized);
        run_model(optimized, &path)
    } else {
        trace!("Run analysed model:\n{:#?}", model);
        run_model(model, &path)
    };
}

fn run_model<TI, O>(model: ModelImpl<TI, O>, path: &path::Path)
where
    TI: Fact + Clone + 'static,
    O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let plan = SimplePlan::new(&model).unwrap();
    for d in fs::read_dir(path).unwrap() {
        let d = d.unwrap();
        if d.metadata().unwrap().is_dir()
            && d.file_name().to_str().unwrap().starts_with("test_data_set_")
        {
            let (inputs, expected) = load_dataset(&d.path());
            trace!("Loaded inputs: {:?}", inputs);
            trace!("Loaded output asserts: {:?}", expected);
            let computed = plan.run(inputs).unwrap();
            if computed.len() != expected.len() {
                panic!(
                    "For {:?}, different number of results: got:{} expected:{}",
                    d.file_name(),
                    computed.len(),
                    expected.len()
                );
            }
            for (ix, (a, b)) in computed.iter().zip(expected.iter()).enumerate() {
                use tract_core::error_chain::ChainedError;
                //                println!("computed: {:?}", computed[ix].dump(true));
                //                println!("expected: {:?}", expected[ix].dump(true));
                if let Err(e) = a.close_enough(b, true) {
                    panic!(
                        "For {:?}, different result for output #{}:\ngot:\n{:?}\nexpected:\n{:?}\n{}",
                        d.file_name(),
                        ix,
                        a.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                        b.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                        e.display_chain()
                    )
                }
            }
        }
    }
}
