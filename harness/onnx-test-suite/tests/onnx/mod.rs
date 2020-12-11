use std::convert::TryInto;
use std::hash::Hash;
use std::{fs, path};

use log::*;

use prost::Message;

use tract_onnx::pb::TensorProto;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir;

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
        let bytes = bytes::Bytes::from(std::fs::read(filename).unwrap());
        let tensor = TensorProto::decode(bytes).unwrap();
        vec.push(tensor.try_into().unwrap())
    }
    debug!("{:?}: {:?}", path, vec);
    vec
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Mode {
    Plain,
    Optim,
    NNEF,
}

pub fn run_one<P: AsRef<path::Path>>(
    root: P,
    test: &str,
    mode: Mode,
    more: &'static [&'static str],
) -> TractResult<()> {
    use Mode::*;
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
                .expect("Failed to run wget");
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
    let nnef = tract_nnef::nnef().with_onnx();
    trace!("Proto Model:\n{:#?}", onnx.proto_model_for_path(&model_file));
    for d in fs::read_dir(&path)? {
        let mut model = onnx.model_for_path(&model_file)?;
        let d = d.unwrap();
        if d.metadata().unwrap().is_dir()
            && d.file_name().to_str().unwrap().starts_with("test_data_set_")
        {
            let data_path = d.path();
            let mut inputs = load_half_dataset("input", &data_path);
            for setup in more {
                if setup.starts_with("input:") {
                    let input = setup.split(":").nth(1).unwrap();
                    let mut actual_input = None;
                    let input_outlets = model.input_outlets().unwrap().to_vec();
                    for (ix, outlet) in input_outlets.iter().enumerate() {
                        if model.node(outlet.node).name == input {
                            actual_input = Some((outlet, inputs[ix].clone()));
                        } else {
                            model.node_mut(outlet.node).op =
                                Box::new(tract_hir::ops::konst::Const::new(
                                    inputs[ix].clone().into_arc_tensor(),
                                ));
                        }
                    }
                    let (outlet, value) = actual_input.unwrap_or_else(|| {
                        panic!(
                            "specified input: {}, input names: {:?}",
                            setup,
                            model
                                .input_outlets()
                                .unwrap()
                                .iter()
                                .map(|n| &model.node(n.node).name)
                                .collect::<Vec<_>>()
                        )
                    });
                    model.set_input_outlets(&[*outlet]).unwrap();
                    inputs = tvec!(value);
                }
            }
            info!("Analyse");
            trace!("Model:\n{:#?}", model);
            model.analyse(false).unwrap();
            info!("Incorporate");
            let model = model.incorporate().unwrap();
            info!("Test model (mode: {:?}) {:#?}", mode, path);
            match mode {
                Optim => {
                    info!("Check full inference");
                    if model.missing_type_shape().unwrap().len() != 0 {
                        panic!("Incomplete inference {:?}", model.missing_type_shape());
                    }
                    info!("Into type");
                    let model = model.into_typed().unwrap();
                    let optimized = model.declutter().unwrap().optimize().unwrap();
                    trace!("Run optimized model:\n{:#?}", optimized);
                    run_model(optimized, inputs, &data_path)
                }
                Plain => {
                    trace!("Run analysed model:\n{:#?}", model);
                    run_model(model, inputs, &data_path)
                }
                NNEF => {
                    let model = model.into_typed().unwrap();
                    info!("Declutter");
                    let optimized = model.declutter().unwrap();
                    info!("Store to NNEF");
                    let mut buffer = vec![];
                    nnef.write_to_tar(&optimized, &mut buffer).unwrap();
                    info!("Reload from NNEF");
                    let reloaded = nnef.model_for_read(&mut &*buffer).unwrap();
                    run_model(reloaded, inputs, &data_path)
                }
            }
            info!("Test model (mode: {:?}) {:#?} OK.", mode, path);
        }
    }
    Ok(())
}

fn run_model<F, O>(model: Graph<F, O>, inputs: TVec<Tensor>, data_path: &path::Path)
where
    F: Fact + Clone + 'static + Hash,
    O: std::fmt::Debug + std::fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    let plan = SimplePlan::new(&model).unwrap();
    let expected = load_half_dataset("output", data_path);
    trace!("Loaded output asserts: {:?}", expected);
    let computed = plan.run(inputs).unwrap();
    if computed.len() != expected.len() {
        panic!(
            "For {:?}, different number of results: got:{} expected:{}",
            data_path,
            computed.len(),
            expected.len()
        );
    }
    for (ix, (a, b)) in computed.iter().zip(expected.iter()).enumerate() {
        //                println!("computed: {:?}", computed[ix].dump(true));
        //                println!("expected: {:?}", expected[ix].dump(true));
        if let Err(e) = a.close_enough(b, true) {
            panic!(
                "For {:?}, different result for output #{}:\ngot:\n{:?}\nexpected:\n{:?}\n{}",
                data_path,
                ix,
                a.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                b.cast_to::<f32>().unwrap().to_array_view::<f32>().unwrap(),
                e //                e.display_chain()
            )
        }
    }
}
