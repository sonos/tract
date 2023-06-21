use std::convert::TryInto;
use std::{fs, path};

use log::*;

use prost::Message;

use tract_onnx::tract_hir::internal::*;
use tract_onnx::pb::TensorProto;
use tract_onnx::tract_hir;

#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

pub fn load_half_dataset(prefix: &str, path: &path::Path) -> TVec<Tensor> {
    let mut vec = tvec!();
    let len = fs::read_dir(path)
        .map_err(|e| format!("accessing {path:?}, {e:?}"))
        .unwrap()
        .filter(|d| d.as_ref().unwrap().file_name().to_str().unwrap().starts_with(prefix))
        .count();
    for i in 0..len {
        let filename = path.join(format!("{prefix}_{i}.pb"));
        let bytes = bytes::Bytes::from(std::fs::read(filename).unwrap());
        let tensor = TensorProto::decode(bytes).unwrap();
        vec.push(tensor.try_into().unwrap())
    }
    debug!("{:?}: {:?}", path, vec);
    vec
}

pub fn run_one<P: AsRef<path::Path>>(
    root: P,
    test: &str,
    runtime: &dyn Runtime,
    more: &'static [&'static str],
) -> TractResult<()> {
    setup_test_logger();
    let test_path = root.as_ref().join(test);
    let path = if test_path.join("data.json").exists() {
        use fs2::FileExt;
        let url = fs::read_to_string(test_path.join("data.json"))?
            .split('\"')
            .find(|s| s.starts_with("https://"))
            .unwrap()
            .to_string();
        let f = fs::File::open(test_path.join("data.json"))?;
        let _lock = f.lock_exclusive();
        let name: String =
            test_path.file_name().unwrap().to_str().unwrap().chars().skip(5).collect();
        info!("Locked {:?}", f);
        if !test_path.join(&name).exists() {
            let tgz_name = test_path.join(format!("{name}.tgz"));
            info!("Downloading {:?}", tgz_name);
            let wget = std::process::Command::new("wget")
                .arg("-q")
                .arg(&url)
                .arg("-O")
                .arg(&tgz_name)
                .output()
                .expect("Failed to start wget");
            if !wget.status.success() {
                panic!("wget: {wget:?}");
            }
            let tar = std::process::Command::new("tar").arg("zxf").arg(&tgz_name).status()?;
            if !tar.success() {
                panic!("tar: {tar:?}");
            }
            fs::rename(&name, test_path.join(&name))?;
            fs::remove_file(&tgz_name)?;
        }
        info!("Done with {:?}", f);
        test_path.join(&name)
    } else {
        test_path
    };
    let model_file = path.join("model.onnx");
    info!("Loading {:?}", model_file);
    let mut onnx = tract_onnx::onnx();

    // hack: some tests (test_nonmaxsuppression_*) include the output shapes in the onnx model
    // even though there should be no way of knowing them at optimization time. This breaks
    // the solver.
    if more.contains(&"onnx-ignore-output-shape") {
        onnx = onnx.with_ignore_output_shapes(true);
    }
    // in some other cases, we need to deal with a tdim vs i64 mismatch (test for Shape, and Size)
    if more.contains(&"onnx-ignore-output-type") {
        onnx = onnx.with_ignore_output_types(true);
    }

    trace!("Proto Model:\n{:#?}", onnx.proto_model_for_path(&model_file));
    for d in fs::read_dir(&path)? {
        let mut model = onnx.model_for_path(&model_file)?;
        let d = d?;
        if d.metadata().unwrap().is_dir()
            && d.file_name().to_str().unwrap().starts_with("test_data_set_")
        {
            let data_path = d.path();
            let mut inputs = load_half_dataset("input", &data_path);
            for setup in more {
                if setup.starts_with("input:") {
                    let input = setup.split(':').nth(1).unwrap_or("");
                    let mut actual_inputs = vec![];
                    let mut actual_input_values = tvec![];
                    let input_outlets = model.input_outlets()?.to_vec();
                    for (ix, outlet) in input_outlets.iter().enumerate() {
                        if model.node(outlet.node).name == input {
                            actual_inputs.push(*outlet);
                            actual_input_values.push(inputs[ix].clone());
                        } else {
                            model.node_mut(outlet.node).op =
                                Box::new(tract_hir::ops::konst::Const::new(
                                    inputs[ix].clone().into_arc_tensor(),
                                ));
                        }
                    }
                    model.set_input_outlets(&actual_inputs)?;
                    inputs = actual_input_values;
                }
            }
            info!("Analyse");
            trace!("Model:\n{:#?}", model);
            model.analyse(false)?;
            model = model.incorporate()?;
            let model = model.into_typed()?.into_decluttered()?;
            info!("Test model (mode: {}) {:#?}", runtime.name(), path);
            let runnable = runtime.prepare(model)?;
            run_model(&*runnable, inputs, &data_path);
            info!("Test model (mode: {}) {:#?} OK.", runtime.name(), path);
        }
    }
    Ok(())
}

fn run_model(model: &dyn Runnable, inputs: TVec<Tensor>, data_path: &path::Path) {
    let expected = load_half_dataset("output", data_path);
    trace!("Loaded output asserts: {:?}", expected);
    let inputs = inputs.into_iter().map(|t| t.into_tvalue()).collect();
    let computed = model.run(inputs).unwrap();
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
