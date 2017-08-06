#[macro_use]
extern crate maplit;
extern crate ndarray;
extern crate tensorflow as tf;
extern crate tfdeploy;

use std::{fs, path};
use std::collections::HashMap;

use ndarray::ArrayD;
use tfdeploy::*;

fn load_tensorflow_graph<P: AsRef<path::Path>>(p: P) -> Result<tf::Graph> {
    use std::io::Read;

    let mut buffer_model = Vec::new();
    fs::File::open(p)?.read_to_end(&mut buffer_model)?;
    let mut graph = tf::Graph::new();
    graph
        .import_graph_def(&*buffer_model, &tf::ImportGraphDefOptions::new())
        .map_err(|tfs| format!("Tensorflow error {:?}", tfs))?;
    Ok(graph)
}

fn tf_to_mat_f32(t: &tf::Tensor<f32>) -> Matrix {
    let dims = ndarray::IxDyn(&*t.dims().iter().map(|a| *a as _).collect::<Vec<_>>());
    Matrix::F32(ArrayD::from_shape_vec(dims, t.to_vec()).unwrap())
}

fn tf_to_mat_i32(t: &tf::Tensor<i32>) -> Matrix {
    let dims = ndarray::IxDyn(&*t.dims().iter().map(|a| *a as _).collect::<Vec<_>>());
    Matrix::I32(ArrayD::from_shape_vec(dims, t.to_vec()).unwrap())
}

fn mat_f32_to_tf(t: &Matrix) -> tf::Tensor<f32> {
    use ndarray::Dimension;
    let t = t.as_f32s().unwrap();
    let dims = t.dim()
        .as_array_view()
        .iter()
        .map(|x| *x as _)
        .collect::<Vec<_>>();
    let mut tf: tf::Tensor<f32> = tf::Tensor::new(&*dims);
    for (tf, t) in tf.iter_mut().zip(t.iter()) {
        *tf = *t
    }
    tf
}

fn run_tensorflow_graph(
    graph: &tf::Graph,
    inputs: &HashMap<&str, Matrix>,
    outputs: &[&str],
) -> Vec<Matrix> {
    let mut session = tf::Session::new(&tf::SessionOptions::new(), &graph).unwrap();
    let tf_inputs: Vec<tf::Tensor<f32>> = inputs.iter().map(|p| mat_f32_to_tf(p.1)).collect();
    let mut step = tf::StepWithGraph::new();
    for (ix, (name, _)) in inputs.iter().enumerate() {
        step.add_input(
            &graph.operation_by_name_required(name).unwrap(),
            0,
            &tf_inputs[ix],
        );
    }
    let outputs: Vec<_> = outputs
        .iter()
        .map(|name| {
            step.request_output(&graph.operation_by_name_required(name).unwrap(), 0)
        })
        .collect();
    session.run(&mut step).unwrap();
    outputs
        .into_iter()
        .map(|o| tf_to_mat_f32(&step.take_output(o).unwrap()))
        .collect()
}

const INPUTS: [[f32; 40]; 334] = include!("../inputs.json");

fn frame() -> Matrix {
    use ndarray::IxDyn;
    Matrix::F32(
        ndarray::Array::from_shape_fn(IxDyn(&[82, 40]), |d| INPUTS[d[0]][d[1]])
    )
}

#[test]
fn nd2tf2nd() {
    let a: ndarray::ArrayD<f32> = ndarray::arr2(&[[1., 2.], [3., 4.]])
        .into_shape(ndarray::IxDyn(&[2, 2]))
        .unwrap();
    let a = Matrix::F32(a);
    let tf = mat_f32_to_tf(&a);
    let a2 = tf_to_mat_f32(&tf);
    assert_eq!(a2, a);
}

#[test]
fn op_const() {
    let ours = tfdeploy::GraphAnalyser::from_file("model.pb");
    let theirs = load_tensorflow_graph("model.pb");
    println!("ours: {:?}", ours.node_names());
    let a: ndarray::ArrayD<f32> = ndarray::arr2(&[[1., 2.], [3., 4.]])
        .into_shape(ndarray::IxDyn(&[2, 2]))
        .unwrap();
    let a = Matrix::F32(a);
    println!("{:?}", a);
    let outputs = run_tensorflow_graph(
        &theirs.unwrap(),
        &hashmap!("inputs" => a),
        &[&"word_cnn/ExpandDims_2"],
    );
    println!("{:?}", outputs[0]);
}

fn test_one(name: &str) {
    let mut ours = tfdeploy::GraphAnalyser::from_file("model.pb");
    let theirs = load_tensorflow_graph("model.pb");
    let input = frame();
    let outputs = run_tensorflow_graph(
        &theirs.unwrap(),
        &hashmap!("inputs" => input.clone()),
        &[name],
    );
//    println!("input shape {:?}", input.as_f32s().unwrap().dim());
    ours.set_value("inputs", input).unwrap();
    let our_output = ours.eval(name).unwrap().unwrap();
    assert!(outputs[0].as_f32s().unwrap().all_close(our_output[0].as_f32s().unwrap(), 0.001));
    println!("name: {} me:{:?} tf:{:?}", name, our_output[0].as_f32s().unwrap().dim(), outputs[0].as_f32s().unwrap().dim(), );
}

#[test]
fn test_net() {
    let mut ours = tfdeploy::GraphAnalyser::from_file("model.pb");
    /*
    println!("{:?}", ours.node_names());
    test_one("word_cnn/ExpandDims_2");
    test_one("word_cnn/ExpandDims_3");
    */
    test_one("word_cnn/layer_0_2/convolution");
    /*
    test_one("word_cnn/layer_0_2/BiasAdd");
    test_one("logits");
    */
}
