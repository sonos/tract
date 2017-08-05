#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate maplit;
extern crate ndarray;
extern crate tensorflow as tf;
extern crate tfdeploy;

use std::{fs, path};
use std::collections::HashMap;

use ndarray::ArrayD;

error_chain!{
    foreign_links {
        Io(::std::io::Error);
    }
}

fn load_tensorflow_graph<P: AsRef<path::Path>>(p: P) -> Result<tf::Graph> {
    use std::io::Read;

    let mut buffer_model = Vec::new();
    fs::File::open(p)?.read_to_end(&mut buffer_model)?;
    let mut graph = tf::Graph::new();
    graph
        .import_graph_def(&*buffer_model, &tf::ImportGraphDefOptions::new())
        .map_err(|tfs| format!("Tensorflow error {:?}", tfs));
    Ok(graph)
}

fn tf2nd<D: tf::TensorType>(t: &tf::Tensor<D>) -> ArrayD<D> {
    let dims = ndarray::IxDyn(&*t.dims().iter().map(|a| *a as _).collect::<Vec<_>>());
    ArrayD::from_shape_vec(dims, t.to_vec()).unwrap()
}

fn nd2tf<D: tf::TensorType>(t: &ArrayD<D>) -> tf::Tensor<D> {
    use ndarray::Dimension;
    let dims = t.dim()
        .as_array_view()
        .iter()
        .map(|x| *x as _)
        .collect::<Vec<_>>();
    let mut tf: tf::Tensor<D> = tf::Tensor::new(&*dims);
    for (tf, t) in tf.iter_mut().zip(t.iter()) {
        *tf = *t
    }
    tf
}

fn run_tensorflow_graph(
    graph: &tf::Graph,
    inputs: &HashMap<&str, ArrayD<f32>>,
    outputs: &[&str],
) -> Vec<ArrayD<f32>> {
    let mut session = tf::Session::new(&tf::SessionOptions::new(), &graph).unwrap();
    let tf_inputs: Vec<tf::Tensor<f32>> = inputs.iter().map(|p| nd2tf(p.1)).collect();
    let mut step = tf::StepWithGraph::new();
    for (ix, (name, value)) in inputs.iter().enumerate() {
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
        .map(|o| tf2nd(&step.take_output(o).unwrap()))
        .collect()
}

#[test]
fn nd2tf2nd() {
    let a: ndarray::ArrayD<f32> = ndarray::arr2(&[[1., 2.], [3., 4.]])
        .into_shape(ndarray::IxDyn(&[2, 2]))
        .unwrap();
    let tf = nd2tf(&a);
    let a2 = tf2nd(&tf);
    assert_eq!(a2, a);
}

#[test]
fn op_const() {
    let ours = tfdeploy::GraphAnalyser::from_file("model.pb");
    let theirs = load_tensorflow_graph("model.pb");
    println!("ours: {:?}", ours.node_names());
    let a: ndarray::ArrayD<f32> = ndarray::arr2(&[[1., 2.], [3., 4.]])
        .into_shape(ndarray::IxDyn(&[2,2]))
        .unwrap();
    println!("{:?}", a);
    let outputs = run_tensorflow_graph(&theirs.unwrap(),
        &hashmap!("inputs" => a), &[&"word_cnn/ExpandDims_2"]);
    println!("{:?}", outputs[0]);
    panic!();
}
