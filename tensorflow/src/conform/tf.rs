#![allow(dead_code)]

use std::{fs, path};

use tensorflow as tf;
use tensorflow::DataType;
use tensorflow::FetchToken;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionRunArgs;

use tract_hir::internal::*;
use tract_ndarray::prelude::*;

use std::collections::HashMap;
use std::collections::HashSet;

pub struct Tensorflow {
    graph: Graph,
}

pub fn version() -> String {
    tf::version().unwrap()
}

pub fn for_path<P: AsRef<path::Path>>(p: P) -> TractResult<Tensorflow> {
    use std::io::Read;
    let mut model = vec![];
    fs::File::open(p)?.read_to_end(&mut model)?;
    for_slice(&*model)
}

pub fn for_slice(buf: &[u8]) -> TractResult<Tensorflow> {
    let mut graph = Graph::new();
    graph.import_graph_def(buf, &::tensorflow::ImportGraphDefOptions::new())?;
    Ok(Tensorflow { graph })
}

enum TensorHolder {
    Bool(tf::Tensor<bool>),
    F16(tf::Tensor<::tensorflow::BFloat16>),
    F32(tf::Tensor<f32>),
    F64(tf::Tensor<f64>),
    U8(tf::Tensor<u8>),
    U16(tf::Tensor<u16>),
    I8(tf::Tensor<i8>),
    I16(tf::Tensor<i16>),
    I32(tf::Tensor<i32>),
    I64(tf::Tensor<i64>),
    String(tf::Tensor<i8>),
}

impl TensorHolder {
    fn to_tensor<T: ::tensorflow::TensorType + Copy>(m: ArrayD<T>) -> tf::Tensor<T> {
        let dims: Vec<u64> = m.shape().iter().map(|d| *d as _).collect();
        let mut tensor = tf::Tensor::<T>::new(&*dims);
        tensor.copy_from_slice(m.as_slice().unwrap());
        tensor
    }
}

impl From<Tensor> for TensorHolder {
    fn from(m: Tensor) -> TensorHolder {
        match m.datum_type() {
            DatumType::Bool => TensorHolder::Bool(Self::to_tensor(m.into_array().unwrap())),
            DatumType::F16 => unimplemented!(),
            DatumType::F32 => TensorHolder::F32(Self::to_tensor(m.into_array().unwrap())),
            DatumType::F64 => TensorHolder::F64(Self::to_tensor(m.into_array().unwrap())),
            DatumType::I8 => TensorHolder::I8(Self::to_tensor(m.into_array().unwrap())),
            DatumType::I16 => TensorHolder::I16(Self::to_tensor(m.into_array().unwrap())),
            DatumType::I32 => TensorHolder::I32(Self::to_tensor(m.into_array().unwrap())),
            DatumType::I64 => TensorHolder::I64(Self::to_tensor(m.into_array().unwrap())),
            DatumType::U8 => TensorHolder::U8(Self::to_tensor(m.into_array().unwrap())),
            DatumType::U16 => TensorHolder::U16(Self::to_tensor(m.into_array().unwrap())),
            DatumType::U32 => TensorHolder::U16(Self::to_tensor(m.into_array().unwrap())),
            DatumType::U64 => TensorHolder::U16(Self::to_tensor(m.into_array().unwrap())),
            DatumType::QU8(_) => TensorHolder::U8(Self::to_tensor(m.into_array().unwrap())),
            DatumType::QI8(_) => TensorHolder::I8(Self::to_tensor(m.into_array().unwrap())),
            DatumType::QI32(_) => TensorHolder::I32(Self::to_tensor(m.into_array().unwrap())),
            #[cfg(feature="complex")]
            DatumType::ComplexI16 => unimplemented!(),
            #[cfg(feature="complex")]
            DatumType::ComplexI32 => unimplemented!(),
            #[cfg(feature="complex")]
            DatumType::ComplexI64 => unimplemented!(),
            #[cfg(feature="complex")]
            DatumType::ComplexF16 => unimplemented!(),
            #[cfg(feature="complex")]
            DatumType::ComplexF32 => unimplemented!(),
            #[cfg(feature="complex")]
            DatumType::ComplexF64 => unimplemented!(),
            DatumType::TDim => {
                let dims = m.to_array_view::<TDim>().unwrap();
                if let Ok(dims) = dims.iter().map(|d| d.to_i32()).collect::<TractResult<Vec<_>>>() {
                    TensorHolder::I32(Self::to_tensor(arr1(&dims).into_dyn()))
                } else {
                    panic!("Streaming used in tensorflow settings")
                }
            }
            DatumType::String => TensorHolder::String(Self::to_tensor(m.into_array().unwrap())),
            DatumType::Blob => TensorHolder::String(Self::to_tensor(m.into_array().unwrap())),
            DatumType::Opaque => panic!("No support for Opaque DT in tensorflow"),
        }
    }
}

fn tensor_to_array<T: ::tensorflow::TensorType>(tensor: &tf::Tensor<T>) -> TractResult<ArrayD<T>> {
    let shape: Vec<usize> = tensor.dims().iter().map(|d| *d as _).collect();
    Ok(Array::from(tensor.into_iter().cloned().collect::<Vec<_>>()).into_shape_with_order(shape)?)
}

impl Tensorflow {
    /// Executes the graph in one batch.
    pub fn run(
        &mut self,
        inputs: Vec<(&str, Tensor)>,
        output_name: &str,
    ) -> TractResult<Vec<Tensor>> {
        let tensors: Vec<(&str, TensorHolder)> =
            inputs.into_iter().map(|(name, mat)| (name, mat.into())).collect();

        let mut step = SessionRunArgs::new();
        for t in &tensors {
            let op = self.graph.operation_by_name_required(t.0)?;
            match t.1 {
                TensorHolder::Bool(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::U8(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::U16(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I8(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I16(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I32(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I64(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::F16(_) => unimplemented!(),
                TensorHolder::F32(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::F64(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::String(ref it) => step.add_feed(&op, 0, &it),
            }
        }

        let op = &self.graph.operation_by_name_required(output_name)?;
        let tokens =
            (0..op.num_outputs()).map(|ix| step.request_fetch(&op, ix as i32)).collect::<Vec<_>>();

        let mut session = Session::new(&::tensorflow::SessionOptions::new(), &self.graph)?;
        session.run(&mut step)?;

        tokens
            .into_iter()
            .enumerate()
            .map(|(ix, tok)| {
                let output_type =
                    &self.graph.operation_by_name_required(&output_name)?.output_type(ix);
                convert_output(&mut step, output_type, tok)
            })
            .collect()
    }

    /// Executes the graph in one batch, and returns the output for every node but the inputs.
    pub fn run_get_many<'a>(
        &mut self,
        inputs: Vec<(&'a str, Tensor)>,
        targets: Vec<&'a str>,
    ) -> TractResult<HashMap<&'a str, Vec<Tensor>>> {
        let mut input_pairs: Vec<(&str, TensorHolder)> = Vec::new();
        let mut excluded = HashSet::new();

        for (name, mat) in inputs {
            input_pairs.push((name, mat.into()));
            excluded.insert(name.to_string());
        }

        let mut step = SessionRunArgs::new();
        for t in &input_pairs {
            let op = self.graph.operation_by_name_required(t.0)?;
            match t.1 {
                TensorHolder::Bool(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::U8(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::U16(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I8(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I16(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I32(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::I64(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::F16(_) => unimplemented!(),
                TensorHolder::F32(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::F64(ref it) => step.add_feed(&op, 0, &it),
                TensorHolder::String(ref it) => step.add_feed(&op, 0, &it),
            }
        }

        let mut tokens = HashMap::new();
        trace!("Targets: {:?}", targets);
        for name in targets {
            if excluded.contains(name) {
                continue;
            }

            if let Some(operation) = self.graph.operation_by_name(name)? {
                // switch only computes one of its outputs. tf explodes during
                // the call to run() if we registers them
                if operation.op_type()? == "Switch" {
                    continue;
                }

                // this one pretends to have 5 outputs, but has only one
                if operation.op_type()? == "FusedBatchNorm" {
                    continue;
                }

                let outputs = (0..operation.num_outputs())
                    .map(|ix| step.request_fetch(&operation, ix as i32))
                    .collect::<Vec<_>>();

                tokens.insert(name, outputs);
            }
        }
        trace!("Generated all output tokens");
        trace!("{:?}", tokens);

        // Execute the graph using tensorflow.
        let mut session = Session::new(&::tensorflow::SessionOptions::new(), &self.graph)?;
        session.run(&mut step)?;
        trace!("Tensorflow ran succesfully");

        // Return the output for every node.
        let mut outputs = HashMap::new();
        for (name, tokens) in tokens {
            let tensors = tokens
                .iter()
                .enumerate()
                .map(|(ix, tok)| {
                    let output_type =
                        &self.graph.operation_by_name_required(&name)?.output_type(ix);
                    convert_output(&mut step, output_type, *tok)
                })
                .collect::<TractResult<Vec<_>>>()?;
            outputs.insert(name, tensors);
        }

        Ok(outputs)
    }
}

/// Converts the output of a Tensorflow node into a Tensor.
fn convert_output(
    step: &mut SessionRunArgs,
    output_type: &DataType,
    output: FetchToken,
) -> TractResult<Tensor> {
    macro_rules! convert {
        ($dt:ident) => {
            match step.fetch(output) {
                Err(r) => Err(r)?,
                Ok(output) => tensor_to_array::<$dt>(&output)?.into(),
            }
        };
    };

    let tract_tensor = match output_type {
        DataType::Bool => convert!(bool),
        DataType::Float => convert!(f32),
        DataType::Double => convert!(f64),
        DataType::UInt8 => convert!(u8),
        DataType::Int8 => convert!(i8),
        DataType::Int32 => convert!(i32),
        DataType::Int64 => convert!(i64),
        t => bail!("Missing conversion for tensorflow to tract (type: {:?})", t),
    };

    Ok(tract_tensor)
}
