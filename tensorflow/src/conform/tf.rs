#![allow(dead_code)]

use std::{fs, path};

use tensorflow::DataType;
use tensorflow::FetchToken;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

use ndarray::ArrayD;
use tract_core::DtArray;

use std::collections::HashMap;
use std::collections::HashSet;

use conform::Result;

pub struct Tensorflow {
    session: Session,
    graph: Graph,
}

pub fn for_path<P: AsRef<path::Path>>(p: P) -> Result<Tensorflow> {
    use std::io::Read;
    let mut model = vec![];
    fs::File::open(p)?.read_to_end(&mut model)?;
    for_slice(&*model)
}

pub fn for_slice(buf: &[u8]) -> Result<Tensorflow> {
    let mut graph = Graph::new();
    graph.import_graph_def(buf, &::tensorflow::ImportGraphDefOptions::new())?;
    let session = Session::new(&::tensorflow::SessionOptions::new(), &graph)?;
    Ok(Tensorflow { session, graph })
}

enum TensorHolder {
    Bool(Tensor<bool>),
    F16(Tensor<::tensorflow::BFloat16>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    String(Tensor<i8>),
}

impl TensorHolder {
    fn to_tensor<T: ::tensorflow::TensorType + Copy>(m: ArrayD<T>) -> Tensor<T> {
        let dims: Vec<u64> = m.shape().iter().map(|d| *d as _).collect();
        let mut tensor = Tensor::<T>::new(&*dims);
        tensor.copy_from_slice(m.as_slice().unwrap());
        tensor
    }
}

impl From<DtArray> for TensorHolder {
    fn from(m: DtArray) -> TensorHolder {
        use tract_core::DatumType::*;
        use tract_core::TDim;
        match m.datum_type() {
            Bool => TensorHolder::Bool(Self::to_tensor(m.into_array().unwrap())),
            F16 => unimplemented!(),
            F32 => TensorHolder::F32(Self::to_tensor(m.into_array().unwrap())),
            F64 => TensorHolder::F64(Self::to_tensor(m.into_array().unwrap())),
            I8 => TensorHolder::I8(Self::to_tensor(m.into_array().unwrap())),
            I16 => TensorHolder::I16(Self::to_tensor(m.into_array().unwrap())),
            I32 => TensorHolder::I32(Self::to_tensor(m.into_array().unwrap())),
            I64 => TensorHolder::I64(Self::to_tensor(m.into_array().unwrap())),
            U8 => TensorHolder::U8(Self::to_tensor(m.into_array().unwrap())),
            U16 => TensorHolder::U16(Self::to_tensor(m.into_array().unwrap())),
            TDim => {
                let dims = m.to_array_view::<TDim>().unwrap();
                if dims.iter().all(|d| d.to_integer().is_ok()) {
                    let dims: ArrayD<i32> = dims.map(|d| d.to_integer().unwrap() as i32);
                    TensorHolder::I32(Self::to_tensor(dims))
                } else {
                    panic!("Streaming used in tensorflow settings")
                }
            }
            tract_core::DatumType::String => TensorHolder::String(Self::to_tensor(m.into_array().unwrap())),
        }
    }
}

fn tensor_to_array<T: ::tensorflow::TensorType>(tensor: &Tensor<T>) -> Result<ArrayD<T>> {
    let shape: Vec<usize> = tensor.dims().iter().map(|d| *d as _).collect();
    Ok(::ndarray::Array::from_iter(tensor.iter().cloned()).into_shape(shape)?)
}

impl Tensorflow {
    /// Executes the graph in one batch.
    pub fn run(&mut self, inputs: Vec<(&str, DtArray)>, output_name: &str) -> Result<Vec<DtArray>> {
        let tensors: Vec<(&str, TensorHolder)> = inputs
            .into_iter()
            .map(|(name, mat)| (name, mat.into()))
            .collect();

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

        let token = step.request_fetch(&self.graph.operation_by_name_required(output_name)?, 0);
        self.session.run(&mut step)?;

        let output_type = &self
            .graph
            .operation_by_name_required(&output_name)?
            .output_type(0);
        convert_output(&mut step, output_type, token)
    }

    /// Executes the graph in one batch, and returns the output for every node but the inputs.
    pub fn run_get_all(
        &mut self,
        inputs: Vec<(&str, DtArray)>,
    ) -> Result<HashMap<String, Vec<DtArray>>> {
        let mut tensors: Vec<(&str, TensorHolder)> = Vec::new();
        let mut excluded = HashSet::new();

        for (name, mat) in inputs {
            tensors.push((name, mat.into()));
            excluded.insert(name.to_string());
        }

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

        // Request the output of every node that's not an input.
        let mut tokens = HashMap::new();
        for operation in self.graph.operation_iter() {
            let name = operation.name()?;

            if excluded.contains(&name) {
                continue;
            }

            tokens.insert(name, step.request_fetch(&operation, 0));
        }

        // Execute the graph using tensorflow.
        self.session.run(&mut step)?;

        // Return the output for every node.
        let mut outputs = HashMap::new();
        for (name, token) in tokens {
            let output_type = &self.graph.operation_by_name_required(&name)?.output_type(0);
            outputs.insert(name, convert_output(&mut step, output_type, token)?);
        }

        Ok(outputs)
    }
}

/// Converts the output of a Tensorflow node into a Vec<DtArray>.
fn convert_output(
    step: &mut SessionRunArgs,
    output_type: &DataType,
    output: FetchToken,
) -> Result<Vec<DtArray>> {
    macro_rules! convert {
        ($dt:ident) => {
            tensor_to_array::<$dt>(&step.fetch(output)?)?.into()
        };
    };

    let tract_tensor = match output_type {
        DataType::Float => convert!(f32),
        DataType::UInt8 => convert!(u8),
        DataType::Int8 => convert!(i8),
        DataType::Int32 => convert!(i32),
        t => bail!("Missing Tensor to DtArray for type {:?}", t),
    };

    Ok(vec![tract_tensor])
}
