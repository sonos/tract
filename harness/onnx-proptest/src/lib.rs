#![allow(dead_code)]
use onnx_helpers::builder::Graph;
use onnx_helpers::nodes::Node;
use onnx_helpers::prelude::*;
use onnx_pb::tensor_proto::DataType;
use onnx_pb::Attribute;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, LoggingLevel};
use proptest::{
    arbitrary::Arbitrary,
    collection::vec,
    strategy::{BoxedStrategy, Strategy},
};
use proptest::{option, prelude::*};
use prost::Message;
use tract_onnx::prelude::*;

struct Problem {
    model: Vec<u8>,
    inputs: TVec<Tensor>,
}

impl Problem {
    pub fn onnxrt(&self) -> TractResult<TVec<Tensor>> {
        let environment = Environment::builder()
            .with_name("test")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;

        let mut session = environment.new_session_builder()?.with_model_from_memory(&self.model)?;
        let input = self.inputs.iter().map(|i| i.clone().into_array::<f32>().unwrap()).collect();
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)?;
        Ok(outputs.into_iter().map(|a| (*a).to_owned().into_tensor()).collect())
    }

    pub fn tract(&self) -> TractResult<TVec<Tensor>> {
        let tract = tract_onnx::onnx()
            .model_for_read(&mut &*self.model)?
            .into_optimized()?
            .into_runnable()?;

        let output = tract.run(self.inputs.iter().cloned().collect())?;
        Ok(output.into_iter().map(|t| t.into_tensor()).collect())
    }

    pub fn check(&self) -> TractResult<()> {
        self.onnxrt()?
            .into_iter()
            .zip(self.tract()?.into_iter())
            .map(|(o, t)| o.close_enough(&t, true))
            .collect()
    }
}

pub fn tensor(shape: &[usize]) -> BoxedStrategy<Tensor> {
    let shape = shape.to_owned();
    let len = shape.iter().product::<usize>();
    vec(any::<i8>().prop_map(|i| i as f32), len..=len)
        .prop_map(move |vec| tensor1(&vec).into_shape(&shape).unwrap())
        .boxed()
}

#[derive(Debug)]
struct GruProblem {
    hidden_size: usize,
    x: Tensor,
    w: Tensor,
    r: Tensor,
    bias: Option<Tensor>,
    sl: Option<Tensor>,
    initial_h: Option<Tensor>,
    linear_before_reset: bool,
}

impl Arbitrary for GruProblem {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;
    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (any::<bool>(), 1usize..5, 1usize..3, 1usize..5, 1usize..5)
            .prop_flat_map(|(bidi, s, b, i, h)| {
                let dir = 1 + bidi as usize;
                let x = tensor(&[s, b, i]);
                let w = tensor(&[dir, 3 * h, i]);
                let r = tensor(&[dir, 3 * h, h]);
                let bias = option::of(tensor(&[dir, 6 * h]));
                /*
                let sl = if s > 1 {
                option::of(vec(1..s as i32, b..=b).prop_map(|v| tensor1(&*v))).boxed()
                } else {
                Just(None).boxed()
                };
                */
                let sl = Just(None).boxed();
                let initial_h = option::of(tensor(&[dir, b, h]));
                let linear_before_reset = any::<bool>();
                (x, w, r, bias, sl, Just(h), initial_h, linear_before_reset)
            })
            .prop_map(|(x, w, r, bias, sl, h, initial_h, linear_before_reset)| GruProblem {
                hidden_size: h,
                x,
                w,
                r,
                bias,
                sl,
                initial_h,
                linear_before_reset,
            })
            .boxed()
    }
}

impl GruProblem {
    fn i(&self) -> usize {
        self.x.shape()[2]
    }
    fn b(&self) -> usize {
        self.x.shape()[1]
    }
    fn lower(&self) -> TractResult<Problem> {
        let mut graph = builder::Graph::new("gru");
        lower_input(&mut graph, "x", self.x.shape());
        lower_const_f32(&mut graph, "w", &self.w);
        lower_const_f32(&mut graph, "r", &self.r);
        let dirs = self.w.shape()[0];
        let mut inputs = vec!["x", "wO", "rO", "", "", ""];
        if let Some(b) = &self.bias {
            lower_const_f32(&mut graph, "b", &b);
            inputs[3] = "bO";
        }
        if let Some(sl) = &self.sl {
            lower_const_i32(&mut graph, "sl", &sl);
            inputs[4] = "slO";
        }
        if let Some(h) = &self.initial_h {
            lower_const_f32(&mut graph, "initial_h", &h);
            inputs[5] = "initial_hO";
        }
        graph
            .node("gru")
            .inputs(inputs)
            .outputs(["gru0", "gru1"])
            .op("GRU")
            .attribute("hidden_size", Attribute::Int(self.hidden_size as _))
            .attribute(
                "direction",
                Attribute::String(if dirs == 2 { "bidirectional" } else { "forward" }.to_string()),
            )
            .attribute(
                "linear_before_reset",
                Attribute::Int(self.linear_before_reset as usize as _),
            )
            .build();
        graph = graph.outputs("gru0").outputs("gru1");
        let model = graph.model().build();
        let mut buffer = vec![];
        model.encode(&mut buffer)?;
        Ok(Problem { model: buffer, inputs: tvec!(self.x.clone()) })
    }
}

fn lower_input(graph: &mut Graph, name: &str, shape: &[usize]) -> Node {
    let mut node = graph.input(name).typed(DataType::Float);
    for dim in shape {
        node = node.dim(*dim as i64);
    }
    node.node()
}

fn lower_const_f32(graph: &mut Graph, name: &str, tensor: &Tensor) -> Node {
    let mut t = onnx_pb::TensorProto::default();
    t.float_data = tensor.as_slice::<f32>().unwrap().to_vec();
    t.data_type = onnx_pb::tensor_proto::DataType::Float as _;
    t.dims = tensor.shape().iter().map(|d| *d as i64).collect();
    t.name = name.to_string();
    graph.constant(name, t)
}

fn lower_const_i32(graph: &mut Graph, name: &str, tensor: &Tensor) -> Node {
    let mut t = onnx_pb::TensorProto::default();
    t.int32_data = tensor.as_slice::<i32>().unwrap().to_vec();
    t.data_type = onnx_pb::tensor_proto::DataType::Int32 as _;
    t.dims = tensor.shape().iter().map(|d| *d as i64).collect();
    t.name = name.to_string();
    graph.constant(name, t)
}

proptest! {
    #[test]
    fn gru_prop(pb in any::<GruProblem>()) {
        pb.lower().unwrap().check().unwrap()
    }
}

#[test]
fn gru_0() {
    let pb = GruProblem {
        hidden_size: 1,
        x: tensor3(&[[[1f32]]]),
        w: tensor3(&[[[0f32], [0f32], [0f32]]]),
        r: tensor3(&[[[0f32], [0f32], [0f32]]]),
        bias: None,
        sl: None,
        initial_h: None,
        linear_before_reset: false,
    };
    pb.lower().unwrap().check().unwrap()
}

/*
#[test]
fn gru_sl_0() {
let pb = GruProblem {
hidden_size: 1,
x: tensor3(&[[[0f32]], [[0f32]]]),
w: tensor3(&[[[0f32], [0f32], [0f32]]]),
r: tensor3(&[[[0f32], [0f32], [0f32]]]),
bias: None,
sl: None,
};
pb.lower().unwrap().check().unwrap()
}
*/

#[test]
fn gru_linear_before_reset_0() {
    let pb = GruProblem {
        hidden_size: 2,
        x: tensor3(&[[[1f32]]]),
        w: tensor3(&[[[0f32], [0f32], [0f32], [0f32], [4f32], [0f32]]]),
        r: tensor3(&[[
            [0f32, 0f32],
            [0f32, 0f32],
            [0f32, 1f32],
            [0f32, 0f32],
            [0f32, 0f32],
            [0f32, 0f32],
        ]]),
        bias: Some(tensor2(&[[0f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., -2., 0.]])),
        initial_h: None,
        sl: None,
        linear_before_reset: true,
    };
    pb.lower().unwrap().check().unwrap()
}

#[test]
fn gru_linear_before_reset_1() {
    let pb = GruProblem {
        hidden_size: 2,
        x: tensor3(&[[[0f32], [0f32], [0f32], [0f32]]]),
        w: tensor3(&[[[0f32], [0f32], [0f32], [0f32], [0f32], [0f32]]]),
        r: tensor3(&[[
            [0f32, 0f32],
            [0f32, -1f32],
            [0f32, 0f32],
            [0f32, -1f32],
            [0f32, 0f32],
            [35f32, 16f32],
        ]]),
        bias: None,
        initial_h: Some(tensor3(&[[[51f32, 60.], [0., 0.], [0., 0.], [0., 0.]]])),
        sl: None,
        linear_before_reset: true,
    };
    pb.lower().unwrap().check().unwrap()
}

#[test]
fn gru_linear_before_reset_2() {
    let pb = GruProblem {
        hidden_size: 2,
        x: tensor3(&[[[0f32]]]),
        w: tensor3(&[[[0f32], [0f32], [0f32], [0f32], [0f32], [0f32]]]),
        r: tensor3(&[[
            [0f32, 0f32],
            [0f32, -1f32],
            [0f32, 0f32],
            [0f32, -1f32],
            [0f32, 0f32],
            [35f32, 16f32],
        ]]),
        bias: None,
        initial_h: Some(tensor3(&[[[51f32, 60.]]])),
        sl: None,
        linear_before_reset: true,
    };
    pb.lower().unwrap().check().unwrap()
}
