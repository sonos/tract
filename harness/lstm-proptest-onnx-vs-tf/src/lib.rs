#![allow(dead_code)]

use proptest::prelude::*;

use tract_hir::internal::*;
use tract_ndarray::prelude::*;
use tract_ndarray::Order;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir;

#[derive(Clone, Debug)]
pub struct LstmProblem {
    pub length: usize,
    pub batch_size: usize,
    pub cell_size: usize,
    pub x: TValue,
    pub w_xh_icfo: Array2<f32>,
    pub b_icfo: Array1<f32>,
    pub h0: Array2<f32>,
    pub c0: Array2<f32>,
}

impl LstmProblem {
    pub fn onnx_model(&self) -> TractResult<TypedModel> {
        let mut model = InferenceModel::default();
        let s = self.cell_size;
        let mut w_iofc = Array2::zeros((s, 4 * s));
        let mut r_iofc = Array2::zeros((s, 4 * s));
        let mut b_iofc = Array1::zeros(8 * s);
        let w_x_icfo = self.w_xh_icfo.slice_axis(Axis(0), (0..s).into());
        let w_h_icfo = self.w_xh_icfo.slice_axis(Axis(0), (s..2 * s).into());
        for (icfo, iofc) in [0, 3, 2, 1].iter().enumerate() {
            w_iofc
                .slice_axis_mut(Axis(1), (s * icfo..s * (icfo + 1)).into())
                .assign(&w_x_icfo.slice_axis(Axis(1), (s * iofc..s * (iofc + 1)).into()));
            r_iofc
                .slice_axis_mut(Axis(1), (s * icfo..s * (icfo + 1)).into())
                .assign(&w_h_icfo.slice_axis(Axis(1), (s * iofc..s * (iofc + 1)).into()));
            b_iofc
                .slice_axis_mut(Axis(0), (s * icfo..s * (icfo + 1)).into())
                .assign(&self.b_icfo.slice_axis(Axis(0), (s * iofc..s * (iofc + 1)).into()));
        }
        let w_iofc = w_iofc.t().into_shape_with_order(((1, 4 * s, s), Order::ColumnMajor))?.to_owned();
        let r_iofc = r_iofc.t().into_shape_with_order(((1, 4 * s, s), Order::ColumnMajor))?.to_owned();
        let b_iofc = b_iofc.into_shape_with_order(((1, 8 * s), Order::ColumnMajor))?;

        let x = model.add_source("x", self.x.datum_type().fact(self.x.shape()).into())?;
        let op = tract_onnx::ops::rec::common::CommonRec {
            optional_y_output: Some(0),
            optional_bias_input: Some(3),
            optional_initial_h_input: Some(4),
            optional_initial_c_input: Some(5),
            optional_sequence_lens_input: None,
            optional_p_input: None,
            optional_y_h_output: None,
            optional_y_c_output: None,
            body: Box::new(tract_onnx::ops::rec::lstm::LSTM {
                f: Box::new(tract_core::ops::nn::sigmoid()),
                g: Box::new(tract_core::ops::math::tanh()),
                h: Box::new(tract_core::ops::math::tanh()),
            }),
            batch_first: false,
        };
        let w = model.add_const("w", w_iofc)?;
        let r = model.add_const("r", r_iofc)?;
        let b = model.add_const("b", b_iofc)?;
        let h0 = model.add_const("h0", self.h0.clone().insert_axis(Axis(0)))?;
        let c0 = model.add_const("c0", self.c0.clone().insert_axis(Axis(0)))?;
        let lstm = model.wire_node("lstm", expand(op), &[x, w, r, b, h0, c0]).unwrap();
        model.set_output_outlets(&lstm).unwrap();
        model.analyse(false)?;
        model.into_typed()
    }

    pub fn tf_model(&self) -> TractResult<TypedModel> {
        let mut model = InferenceModel::default();

        let x = model.add_source("x", self.x.datum_type().fact(self.x.shape()).into())?;
        let memory_shape = tvec!(self.batch_size, self.cell_size);
        let h = model.wire_node(
            "h",
            tract_tensorflow::ops::vars::VariableV2::new(
                None,
                None,
                "h".into(),
                "h".into(),
                memory_shape.clone(),
                f32::datum_type(),
                None,
            ),
            &[],
        )?[0];
        let cs = model.wire_node(
            "cs",
            tract_tensorflow::ops::vars::VariableV2::new(
                None,
                None,
                "cs".into(),
                "cs".into(),
                memory_shape,
                f32::datum_type(),
                None,
            ),
            &[],
        )?[0];
        let seq_length = model.add_const("seq_length", tensor0(self.length as i64))?;
        let w = model.add_const("w", self.w_xh_icfo.clone())?;
        let wc1 = model.add_const("wc1", tensor1(&[0f32]))?;
        let wc2 = model.add_const("wc2", tensor1(&[0f32]))?;
        let wc3 = model.add_const("wc3", tensor1(&[0f32]))?;
        let b = model.add_const("b", self.b_icfo.clone())?;

        let lstm = model
            .wire_node(
                "lstm",
                expand(tract_tensorflow::ops::rec::block_lstm::BlockLSTM::new(
                    0.0,
                    -1.0,
                    f32::datum_type(),
                    false,
                )),
                &[seq_length, x, cs, h, w, wc1, wc2, wc3, b],
            )
            .unwrap();

        let last_h = model.wire_node(
            "last_h",
            expand(tract_hir::ops::array::Split::new(0, 2, Some(vec![self.length - 1, 1]))),
            &[lstm[6]],
        )?[1];
        let last_h_squeezed = model.wire_node(
            "last_h_squeezed",
            expand(tract_hir::ops::array::RmDims::new(vec![0])),
            &[last_h],
        )?[0];

        let last_cs = model.wire_node(
            "last_cs",
            expand(tract_hir::ops::array::Split::new(0, 2, Some(vec![self.length - 1, 1]))),
            &[lstm[1]],
        )?[1];
        let last_cs_squeezed = model.wire_node(
            "last_cs_squeezed",
            expand(tract_hir::ops::array::RmDims::new(vec![0])),
            &[last_cs],
        )?[0];

        let a_h = model.wire_node(
            "a_h",
            ::tract_tensorflow::ops::vars::Assign::new(Some("h".into())),
            &[h, last_h_squeezed],
        )?[0];

        let a_cs = model.wire_node(
            "a_cs",
            ::tract_tensorflow::ops::vars::Assign::new(Some("cs".into())),
            &[cs, last_cs_squeezed],
        )?[0];

        let _memo = model.wire_node("memo", ::tract_tensorflow::ops::Noop::new(), &[a_h, a_cs])?[0];

        let h0 = model.add_const("h0", self.h0.clone())?;
        let a_h0 = model.wire_node(
            "a_h0",
            ::tract_tensorflow::ops::vars::Assign::new(Some("h".into())),
            &[h, h0],
        )?[0];

        let cs0 = model.add_const("cs0", self.c0.clone()).unwrap();
        let a_cs0 = model.wire_node(
            "a_cs0",
            ::tract_tensorflow::ops::vars::Assign::new(Some("cs".into())),
            &[cs, cs0],
        )?[0];

        let init = model.wire_node("init", ::tract_tensorflow::ops::Noop::new(), &[a_h0, a_cs0])?;

        model.set_output_names(["lstm", "memo"])?;
        let extensions = tract_tensorflow::model::TfModelExtensions {
            control_inputs: vec![],
            initializing_nodes: vec![init[0].node],
        };
        let model = extensions.preproc(model)?;

        // println!("{:#?}", model);
        model.into_typed()
    }

    pub fn onnx_run(&self) -> TractResult<TValue> {
        let model = self.onnx_model()?;
        let plan = SimplePlan::new(model)?;
        let mut state = SimpleState::new(plan)?;
        let y = state.run(tvec!(self.x.clone()))?.remove(0).into_tensor().into_array::<f32>()?;
        let y = y.into_shape_with_order((self.length, self.batch_size, self.cell_size)).unwrap();
        Ok(y.into_tvalue())
    }

    pub fn tf_run(&self) -> TractResult<TValue> {
        let model = self.tf_model()?;
        let lstm_id = model.node_by_name("lstm")?.id;
        let memo_id = model.node_by_name("memo")?.id;
        let plan_run = SimplePlan::build(
            &model,
            &[OutletId::new(lstm_id, 6), OutletId::new(memo_id, 0)],
            &[],
            &PlanOptions::default(),
        )?;
        let mut state = SimpleState::new(plan_run)?;
        let y = state.run(tvec!(self.x.clone()))?.remove(0);
        Ok(y)
    }
}

fn strat() -> BoxedStrategy<LstmProblem> {
    (1usize..4, 1usize..4, 1usize..4)
        .prop_flat_map(|(length, batch_size, cell_size)| {
            (
                Just((length, batch_size, cell_size)),
                proptest::collection::vec(
                    (-3..3).prop_map(|a| a as f32),
                    length * batch_size * cell_size,
                ),
                proptest::collection::vec(
                    (-3..3).prop_map(|a| a as f32),
                    8 * cell_size * cell_size,
                ),
                proptest::collection::vec((-3..3).prop_map(|a| a as f32), 4 * cell_size),
                proptest::collection::vec((-3..3).prop_map(|a| a as f32), cell_size * batch_size),
                proptest::collection::vec((-3..3).prop_map(|a| a as f32), cell_size * batch_size),
            )
        })
        .prop_map(|((length, batch_size, cell_size), x, w_xh_icfo, b_icfo, h0, c0)| {
            let x =
                Array3::from_shape_vec((length, batch_size, cell_size), x).unwrap().into_tvalue();
            let w_xh_icfo =
                Array2::from_shape_vec((cell_size * 2, cell_size * 4), w_xh_icfo).unwrap();
            let b_icfo = Array1::from_shape_vec(cell_size * 4, b_icfo).unwrap();
            let h0 = Array2::from_shape_vec((batch_size, cell_size), h0).unwrap();
            let c0 = Array2::from_shape_vec((batch_size, cell_size), c0).unwrap();
            LstmProblem { length, batch_size, cell_size, x, w_xh_icfo, b_icfo, h0, c0 }
        })
        .boxed()
}

proptest::proptest! {
    #[test]
    fn test(pb in strat()) {
        let o = pb.onnx_run().unwrap();
        let t = pb.tf_run().unwrap();
        prop_assert!(o.close_enough(&t, true).is_ok(), "\nonnx:{:?}\n tf :{:?}\n", o, t);
    }
}

#[test]
fn test_x() {
    let pb = LstmProblem {
        length: 1,
        batch_size: 1,
        cell_size: 1,
        x: tensor3(&[[[-3f32]]]).into(),
        w_xh_icfo: arr2(&[[0.0f32, -1.0, 0.0, -6.0], [0.0, 0.0, 0.0, 0.0]]),
        b_icfo: arr1(&[0.0f32, 0.0, 0.0, 0.0]),
        h0: arr2(&[[0.0f32]]),
        c0: arr2(&[[0.0f32]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}

#[test]
fn test_seq() {
    let pb = LstmProblem {
        length: 2,
        batch_size: 1,
        cell_size: 1,
        x: tensor3(&[[[1f32]], [[2.0]]]).into(),
        w_xh_icfo: arr2(&[[0.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        b_icfo: arr1(&[0.0f32, 0.0, 0.0, 0.0]),
        h0: arr2(&[[0.0f32]]),
        c0: arr2(&[[0.0f32]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}

#[test]
fn test_c0() {
    let pb = LstmProblem {
        length: 1,
        batch_size: 1,
        cell_size: 1,
        x: tensor3(&[[[-0f32]]]).into(),
        w_xh_icfo: arr2(&[[0.0f32, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 0.0]]),
        b_icfo: arr1(&[0.0f32, 0.0, 0.0, 0.0]),
        h0: arr2(&[[0.0f32]]),
        c0: arr2(&[[1.0f32]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}

#[test]
fn test_b() {
    let pb = LstmProblem {
        length: 1,
        batch_size: 1,
        cell_size: 2,
        x: tensor3(&[[[0f32, 0.0]]]).into(),
        w_xh_icfo: Array2::<f32>::zeros((4, 8)),
        b_icfo: arr1(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        h0: arr2(&[[0.0f32, 0.0]]),
        c0: arr2(&[[0.0f32, 0.0]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}

#[test]
fn test_w() {
    let pb = LstmProblem {
        length: 2,
        batch_size: 1,
        cell_size: 2,
        x: tensor3(&[[[2.0f32, 1.0]], [[0.0, -3.0]]]).into(),
        w_xh_icfo: arr2(&[
            [0f32, -2.0, 0.0, 0.0, -2.0, -2.0, -3.0, -3.0],
            [0.0, -3.0, 2.0, 0.0, -2.0, 2.0, 1.0, -3.0],
            [0.0, 2.0, 1.0, 0.0, -3.0, -2.0, -3.0, -1.0],
            [0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        b_icfo: arr1(&[0.0f32, 0.0, -3.0, -1.0, -1.0, 0.0, 2.0, -2.0]),
        h0: arr2(&[[1.0f32, 0.0]]),
        c0: arr2(&[[-1.0f32, -2.0]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}
