#![allow(dead_code)]

use proptest::prelude::*;

use tract_core::internal::*;
use tract_core::ndarray::*;

#[derive(Clone, Debug)]
pub struct LstmProblem {
    pub length: usize,
    pub batch_size: usize,
    pub cell_size: usize,
    pub x: Arc<Tensor>,
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
        let w_iofc = w_iofc.t().into_shape((1, 4 * s, s))?.to_owned();
        let r_iofc = r_iofc.t().into_shape((1, 4 * s, s))?.to_owned();
        let b_iofc = b_iofc.into_shape((1, 8 * s))?;

        let _x =
            model.add_source("x", TensorFact::dt_shape(self.x.datum_type(), self.x.shape()))?;
        let mut op = tract_onnx::ops::rec::lstm::LSTM::default();
        op.optional_y_output = Some(0);
        op.optional_bias_input = Some(3);
        op.optional_initial_h_input = Some(4);
        op.optional_initial_c_input = Some(5);
        let lstm = model.chain("lstm", op, tvec!(TensorFact::default())).unwrap();
        model.plug_const(InletId::new(lstm, 1), "w", w_iofc)?;
        model.plug_const(InletId::new(lstm, 2), "r", r_iofc)?;
        model.plug_const(InletId::new(lstm, 3), "b", b_iofc)?;
        model.plug_const(
            InletId::new(lstm, 4),
            "initial_h",
            self.h0.clone().insert_axis(Axis(0)),
        )?;
        model.plug_const(
            InletId::new(lstm, 5),
            "initial_c",
            self.c0.clone().insert_axis(Axis(0)),
        )?;
        model.set_output_outlets(&[OutletId::new(lstm, 0)])?;
        model.analyse(false)?;
        Ok(model.into_typed()?)
    }

    pub fn tf_model(&self) -> TractResult<TypedModel> {
        let mut model = InferenceModel::default();

        let x = model.add_source("x", TensorFact::dt_shape(self.x.datum_type(), self.x.shape()))?;
        let lstm = model
            .add_node(
                "lstm",
                tract_tensorflow::ops::rec::block_lstm::BlockLSTM::new(
                    0.0,
                    -1.0,
                    f32::datum_type(),
                    false,
                ),
                tvec!(TensorFact::default(); 7),
            )
            .unwrap();
        let memory_shape = tvec!(self.batch_size, self.cell_size);
        let memory_fact = TensorFact::dt_shape(f32::datum_type(), memory_shape.clone());
        let h = model.add_node(
            "h",
            tract_tensorflow::ops::vars::VariableV2::new(
                None,
                None,
                "h".into(),
                "h".into(),
                memory_shape.clone(),
                f32::datum_type(),
            ),
            tvec!(memory_fact.clone()),
        )?;
        let cs = model.add_node(
            "cs",
            tract_tensorflow::ops::vars::VariableV2::new(
                None,
                None,
                "cs".into(),
                "cs".into(),
                memory_shape.clone(),
                f32::datum_type(),
            ),
            tvec!(memory_fact),
        )?;
        model.plug_const(InletId::new(lstm, 0), "seq_length", tensor0(self.length as i64))?;
        model.add_edge(OutletId::new(x, 0), InletId::new(lstm, 1))?;
        model.add_edge(OutletId::new(cs, 0), InletId::new(lstm, 2))?;
        model.add_edge(OutletId::new(h, 0), InletId::new(lstm, 3))?;
        model.plug_const(InletId::new(lstm, 4), "w", self.w_xh_icfo.clone())?;
        model.plug_const(InletId::new(lstm, 5), "wc1", tensor1(&[0f32]))?;
        model.plug_const(InletId::new(lstm, 6), "wc2", tensor1(&[0f32]))?;
        model.plug_const(InletId::new(lstm, 7), "wc3", tensor1(&[0f32]))?;
        model.plug_const(InletId::new(lstm, 8), "b", self.b_icfo.clone())?;

        let last_h = model.add_node(
            "last_h",
            ::tract_core::ops::array::Split::new(0, 2, Some(vec![self.length - 1, 1])),
            tvec!(TensorFact::default(), TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(lstm, 6), InletId::new(last_h, 0))?;
        let last_h_squeezed = model.add_node(
            "last_h_squeezed",
            ::tract_core::ops::array::RmDims::new(vec![0]),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(last_h, 1), InletId::new(last_h_squeezed, 0))?;

        let last_cs = model.add_node(
            "last_cs",
            ::tract_core::ops::array::Split::new(0, 2, Some(vec![self.length - 1, 1])),
            tvec!(TensorFact::default(), TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(lstm, 1), InletId::new(last_cs, 0))?;
        let last_cs_squeezed = model.add_node(
            "last_cs_squeezed",
            ::tract_core::ops::array::RmDims::new(vec![0]),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(last_cs, 1), InletId::new(last_cs_squeezed, 0))?;

        let a_h = model.add_node(
            "a_h",
            ::tract_tensorflow::ops::vars::Assign::new(Some("h".into())),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(h, 0), InletId::new(a_h, 0))?;
        model.add_edge(OutletId::new(last_h_squeezed, 0), InletId::new(a_h, 1))?;

        let a_cs = model.add_node(
            "a_cs",
            ::tract_tensorflow::ops::vars::Assign::new(Some("cs".into())),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(cs, 0), InletId::new(a_cs, 0))?;
        model.add_edge(OutletId::new(last_cs_squeezed, 0), InletId::new(a_cs, 1))?;

        let memo = model.add_node(
            "memo",
            ::tract_tensorflow::ops::Noop::new(),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(a_h, 0), InletId::new(memo, 0))?;
        model.add_edge(OutletId::new(a_cs, 0), InletId::new(memo, 1))?;

        let a_h0 = model.add_node(
            "a_h0",
            ::tract_tensorflow::ops::vars::Assign::new(Some("h".into())),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(h, 0), InletId::new(a_h0, 0))?;
        model.plug_const(InletId::new(a_h0, 1), "h0", self.h0.clone())?;

        let a_cs0 = model.add_node(
            "a_cs0",
            ::tract_tensorflow::ops::vars::Assign::new(Some("cs".into())),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(cs, 0), InletId::new(a_cs0, 0))?;
        model.plug_const(InletId::new(a_cs0, 1), "cs0", self.c0.clone())?;

        let init = model.add_node(
            "init",
            ::tract_tensorflow::ops::Noop::new(),
            tvec!(TensorFact::default()),
        )?;
        model.add_edge(OutletId::new(a_h0, 0), InletId::new(init, 0))?;
        model.add_edge(OutletId::new(a_cs0, 0), InletId::new(init, 1))?;

        model.set_output_names(&["lstm", "init", "memo"])?;

        model.analyse(false)?;
        // println!("{:#?}", model);
        Ok(model.into_typed()?)
    }

    pub fn onnx_run(&self) -> TractResult<Arc<Tensor>> {
        let model = self.onnx_model()?;
        let plan = SimplePlan::new(model)?;
        let mut state = SimpleState::new(plan)?;
        let y = state
            .run(tvec!(self.x.clone().into_tensor()))?
            .remove(0)
            .into_tensor()
            .into_array::<f32>()?;
        let y = y.into_shape((self.length, self.batch_size, self.cell_size)).unwrap();
        Ok(y.into_arc_tensor())
    }

    pub fn tf_run(&self) -> TractResult<Arc<Tensor>> {
        let model = self.tf_model()?;
        let init_id = model.node_by_name("init")?.id;
        let lstm_id = model.node_by_name("lstm")?.id;
        let memo_id = model.node_by_name("memo")?.id;
        let plan_init = SimplePlan::new_for_output(&model, OutletId::new(init_id, 0))?;
        let plan_run = SimplePlan::new_for_outputs(
            &model,
            &[OutletId::new(lstm_id, 6), OutletId::new(memo_id, 0)],
        )?;
        let mut state = SimpleState::new_multiplan(vec![plan_init, plan_run])?;
        state.run_plan(tvec!(), 0)?;
        let y = state.run_plan(tvec!(self.x.clone().into_tensor()), 1)?.remove(0);
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
            let x = Array3::from_shape_vec((length, batch_size, cell_size), x)
                .unwrap()
                .into_arc_tensor();
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
        x: rctensor3(&[[[-3f32]]]),
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
        x: rctensor3(&[[[1f32]], [[2.0]]]),
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
        x: rctensor3(&[[[-0f32]]]),
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
        x: rctensor3(&[[[0f32, 0.0]]]),
        w_xh_icfo: Array2::<f32>::zeros((4, 8)).into(),
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
        x: rctensor3(&[[[2.0f32, 1.0]], [[0.0, -3.0]]]),
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

/*
#[test]
fn test_loops() {
    let pb = LstmProblem {
        loops: 2,
        chunk_length: 1,
        batch_size: 1,
        cell_size: 1,
        x: vec![rctensor3(&[[[0.0f32]]]), rctensor3(&[[[0.0f32]]])],
        w_xh_icfo: Array2::<f32>::zeros((2, 4)).into(),
        b_icfo: arr1(&[0.0f32, 0.0, 0.0, 0.0]),
        h0: arr2(&[[0.0f32]]),
        c0: arr2(&[[1.0f32]]),
    };
    let o = pb.onnx_run().unwrap();
    let t = pb.tf_run().unwrap();
    assert_eq!(o, t)
}
*/
