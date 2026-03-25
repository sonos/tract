use crate::axes::Axis;
use crate::internal::*;
use ndarray::*;
use tract_linalg::block_quant::{
    BlockQuantStorage, PackedBlockQuantFact, PackedBlockQuantFormat, block_quant_slice,
};
use tract_linalg::mmm::{MMMInputValue, PackedMatrixStorage};
use tract_linalg::pack::PackedFormat;

use super::ModePicker;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptMatMulPack {
    pub(crate) packers: Vec<PackedFormat>,
    pub(crate) mode_picker: ModePicker,
    pub(crate) k_axis: usize,
    pub(crate) mn_axis: usize,
}

impl Op for OptMatMulPack {
    fn name(&self) -> StaticName {
        "OptMatMulPack".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}. k axis: {}, mn axis: {}", self.packers, self.k_axis, self.mn_axis)])
    }

    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for OptMatMulPack {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        mut inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        self.do_eval(session, inputs.remove(0))
    }
}

impl TypedOp for OptMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        match self.mode_picker {
            ModePicker::Single => ensure!(self.packers.len() == 1),
            ModePicker::VecVsMat => ensure!(self.packers.len() == 2),
        }
        let k = inputs[0].shape[self.k_axis].clone();
        let mn = inputs[0].shape[self.mn_axis].clone();
        let opaque_fact = DynPackedOpaqueFact { k, mn, packers: self.packers.clone() };
        Ok(tvec!(
            inputs[0]
                .datum_type
                .fact(self.output_shape(&inputs[0].shape))
                .with_opaque_fact(opaque_fact)
        ))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes: Vec<Axis> = (0..inputs[0].rank())
            .filter(|&ix| ix != self.k_axis && ix != self.mn_axis)
            .enumerate()
            .zip('a'..)
            .map(|((o, i), repr)| Axis::new(repr, 1, 1).input(0, i).output(0, o))
            .collect();
        axes.push(Axis::new('K', 1, 1).input(0, self.k_axis));
        axes.push(Axis::new('M', 1, 1).input(0, self.mn_axis));
        axes.push(Axis::new('P', 1, 1).output(0, outputs[0].rank()));
        AxesMapping::new(1, 1, axes)
    }

    as_op!();
}

impl OptMatMulPack {
    fn do_eval(&self, _session: &TurnState, input: TValue) -> TractResult<TVec<TValue>> {
        unsafe {
            let mode = self.mode_picker.pick(input.shape()[self.mn_axis])?;
            let packer = &self.packers[mode];
            let output_shape: TVec<usize> = self.output_shape(input.shape());
            let stores = if output_shape.iter().all(|d| *d == 1) {
                let packed = packer.pack_tensor_view(&input.view(), self.k_axis, self.mn_axis)?;
                PackedMatrixStorage::new_batched(&output_shape, vec![packed])
                    .into_tensor(input.datum_type())
            } else {
                let mut bc_shape: TVec<usize> = input.shape().into();
                bc_shape[self.k_axis] = 1;
                bc_shape[self.mn_axis] = 1;

                let mut values: Vec<Box<dyn MMMInputValue>> =
                    Vec::with_capacity(output_shape.iter().product());
                for coord in indices(&*bc_shape) {
                    let offset = coord
                        .as_array_view()
                        .iter()
                        .zip(input.strides())
                        .map(|(x, s)| *x as isize * s)
                        .sum::<isize>()
                        * input.datum_type().size_of() as isize;
                    values.push(packer.pack_tensor_view(
                        &TensorView::from_bytes(&input, offset, input.shape(), input.strides()),
                        self.k_axis,
                        self.mn_axis,
                    )?);
                }
                PackedMatrixStorage::new_batched(&output_shape, values)
                    .into_tensor(input.datum_type())
            };
            Ok(tvec!(stores.into_tvalue()))
        }
    }

    pub fn output_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        let mut packed_shape: TVec<D> = input.into();
        packed_shape.remove(self.mn_axis.max(self.k_axis));
        packed_shape.remove(self.mn_axis.min(self.k_axis));
        packed_shape
    }
}

#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub struct DynPackedOpaqueFact {
    pub k: TDim,
    pub mn: TDim,
    pub packers: Vec<PackedFormat>,
}

impl OpaqueFact for DynPackedOpaqueFact {
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(self.k.clone() * &self.mn * self.packers[0].dt.size_of())
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct OptSimpleMatMulPack {
    pub(crate) packed_format: PackedBlockQuantFormat,
    pub(crate) k: usize,
    pub(crate) m: usize,
}

impl Op for OptSimpleMatMulPack {
    fn name(&self) -> StaticName {
        "OptSimpleMatMulPack".into()
    }
    op_as_typed_op!();
    impl_op_same_as!();
}

impl EvalOp for OptSimpleMatMulPack {
    fn is_stateless(&self) -> bool {
        true
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let bqs = input.try_storage_as::<BlockQuantStorage>()?;
        // Leading dims before the last 2 (M, K) are batch/group dims
        let num_groups: usize = input.shape()[..input.rank().saturating_sub(2)].iter().product();
        let m_per_group = input.shape()[input.rank() - 2];
        let k = *input.shape().last().unwrap();
        let values = (0..num_groups)
            .map(|g| {
                let slice = block_quant_slice(bqs.value(), bqs.format(), m_per_group, k, g);
                let iv: Box<dyn MMMInputValue> = Box::new(self.packed_format.pack(slice, k)?);
                Ok(iv)
            })
            .collect::<TractResult<Vec<_>>>()?;
        let leading_shape = &input.shape()[..input.rank().saturating_sub(2)];
        let output =
            PackedMatrixStorage::new_batched(leading_shape, values).into_tensor(input.datum_type());
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for OptSimpleMatMulPack {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input = inputs[0];
        // Input shape is [G, M, K] — output removes M and K, keeping leading dims
        let output_shape: TVec<TDim> = if input.rank() > 2 {
            input.shape[..input.rank() - 2].to_vec().into()
        } else {
            tvec!()
        };
        let fact =
            inputs[0].datum_type.fact(&*output_shape).with_opaque_fact(PackedBlockQuantFact {
                format: self.packed_format.clone(),
                shape: tvec!(self.m, self.k),
            });
        Ok(tvec!(fact))
    }

    as_op!();
}
