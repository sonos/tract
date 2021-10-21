use tract_linalg::frame::{MatMatMul, Packer};

use crate::internal::*;
use ndarray::prelude::*;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry};
use crate::ops::cnn::{GeometryBound, PoolSpec, ResolveTo};
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape};

#[derive(Debug, Clone, PartialEq, Educe)]
#[educe(Hash)]
pub struct Im2Col {
    pub pool_spec: PoolSpec,
    pub group: usize,
    geometry: GeometryBound<SymbolicGeometry, ConcreteGeometry>,
}

#[derive(Debug, Clone, Hash, PartialEq)]
struct SymbolicGeometry {
    group: usize,
    pool_spec: PoolSpec,
    pool_geometry: PoolGeometry,
    b_pack: Packer,
}

#[derive(Debug, Clone, Hash, PartialEq)]
struct ConcreteGeometry {
    pool: ConcretePoolGeometry,
    pub n: usize,
    pub b_pack: Packer,
    pub ci_per_group: usize,
    patcher: Patcher,
    input_shape_with_n: DataShape,
    packing_shape: TVec<usize>, // always has n and g
    packed_shape: TVec<usize>,
}

impl GeometryBound<SymbolicGeometry, ConcreteGeometry> {
    pub fn b_pack(&self) -> &Packer {
        match self {
            GeometryBound::Symbolic(s) => &s.b_pack,
            GeometryBound::Concrete(s) => &s.b_pack,
        }
    }
}

impl ResolveTo<ConcreteGeometry> for SymbolicGeometry {
    type Param = [usize];
    fn resolve(&self, input_full_shape: &[usize]) -> TractResult<ConcreteGeometry> {
        let pool = self.pool_geometry.to_concrete(input_full_shape)?.into_owned();
        let patcher = if !pool.patch.padded && pool.patch.rank() == 2 {
            Patcher::Valid2d
        } else if pool.patch.rank() == 2 {
            Patcher::Padded2d
        } else if !pool.patch.padded && pool.patch.rank() == 1 {
            Patcher::Valid1d
        } else {
            Patcher::Generic
        };
        let ci_per_group = pool.input_shape.c_dim() / self.group;
        let n = pool.output_shape.hw_dims().iter().product();
        let input_shape_with_n = match self.pool_spec.data_format {
            DataFormat::HWC => DataFormat::NHWC.from_n_c_hw(
                1,
                *pool.input_shape.c(),
                pool.input_shape.hw_dims(),
            )?,
            DataFormat::CHW => DataFormat::NCHW.from_n_c_hw(
                1,
                *pool.input_shape.c(),
                pool.input_shape.hw_dims(),
            )?,
            _ => pool.input_shape.clone(),
        };
        let packed_shape =
            Im2Col::packed_shape(&pool.input_shape, &pool.output_shape, self.group, &self.b_pack)?;
        let mut packing_shape = packed_shape.clone();
        if !pool.input_shape.fmt.has_n() {
            packing_shape.insert(0, 1);
        }
        if self.group == 1 {
            packing_shape.insert(1, 1);
        }
        Ok(ConcreteGeometry {
            pool,
            n,
            ci_per_group,
            b_pack: self.b_pack.clone(),
            patcher,
            input_shape_with_n,
            packed_shape,
            packing_shape,
        })
    }
}

impl DynHash for Im2Col {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        dyn_hash(self, state)
    }
}

impl Im2Col {
    pub fn new(
        pool_spec: PoolSpec,
        group: usize,
        k: usize,
        input_full_shape: &ShapeFact,
        mmm: Box<dyn MatMatMul>,
    ) -> TractResult<Im2Col> {
        let b_pack = mmm.b_pack(k);
        let pool_geometry = pool_spec.compute_geo(input_full_shape)?;
        let geometry: GeometryBound<_, _> =
            SymbolicGeometry { group, pool_spec: pool_spec.clone(), pool_geometry, b_pack }.into();
        let geometry = geometry.optimize_if(input_full_shape.as_concrete())?;
        Ok(Im2Col { pool_spec, group, geometry })
    }

    fn packed_shape<D: DimLike>(
        input_shape: &BaseDataShape<D, TVec<D>>,
        conv_output_shape: &BaseDataShape<D, TVec<D>>,
        group: usize,
        b_pack: &Packer,
    ) -> TractResult<TVec<D>> {
        let mut output_shape: TVec<D> = tvec!();
        if let Some(n) = input_shape.n() {
            output_shape.push(n.clone());
        }
        if group != 1 {
            output_shape.push(group.into());
        }
        let n: D = conv_output_shape.hw_dims().iter().cloned().product();
        output_shape.push(b_pack.len(n).into());
        Ok(output_shape)
    }
}

impl Op for Im2Col {
    fn name(&self) -> Cow<str> {
        "Im2col".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("groups:{}", self.group)])
    }

    op_core_lir!();
    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Im2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let geometry = self.geometry.to_concrete(&inputs[0].shape())?;
        unsafe {
            let mut input = inputs.remove(0).into_tensor();
            let pad_value = if inputs.len() > 0 { Some(inputs.remove(0)) } else { None };
            let mut output = Tensor::uninitialized_aligned_dt(
                input.datum_type(),
                &geometry.packing_shape,
                geometry.b_pack.alignment(),
            )?;
            if !self.pool_spec.data_format.has_n() {
                input.insert_axis(0)?;
            }
            // in the loop, we have normalized the input so that N is
            // always here, and output so that N and G are there.
            if !geometry.pool.output_shape.shape.iter().any(|d| *d == 0) {
                for i in 0..*geometry.input_shape_with_n.n().unwrap_or(&1) {
                    let input = input.view_at_prefix(&[i])?;
                    for g in 0..self.group {
                        let full_prefix = [i, g];
                        let actual_prefix = &full_prefix[..=(self.group > 1) as usize];
                        let mut packed = output.view_at_prefix_mut(actual_prefix)?;
                        dispatch_copy_by_size!(Patcher::patch(input.datum_type())(
                            &geometry.patcher,
                            &geometry,
                            &input,
                            &mut packed,
                            g,
                            pad_value.as_deref()
                        ))?
                    }
                }
            }
            output.set_shape_unchecked(&geometry.packed_shape);
            Ok(tvec!(output.into()))
        }
    }
}

impl TypedOp for Im2Col {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        let output_shape = self.pool_spec.output_shape(&inputs[0].shape)?;
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            Self::packed_shape(&input_shape, &output_shape, self.group, self.geometry.b_pack())?
        )))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if node.inputs.len() == 2
            && model.outlet_fact(node.inputs[1])?.konst.as_ref().and_then(|t| t.as_uniform())
                == Some(Tensor::zero_scalar_dt(input_fact.datum_type)?)
        {
            Ok(Some(
                TypedModelPatch::replace_single_op(model, node, &node.inputs[0..1], self.clone())?
                    .with_context("b0 is zero"),
            ))
        } else {
            Ok(None)
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq)]
enum Patcher {
    Generic,
    Valid1d,
    Valid2d,
    Padded2d,
}

impl Patcher {
    fn patch<'i, 'p, T: Copy + Datum + num_traits::Zero>(
        &self,
        geo: &'p ConcreteGeometry,
        input: &'i TensorView,
        pack: &'p mut TensorView,
        g: usize,
        pad_value: Option<&Tensor>,
    ) -> TractResult<()> {
        match self {
            Patcher::Valid1d => Self::valid_1d::<T>(geo, input, pack, g),
            Patcher::Valid2d => Self::valid_2d::<T>(geo, input, pack, g),
            Patcher::Padded2d => Self::padded_2d::<T>(
                geo,
                input,
                pack,
                g,
                pad_value.unwrap_or(&Tensor::zero_scalar::<T>()?),
            ),
            _ => Self::generic::<T>(
                geo,
                input,
                pack,
                g,
                pad_value.unwrap_or(&Tensor::zero_scalar::<T>()?),
            ),
        }
    }

    #[inline(never)]
    fn generic<'i, 'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &'i TensorView,
        pack: &'p mut TensorView,
        g: usize,
        pad_value: &Tensor,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *pad_value.to_scalar_unchecked();
            let mut mega_matrix = Tensor::uninitialized::<T>(&[geometry.b_pack.k(), geometry.n])?;
            let mut mega_matrix_view = mega_matrix.to_array_view_mut_unchecked::<T>();
            let ptr = input.as_ptr_unchecked::<T>();
            let ptr = ptr.offset(
                (geometry.input_shape_with_n.c_stride() * (g * geometry.ci_per_group)) as isize,
            );
            for (spatial, mut col) in ndarray::indices(&*geometry.pool.patch.output_shape)
                .into_iter()
                .zip(mega_matrix_view.axis_iter_mut(Axis(1)))
            {
                let mut col = col.iter_mut();
                for ci in 0..geometry.ci_per_group {
                    let ptr = ptr.offset((geometry.input_shape_with_n.c_stride() * ci) as isize);
                    for v in geometry.pool.patch.at(spatial.slice()) {
                        *col.next().expect("geometry error in conv") =
                            v.map(|o| *ptr.offset(o)).unwrap_or(pad_value);
                    }
                }
            }
            geometry.b_pack.pack(pack, mega_matrix.view(), 0, 1);
            Ok(())
        }
    }

    #[inline(never)]
    fn valid_1d<'i, 'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &'i TensorView,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let x_stride = *geometry.input_shape_with_n.h_stride() as isize
                * geometry.pool.patch.spec.strides[0] as isize;
            let c_stride = *geometry.input_shape_with_n.c_stride() as isize;
            let pack = pack.as_slice_mut_unchecked::<T>();
            let mut writer = geometry.b_pack.write_with_k_outer(pack, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset(
                (g * geometry.ci_per_group * geometry.input_shape_with_n.c_stride()) as isize,
            );
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for x in 0..*geometry.pool.patch.output_shape.get_unchecked(0) {
                        writer.write(*iptr.offset(x as isize * x_stride));
                    }
                }
            }
            Ok(())
        }
    }

    #[inline(never)]
    fn padded_2d<'i, 'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &'i TensorView,
        pack: &'p mut TensorView,
        g: usize,
        pad_value: &Tensor,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *pad_value.to_scalar_unchecked();
            let pack = pack.as_slice_mut_unchecked::<T>();
            let y_stride = geometry.pool.patch.spec.strides[0] as isize;
            let x_stride = geometry.pool.patch.spec.strides[1] as isize;
            let shape = &geometry.input_shape_with_n;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let input_heigth = shape.hw_dims()[0] as isize;
            let input_width = shape.hw_dims()[1] as isize;
            let kernel_len = geometry.pool.patch.standard_layout_data_field.len();
            let mut writer = geometry.b_pack.write_with_k_outer(pack, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset((g * geometry.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for kitem in 0..kernel_len {
                    let dy = *geometry.pool.patch.data_field.as_ptr().offset(kitem as isize * 2);
                    let dx =
                        *geometry.pool.patch.data_field.as_ptr().offset(1 + kitem as isize * 2);
                    let iptr = iptr.offset(
                        *geometry.pool.patch.standard_layout_data_field.get_unchecked(kitem),
                    );
                    for yo in 0..*geometry.pool.patch.output_shape.get_unchecked(0) {
                        let y = yo as isize * y_stride + dy;
                        let iptr = iptr.offset(yo as isize * y_stride_ptr);
                        if y >= 0 && y < input_heigth {
                            for xo in 0..*geometry.pool.patch.output_shape.get_unchecked(1) {
                                let x = xo as isize * x_stride + dx;
                                if x >= 0 && x < input_width {
                                    writer.write(*iptr.offset(xo as isize * x_stride_ptr));
                                } else {
                                    writer.write(pad_value);
                                }
                            }
                        } else {
                            for _x in 0..*geometry.pool.patch.output_shape.get_unchecked(1) {
                                writer.write(pad_value);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn valid_2d<'i, 'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &'i TensorView,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let pack = pack.as_slice_mut_unchecked::<T>();
            let shape = &geometry.input_shape_with_n;
            let y_stride = geometry.pool.patch.spec.strides[0] as isize;
            let x_stride = geometry.pool.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let mut writer = geometry.b_pack.write_with_k_outer(pack, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.offset((g * geometry.ci_per_group * shape.c_stride()) as isize);
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset as isize);
                    for y in 0..*geometry.pool.patch.output_shape.get_unchecked(0) {
                        let iptr = iptr.offset(y as isize * y_stride_ptr);
                        for x in 0..*geometry.pool.patch.output_shape.get_unchecked(1) {
                            writer.write(*iptr.offset(x as isize * x_stride_ptr));
                        }
                    }
                }
            }
            Ok(())
        }
    }
}
