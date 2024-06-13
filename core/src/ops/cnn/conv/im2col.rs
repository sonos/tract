use tract_linalg::frame::{MatMatMul, PackedFormat, PackingWriter};
use tract_linalg::mmm::{EagerPackedInput, MMMInputValue};

use crate::internal::*;
use ndarray::prelude::*;
use num_integer::Integer;

use crate::ops::cnn::pools::{ConcretePoolGeometry, PoolGeometry};
use crate::ops::cnn::{GeometryBound, PoolSpec, ResolveTo};
use crate::ops::nn::{BaseDataShape, DataFormat, DataShape};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Im2Col {
    pub pool_spec: PoolSpec,
    pub group: usize,
    geometry: GeometryBound<SymbolicGeometry, ConcreteGeometry>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SymbolicGeometry {
    group: usize,
    pool_spec: PoolSpec,
    pool_geometry: PoolGeometry,
    b_pack: PackedFormat,
    k: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConcreteGeometry {
    pool: ConcretePoolGeometry,
    pub n: usize,
    k: usize,
    pub b_pack: PackedFormat,
    pub ci_per_group: usize,
    patcher: Patcher,
    input_shape_with_n: DataShape,
    packed_shape: TVec<usize>, // always Batch,Group,Packed
}

impl GeometryBound<SymbolicGeometry, ConcreteGeometry> {
    pub fn b_pack(&self) -> &PackedFormat {
        match self {
            GeometryBound::Symbolic(s) => &s.b_pack,
            GeometryBound::Concrete(s) => &s.b_pack,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            GeometryBound::Symbolic(s) => s.k,
            GeometryBound::Concrete(s) => s.k,
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
        let packed_shape = Im2Col::packed_shape(&pool.input_shape, self.group)?;
        Ok(ConcreteGeometry {
            pool,
            n,
            k: self.k,
            ci_per_group,
            b_pack: self.b_pack.clone(),
            patcher,
            input_shape_with_n,
            packed_shape,
        })
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
        let b_pack = mmm.packings()[0]
            .1
            .downcast_ref::<PackedFormat>()
            .context("Im2Col expects regular packed format")?
            .clone();

        let pool_geometry = pool_spec.compute_geo(input_full_shape)?;
        let geometry: GeometryBound<_, _> =
            SymbolicGeometry { group, pool_spec: pool_spec.clone(), pool_geometry, b_pack, k }
                .into();
        let geometry = geometry.optimize_if(input_full_shape.as_concrete())?;
        Ok(Im2Col { pool_spec, group, geometry })
    }

    // packed shape is Batch,Group
    fn packed_shape<D: DimLike>(
        input_shape: &BaseDataShape<D, TVec<D>>,
        group: usize,
    ) -> TractResult<TVec<D>> {
        let mut output_shape: TVec<D> = tvec!();
        output_shape.push(input_shape.n().cloned().unwrap_or_else(|| 1.into()));
        output_shape.push(group.into());
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

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Im2Col {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let geometry = self.geometry.to_concrete(inputs[0].shape())?;
        unsafe {
            let mut input = inputs.remove(0).into_tensor();
            let pad_value: Option<&Tensor> = if inputs.len() > 0 { Some(&inputs[0]) } else { None };
            let mut output = Tensor::uninitialized::<Opaque>(&geometry.packed_shape)?;
            if !self.pool_spec.data_format.has_n() {
                input.insert_axis(0)?;
            }
            let mut output_view = output.to_array_view_mut::<Opaque>()?;
            let panel_bytes =
                geometry.b_pack.single_panel_len(geometry.k) * input.datum_type().size_of();

            // in the loop, we have normalized the input so that N is
            // always here, and output so that N and G are there.
            if !geometry.pool.output_shape.shape.iter().any(|d| *d == 0) {
                for i in 0..*geometry.input_shape_with_n.n().unwrap_or(&1) {
                    let input = input.view_at_prefix(&[i])?;
                    for g in 0..self.group {
                        let mut data = Tensor::uninitialized_aligned_dt(
                            input.datum_type(),
                            &[geometry.b_pack.len(geometry.k, geometry.n)],
                            geometry.b_pack.alignment(),
                        )?;
                        dispatch_copy_by_size!(Patcher::patch(input.datum_type())(
                            &geometry.patcher,
                            &geometry,
                            &input,
                            &mut data.view_mut(),
                            g,
                            pad_value
                        ))?;
                        let input: Box<dyn MMMInputValue> = Box::new(EagerPackedInput {
                            packed: data,
                            panel_bytes,
                            k: geometry.k,
                            mn: geometry.n,
                            r: geometry.b_pack.r,
                        });
                        output_view[[i, g]] = input.into();
                    }
                }
            }
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl TypedOp for Im2Col {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        Ok(tvec!(Opaque::fact(&[input_shape.n().cloned().unwrap_or(1.into()), self.group.into()])))
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum Patcher {
    Generic,
    Valid1d,
    Valid2d,
    Padded2d,
}

impl Patcher {
    fn patch<'p, T: Copy + Datum + num_traits::Zero>(
        &self,
        geo: &'p ConcreteGeometry,
        input: &TensorView,
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
    fn generic<'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &TensorView,
        pack: &'p mut TensorView,
        g: usize,
        pad_value: &Tensor,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *pad_value.to_scalar_unchecked();
            let mut mega_matrix = Tensor::uninitialized::<T>(&[geometry.k, geometry.n])?;
            let mut mega_matrix_view = mega_matrix.to_array_view_mut_unchecked::<T>();
            let ptr = input.as_ptr_unchecked::<T>();
            let ptr = ptr.add(geometry.input_shape_with_n.c_stride() * (g * geometry.ci_per_group));
            for (spatial, mut col) in ndarray::indices(&*geometry.pool.patch.output_shape)
                .into_iter()
                .zip(mega_matrix_view.axis_iter_mut(Axis(1)))
            {
                let mut col = col.iter_mut();
                for ci in 0..geometry.ci_per_group {
                    let ptr = ptr.add(geometry.input_shape_with_n.c_stride() * ci);
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
    fn valid_1d<'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &TensorView,
        pack: &'p mut TensorView,
        g: usize,
    ) -> TractResult<()> {
        unsafe {
            let x_stride = *geometry.input_shape_with_n.h_stride() as isize
                * geometry.pool.patch.spec.strides[0] as isize;
            let c_stride = *geometry.input_shape_with_n.c_stride() as isize;
            let pack = pack.as_slice_mut_unchecked::<T>();
            let mut writer =
                geometry.b_pack.write_with_k_outer(pack.as_mut_ptr(), geometry.k, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.add(g * geometry.ci_per_group * geometry.input_shape_with_n.c_stride());
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset);
                    for x in 0..*geometry.pool.patch.output_shape.get_unchecked(0) {
                        writer.write(*iptr.offset(x as isize * x_stride));
                    }
                }
            }
            Ok(())
        }
    }

    #[inline(never)]
    fn padded_2d<'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &TensorView,
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
            let mut writer =
                geometry.b_pack.write_with_k_outer(pack.as_mut_ptr(), geometry.k, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.add(g * geometry.ci_per_group * shape.c_stride());
            let output_width = *geometry.pool.patch.output_shape.get_unchecked(1);
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for kitem in 0..kernel_len {
                    let dy = *geometry.pool.patch.data_field.as_ptr().offset(kitem as isize * 2);
                    let dx =
                        *geometry.pool.patch.data_field.as_ptr().offset(1 + kitem as isize * 2);
                    let valid_x_start =
                        Integer::div_ceil(&-dx, &x_stride).max(0).min(output_width as _);
                    let valid_x_end =
                        Integer::div_ceil(&(input_width - dx), &x_stride).min(output_width as _);

                    let iptr = iptr.offset(
                        *geometry.pool.patch.standard_layout_data_field.get_unchecked(kitem),
                    );
                    for yo in 0..*geometry.pool.patch.output_shape.get_unchecked(0) {
                        let y = yo as isize * y_stride + dy;
                        let iptr = iptr.offset(yo as isize * y_stride_ptr);
                        if y >= 0 && y < input_heigth {
                            Self::padded_2d_invalid_x_loop(
                                valid_x_start as usize,
                                pad_value,
                                &mut writer,
                            );
                            Self::padded_2d_valid_x_loop(
                                valid_x_start,
                                valid_x_end,
                                x_stride_ptr,
                                iptr,
                                &mut writer,
                            );
                            Self::padded_2d_invalid_x_loop(
                                output_width - valid_x_end as usize,
                                pad_value,
                                &mut writer,
                            );
                        } else {
                            Self::padded_2d_invalid_x_loop(output_width, pad_value, &mut writer);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    unsafe fn padded_2d_invalid_x_loop<T: Copy + Datum>(
        count: usize,
        pad_value: T,
        writer: &mut tract_linalg::frame::mmm::pack::KOutWriter<T>,
    ) {
        for _ in 0..count {
            writer.write(pad_value);
        }
    }

    #[inline(never)]
    unsafe fn padded_2d_valid_x_loop<T: Copy + Datum>(
        x_min: isize,
        x_max: isize,
        x_stride_ptr: isize,
        iptr: *const T,
        writer: &mut tract_linalg::frame::mmm::pack::KOutWriter<T>,
    ) {
        for x in x_min..x_max {
            writer.write(*iptr.offset(x * x_stride_ptr));
        }
    }

    #[inline(never)]
    fn valid_2d<'p, T: Copy + Datum>(
        geometry: &'p ConcreteGeometry,
        input: &TensorView,
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
            let mut writer =
                geometry.b_pack.write_with_k_outer(pack.as_mut_ptr(), geometry.k, geometry.n);
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.add(g * geometry.ci_per_group * shape.c_stride());
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset);
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
