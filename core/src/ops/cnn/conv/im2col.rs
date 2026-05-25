use tract_linalg::mmm::{
    EagerPackedInput, MMMInputFormat, MMMInputValue, MatMatMul, PackedExoticFact,
    PackedMatrixStorage,
};
use tract_linalg::pack::{PackedFormat, PackedI8K4, PackingWriter};

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
    // The kernel's activation packing: PackedFormat (K-major) or PackedI8K4 (K=4-inner).
    out_format: Box<dyn MMMInputFormat>,
    k: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConcreteGeometry {
    pool: ConcretePoolGeometry,
    pub n: usize,
    k: usize,
    pub out_format: Box<dyn MMMInputFormat>,
    pub ci_per_group: usize,
    patcher: Patcher,
    input_shape_with_n: DataShape,
    packed_shape: TVec<usize>, // always Batch,Group
}

impl GeometryBound<SymbolicGeometry, ConcreteGeometry> {
    pub fn out_format(&self) -> &dyn MMMInputFormat {
        match self {
            GeometryBound::Symbolic(s) => &*s.out_format,
            GeometryBound::Concrete(s) => &*s.out_format,
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
            out_format: self.out_format.clone(),
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
        packing: usize,
    ) -> TractResult<Im2Col> {
        let out_format = dyn_clone::clone_box(&*mmm.packings()[packing].1);
        let pool_geometry = pool_spec.compute_geo(input_full_shape)?;
        let geometry: GeometryBound<_, _> =
            SymbolicGeometry { group, pool_spec: pool_spec.clone(), pool_geometry, out_format, k }
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
    fn name(&self) -> StaticName {
        "Im2col".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("groups:{}", self.group)])
    }

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
            if !self.pool_spec.data_format.has_n() {
                input.insert_axis(0)?;
            }
            let dt = input.datum_type();
            let r = geometry.out_format.r();
            // Buffer geometry. zero_init for PackedI8K4: the K=4-inner writer skips
            // the K-padding lanes (k..k_aligned), which SMOPA accumulates — they must
            // be 0. PackedFormat has no K padding; its mn-padding maps to discarded
            // output rows, so uninitialized is fine (matches prior behaviour).
            let (single_panel_len, buf_align, zero_init) =
                if let Some(pf) = geometry.out_format.downcast_ref::<PackedFormat>() {
                    (pf.single_panel_len(geometry.k), pf.alignment(), false)
                } else if let Some(p4) = geometry.out_format.downcast_ref::<PackedI8K4>() {
                    (p4.single_panel_len(geometry.k), p4.alignment(), true)
                } else {
                    bail!("Im2Col: unsupported packing format {:?}", geometry.out_format)
                };
            let panel_bytes = single_panel_len * dt.size_of();

            let n_batches = *geometry.input_shape_with_n.n().unwrap_or(&1);
            let n_groups = self.group;
            let mut values: Vec<Box<dyn MMMInputValue>> = Vec::with_capacity(n_batches * n_groups);

            for i in 0..n_batches {
                let input = input.view_at_prefix(&[i])?;
                for g in 0..n_groups {
                    let n =
                        if geometry.pool.output_shape.shape.contains(&0) { 0 } else { geometry.n };
                    let mut data = Tensor::uninitialized_aligned_dt(
                        dt,
                        &[n.divceil(r) * single_panel_len],
                        buf_align,
                    )?;
                    if zero_init {
                        data.as_bytes_mut().fill(0);
                    }
                    if n > 0 {
                        dispatch_copy_by_size!(Patcher::patch(dt)(
                            &geometry.patcher,
                            &geometry,
                            &input,
                            &mut data.view_mut(),
                            g,
                            pad_value
                        ))?;
                    }
                    values.push(Box::new(EagerPackedInput {
                        fact: PackedExoticFact {
                            format: geometry.out_format.clone(),
                            k: geometry.k,
                            mn: n.to_dim(),
                        },
                        packed: data.into_blob()?.into(),
                        panel_bytes: if n > 0 { panel_bytes } else { 0 },
                        mn: n,
                    }));
                }
            }

            let output = PackedMatrixStorage::new_batched(&geometry.packed_shape, values)
                .into_tensor(input.datum_type());
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl TypedOp for Im2Col {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let input_shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        let output_shape = self.pool_spec.output_shape(&inputs[0].shape)?;
        let mn = output_shape.hw_dims().iter().product::<TDim>();
        let pof = PackedExoticFact {
            format: dyn_clone::clone_box(self.geometry.out_format()),
            k: self.geometry.k(),
            mn,
        };
        Ok(tvec!(
            inputs[0]
                .datum_type
                .fact(&[input_shape.n().cloned().unwrap_or(1.into()), self.group.into()])
                .with_exotic_fact(pof)
        ))
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
        // Pick the packing writer for the kernel's output format, then run the
        // (writer-generic) patcher. PackedFormat keeps the K-major fast path;
        // PackedI8K4 writes the SMOPA K=4-inner layout in the same single pass.
        let ptr = unsafe { pack.as_slice_mut_unchecked::<T>().as_mut_ptr() };
        if let Some(pf) = geo.out_format.downcast_ref::<PackedFormat>() {
            let mut w = pf.write_with_k_outer(ptr, geo.k, geo.n);
            self.run::<T, _>(geo, input, g, pad_value, &mut w)
        } else if let Some(p4) = geo.out_format.downcast_ref::<PackedI8K4>() {
            let mut w = p4.write_with_k_outer(ptr, geo.k, geo.n);
            self.run::<T, _>(geo, input, g, pad_value, &mut w)
        } else {
            bail!("Im2Col: unsupported packing format {:?}", geo.out_format)
        }
    }

    fn run<T: Copy + Datum + num_traits::Zero, W: PackingWriter<T>>(
        &self,
        geo: &ConcreteGeometry,
        input: &TensorView,
        g: usize,
        pad_value: Option<&Tensor>,
        writer: &mut W,
    ) -> TractResult<()> {
        match self {
            Patcher::Valid1d => Self::valid_1d::<T, W>(geo, input, g, writer),
            Patcher::Valid2d => Self::valid_2d::<T, W>(geo, input, g, writer),
            Patcher::Padded2d => Self::padded_2d::<T, W>(
                geo,
                input,
                g,
                pad_value.unwrap_or(&Tensor::zero_scalar::<T>()?),
                writer,
            ),
            _ => Self::generic::<T, W>(
                geo,
                input,
                g,
                pad_value.unwrap_or(&Tensor::zero_scalar::<T>()?),
                writer,
            ),
        }
    }

    #[inline(never)]
    fn generic<T: Copy + Datum, W: PackingWriter<T>>(
        geometry: &ConcreteGeometry,
        input: &TensorView,
        g: usize,
        pad_value: &Tensor,
        writer: &mut W,
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
            // mega_matrix is [k, n] (k-major); feed K-outer to the writer, which
            // lays out the kernel's packing (K-major for PackedFormat, K=4-inner
            // for PackedI8K4) — byte-identical to PackedFormat::pack for the former.
            let mv = mega_matrix.as_slice_unchecked::<T>();
            for kk in 0..geometry.k {
                writer.write_slice(&mv[kk * geometry.n..(kk + 1) * geometry.n]);
            }
            Ok(())
        }
    }

    #[inline(never)]
    fn valid_1d<T: Copy + Datum, W: PackingWriter<T>>(
        geometry: &ConcreteGeometry,
        input: &TensorView,
        g: usize,
        writer: &mut W,
    ) -> TractResult<()> {
        unsafe {
            let x_stride = *geometry.input_shape_with_n.h_stride() as isize
                * geometry.pool.patch.spec.strides[0] as isize;
            let c_stride = *geometry.input_shape_with_n.c_stride() as isize;
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.add(g * geometry.ci_per_group * geometry.input_shape_with_n.c_stride());
            let output_x = *geometry.pool.patch.output_shape.get_unchecked(0);
            // Fast path: stride-1 contiguous read along x. Replaces the
            // per-element pointer-arithmetic loop with a single write_slice
            // (memcpy when the slice fits in the current panel).
            // Byte-identical to the slow path (write_slice's contract).
            let contiguous_x = x_stride == 1;
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset);
                    if contiguous_x {
                        let row = std::slice::from_raw_parts(iptr, output_x);
                        writer.write_slice(row);
                    } else {
                        // Hoist multiplication out of inner loop.
                        let mut iptr_x = iptr;
                        for _ in 0..output_x {
                            writer.write(*iptr_x);
                            iptr_x = iptr_x.offset(x_stride);
                        }
                    }
                }
            }
            Ok(())
        }
    }

    #[inline(never)]
    fn padded_2d<T: Copy + Datum, W: PackingWriter<T>>(
        geometry: &ConcreteGeometry,
        input: &TensorView,
        g: usize,
        pad_value: &Tensor,
        writer: &mut W,
    ) -> TractResult<()> {
        unsafe {
            let pad_value = *pad_value.to_scalar_unchecked();
            let y_stride = geometry.pool.patch.spec.strides[0] as isize;
            let x_stride = geometry.pool.patch.spec.strides[1] as isize;
            let shape = &geometry.input_shape_with_n;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let input_heigth = shape.hw_dims()[0] as isize;
            let input_width = shape.hw_dims()[1] as isize;
            let kernel_len = geometry.pool.patch.standard_layout_data_field.len();
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
                    let valid_x_end = Integer::div_ceil(&(input_width - dx), &x_stride)
                        .max(0)
                        .min(output_width as _);

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
                                &mut *writer,
                            );
                            Self::padded_2d_valid_x_loop(
                                valid_x_start,
                                valid_x_end,
                                x_stride_ptr,
                                iptr,
                                &mut *writer,
                            );
                            Self::padded_2d_invalid_x_loop(
                                output_width - valid_x_end as usize,
                                pad_value,
                                &mut *writer,
                            );
                        } else {
                            Self::padded_2d_invalid_x_loop(output_width, pad_value, &mut *writer);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    unsafe fn padded_2d_invalid_x_loop<T: Copy + Datum, W: PackingWriter<T>>(
        count: usize,
        pad_value: T,
        writer: &mut W,
    ) {
        for _ in 0..count {
            writer.write(pad_value);
        }
    }

    #[inline(never)]
    unsafe fn padded_2d_valid_x_loop<T: Copy + Datum, W: PackingWriter<T>>(
        x_min: isize,
        x_max: isize,
        x_stride_ptr: isize,
        iptr: *const T,
        writer: &mut W,
    ) {
        // Fast path: x_stride_ptr == 1 means consecutive x values are at
        // consecutive memory addresses, so the inner loop is a contiguous
        // slice write — byte-identical to the per-element loop.
        if x_stride_ptr == 1 && x_max > x_min {
            unsafe {
                let row = std::slice::from_raw_parts(iptr.offset(x_min), (x_max - x_min) as usize);
                writer.write_slice(row);
            }
        } else {
            for x in x_min..x_max {
                writer.write(unsafe { *iptr.offset(x * x_stride_ptr) });
            }
        }
    }

    #[inline(never)]
    fn valid_2d<T: Copy + Datum, W: PackingWriter<T>>(
        geometry: &ConcreteGeometry,
        input: &TensorView,
        g: usize,
        writer: &mut W,
    ) -> TractResult<()> {
        unsafe {
            let shape = &geometry.input_shape_with_n;
            let y_stride = geometry.pool.patch.spec.strides[0] as isize;
            let x_stride = geometry.pool.patch.spec.strides[1] as isize;
            let y_stride_ptr = y_stride * *shape.h_stride() as isize;
            let x_stride_ptr = x_stride * *shape.w_stride() as isize;
            let c_stride_ptr = *shape.c_stride() as isize;
            let iptr = input.as_ptr_unchecked::<T>();
            let iptr = iptr.add(g * geometry.ci_per_group * shape.c_stride());
            let output_y = *geometry.pool.patch.output_shape.get_unchecked(0);
            let output_x = *geometry.pool.patch.output_shape.get_unchecked(1);
            // Fast path: stride-1 contiguous reads along x within each y-row.
            // Each y-row becomes a single write_slice (memcpy when the slice
            // fits in the current panel). Byte-identical to the slow path.
            let contiguous_x = x_stride_ptr == 1;
            for ci in 0..geometry.ci_per_group {
                let iptr = iptr.offset(ci as isize * c_stride_ptr);
                for koffset in &geometry.pool.patch.standard_layout_data_field {
                    let iptr = iptr.offset(*koffset);
                    let mut iptr_y = iptr;
                    for _ in 0..output_y {
                        if contiguous_x {
                            let row = std::slice::from_raw_parts(iptr_y, output_x);
                            writer.write_slice(row);
                        } else {
                            // Hoist x multiplication out of inner loop.
                            let mut iptr_x = iptr_y;
                            for _ in 0..output_x {
                                writer.write(*iptr_x);
                                iptr_x = iptr_x.offset(x_stride_ptr);
                            }
                        }
                        iptr_y = iptr_y.offset(y_stride_ptr);
                    }
                }
            }
            Ok(())
        }
    }
}
