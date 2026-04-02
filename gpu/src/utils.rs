use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage, Q4_0};

use crate::fact::*;
use crate::tensor::DeviceTensor;

pub fn facts_to_device_facts(
    facts: &[&TypedFact],
    resolve_facts: impl Fn(&[&TypedFact]) -> TractResult<TVec<TypedFact>>,
) -> TractResult<TVec<TypedFact>> {
    if facts.iter().all(|it| it.as_device_fact().is_some()) {
        let device_facts = facts
            .iter()
            .map(|it| it.to_device_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        let output_facts = (resolve_facts)(device_facts.as_slice())?;
        Ok(output_facts
            .into_iter()
            .map(|it| Ok(DeviceFact::new(DeviceTensorOrigin::FromDevice, it)?.into_exotic_fact()))
            .collect::<TractResult<_>>()?)
    } else if facts.iter().all(|it| it.as_device_fact().is_none()) {
        (resolve_facts)(facts)
    } else {
        bail!("Inconsistent facts: mix of device and host facts");
    }
}

pub fn get_device_facts<'a, 'b: 'a, T>(
    facts: &'a [&'b TypedFact],
    map_facts: impl Fn(&[&'b TypedFact]) -> TractResult<T>,
) -> TractResult<T> {
    if facts.iter().all(|it| it.as_device_fact().is_some()) {
        let device_facts = facts
            .iter()
            .map(|it| it.to_device_fact().map(|it| it.as_ref()))
            .collect::<TractResult<TVec<_>>>()?;
        (map_facts)(device_facts.as_slice())
    } else if facts.iter().all(|it| it.as_device_fact().is_none()) {
        (map_facts)(facts)
    } else {
        bail!("Inconsistent facts: mix of device and host facts");
    }
}

pub fn get_device_fact<'a, T: 'a>(
    fact: &'a TypedFact,
    map_fact: impl Fn(&'a TypedFact) -> TractResult<T>,
) -> TractResult<T> {
    if fact.as_device_fact().is_some() {
        (map_fact)(fact.to_device_fact()?)
    } else {
        (map_fact)(fact)
    }
}

pub fn as_quant_fact<'a>(
    fact: &'a TypedFact,
    format: &dyn BlockQuant,
) -> Option<&'a BlockQuantFact> {
    fact.exotic_fact
        .as_ref()
        .and_then(|of| of.downcast_ref::<BlockQuantFact>())
        .and_then(|bqf| if bqf.format.dyn_eq(format) { Some(bqf) } else { None })
}

pub fn as_q40_tensor(a: &Tensor) -> Option<&BlockQuantStorage> {
    a.storage_as::<BlockQuantStorage>().filter(|bqs| bqs.format().dyn_eq(&Q4_0))
}

pub fn get_quant_fact(t: &DeviceTensor, format: &dyn BlockQuant) -> Option<BlockQuantFact> {
    if let DeviceTensor::Owned(t) = t {
        t.exotic_fact()
            .and_then(|of| of.downcast_ref::<BlockQuantFact>())
            .cloned()
            .filter(|bqf| bqf.format.dyn_eq(format))
    } else {
        None
    }
}

// --- Shared array/copy utilities ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastKind {
    Unicast,
    ByScalarLeft,
    ByScalarRight,
    Nd1,
    Nd2,
    Nd3,
    Nd4,
    Nd5,
    Nd6,
}

impl BroadcastKind {
    pub const ALL: [BroadcastKind; 8] = [
        Self::Unicast,
        Self::ByScalarLeft,
        Self::ByScalarRight,
        Self::Nd1,
        Self::Nd2,
        Self::Nd3,
        Self::Nd4,
        Self::Nd5,
    ];

    pub fn from_rank(rank: usize) -> TractResult<Self> {
        match rank {
            1 => Ok(Self::Nd1),
            2 => Ok(Self::Nd2),
            3 => Ok(Self::Nd3),
            4 => Ok(Self::Nd4),
            5 => Ok(Self::Nd5),
            6 => Ok(Self::Nd6),
            _ => bail!("Unsupported rank {rank} for broadcasting"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Unicast => "unicast",
            Self::ByScalarLeft => "by_scalar_lhs",
            Self::ByScalarRight => "by_scalar_rhs",
            Self::Nd1 => "nd1",
            Self::Nd2 => "nd2",
            Self::Nd3 => "nd3",
            Self::Nd4 => "nd4",
            Self::Nd5 => "nd5",
            Self::Nd6 => "nd6",
        }
    }

    /// Map datum type to the copy kernel type name based on element size.
    /// Copy kernels only care about element size, not the actual type.
    pub fn copy_tname(dt: DatumType) -> &'static str {
        match dt.size_of() {
            1 => "u8",
            2 => "u16",
            4 => "u32",
            8 => "u64",
            _ => panic!("Unsupported element size {} for copy kernel", dt.size_of()),
        }
    }

    pub fn copy_kernel_name(&self, dt: DatumType, prefix: &str) -> TractResult<String> {
        Ok(format!("{prefix}copy_{}_{}", self.name(), Self::copy_tname(dt)))
    }

    pub fn all_copy_kernel_names(prefix: &str) -> Vec<String> {
        let copy_types = ["u8", "u16", "u32", "u64"];
        Self::ALL
            .into_iter()
            .flat_map(|bk| {
                copy_types
                    .into_iter()
                    .map(move |tname| format!("{prefix}copy_{}_{tname}", bk.name()))
            })
            .collect()
    }
}

pub fn compute_broadcast_strides<T: num_traits::Zero + Copy + 'static>(
    shape: &[usize],
    strides: &[isize],
) -> TractResult<TVec<T>>
where
    isize: num_traits::AsPrimitive<T>,
{
    use num_traits::AsPrimitive;
    ensure!(
        shape.len() == strides.len(),
        "Mismatch between shape and strides length while computing broadcast strides"
    );
    Ok(strides
        .iter()
        .zip(shape)
        .map(|(s, dim)| if *dim == 1 { T::zero() } else { s.as_() })
        .collect::<TVec<T>>())
}

pub fn reshape_to_rank_2(shape: &[usize], axis: usize) -> TVec<usize> {
    let dim_axis_0 = shape[0..axis].iter().product::<usize>();
    let dim_axis_2 = shape[axis..].iter().product::<usize>();
    tvec![dim_axis_0, dim_axis_2]
}

pub fn reshape_to_rank_3(shape: &[usize], axis: usize) -> TVec<usize> {
    let dim_axis_0 = shape[0..axis].iter().product::<usize>();
    let dim_axis_1 = shape[axis];
    let dim_axis_2 = shape[axis + 1..].iter().product::<usize>();
    tvec![dim_axis_0, dim_axis_1, dim_axis_2]
}

/// Dispatch function for strided copy_nd kernels. All array ops (broadcast,
/// slice, concat, permute_axes) ultimately call a copy_nd kernel with this
/// signature. The backend derives the kernel name from output rank + dtype.
/// Both offsets are in bytes.
pub type DispatchCopyNdFn = fn(
    input: &crate::tensor::DeviceTensor,
    input_offset: usize,
    input_strides: &[isize],
    output: &crate::tensor::DeviceTensor,
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()>;

/// Copy a slice along `axis` from `src[src_range]` into `dst[dst_range]`.
/// Both ranges are along the given axis; other dimensions are copied fully.
pub fn dispatch_assign_slice(
    dispatch: DispatchCopyNdFn,
    dst: &crate::tensor::DeviceTensor,
    dst_range: std::ops::Range<usize>,
    src: &crate::tensor::DeviceTensor,
    src_range: std::ops::Range<usize>,
    axis: usize,
) -> TractResult<()> {
    let mut zone_shape: TVec<usize> = src.shape().into();
    zone_shape[axis] = src_range.len();
    if zone_shape.iter().product::<usize>() == 0 {
        return Ok(());
    }
    let src_offset = src_range.start * src.strides()[axis] as usize * src.datum_type().size_of();
    let dst_offset = dst_range.start * dst.strides()[axis] as usize * dst.datum_type().size_of();
    dispatch(src, src_offset, src.strides(), dst, dst_offset, &zone_shape, dst.strides())
}

/// Copy from `src` into `dst` with given origins and strides.
/// Origins are element indices per dimension, converted to byte offsets internally.
pub fn dispatch_copy_with_origins(
    dispatch: DispatchCopyNdFn,
    zone_shape: &[usize],
    dst: &crate::tensor::DeviceTensor,
    dst_origin: &[usize],
    dst_strides: &[isize],
    src: &crate::tensor::DeviceTensor,
    src_origin: &[usize],
    src_strides: &[isize],
) -> TractResult<()> {
    if zone_shape.iter().product::<usize>() == 0 {
        return Ok(());
    }
    let dt_size = src.datum_type().size_of();
    let src_offset: usize =
        src_origin.iter().zip(src_strides).map(|(o, s)| o * *s as usize).sum::<usize>() * dt_size;
    let dst_offset: usize =
        dst_origin.iter().zip(dst_strides).map(|(o, s)| o * *s as usize).sum::<usize>() * dt_size;
    dispatch(src, src_offset, src_strides, dst, dst_offset, zone_shape, dst_strides)
}

/// Flat memcpy of `len` bytes from `src` at `src_offset` to `dst` at `dst_offset`.
pub fn dispatch_flat_copy(
    dispatch: DispatchCopyNdFn,
    src: &crate::tensor::DeviceTensor,
    src_byte_offset: usize,
    dst: &crate::tensor::DeviceTensor,
    dst_byte_offset: usize,
    byte_len: usize,
) -> TractResult<()> {
    if byte_len == 0 {
        return Ok(());
    }
    dispatch(src, src_byte_offset, &[1], dst, dst_byte_offset, &[byte_len], &[1])
}

pub fn check_strides_validity(shape: TVec<usize>, strides: TVec<isize>) -> TractResult<()> {
    let mut zipped_shape_strides: Vec<_> = shape.into_iter().zip(strides).collect();
    zipped_shape_strides.sort_by_key(|&(_, stride)| stride);

    let mut prev_stride = 1;
    for (dim, stride) in zipped_shape_strides {
        ensure!((stride == prev_stride) || (dim == 1), "Invalid strides");
        prev_stride *= dim as isize;
    }
    Ok(())
}
