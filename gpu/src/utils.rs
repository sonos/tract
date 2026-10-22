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
        .filter(|&bqf| bqf.format.dyn_eq(format))
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
    pub const ALL: [BroadcastKind; 9] = [
        Self::Unicast,
        Self::ByScalarLeft,
        Self::ByScalarRight,
        Self::Nd1,
        Self::Nd2,
        Self::Nd3,
        Self::Nd4,
        Self::Nd5,
        Self::Nd6,
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

    pub fn transpose_kernel_name(dt: DatumType, prefix: &str) -> String {
        format!("{prefix}copy_transpose_{}", Self::copy_tname(dt))
    }

    pub fn all_copy_kernel_names(prefix: &str) -> Vec<String> {
        let copy_types = ["u8", "u16", "u32", "u64"];
        Self::ALL
            .into_iter()
            .map(|bk| bk.name())
            .chain(["transpose"])
            .flat_map(|name| {
                copy_types.into_iter().map(move |tname| format!("{prefix}copy_{name}_{tname}"))
            })
            .collect()
    }
}

/// One tensor's layout as a kernel addresses it: an extent per axis and a stride
/// per axis, a broadcast axis holding a stride of zero.
pub struct Layout<'a> {
    pub shape: &'a mut TVec<usize>,
    pub strides: &'a mut TVec<isize>,
}

/// Merges the adjacent axes every layout walks as one run, until the rank fits
/// `max_rank`, and answers the rank reached.
///
/// A pair merges when, for every layout, either the outer stride is the inner
/// one times its extent -- so the two axes spell out one contiguous run -- or the
/// axis is broadcast on both halves and stays broadcast. Merging is exact: it
/// renames a walk the kernel was already going to make, and the extents the
/// kernel is given are the output's.
pub fn merge_axes_to_fit(layouts: &mut [Layout], max_rank: usize) -> usize {
    let mut rank = layouts.first().map(|l| l.shape.len()).unwrap_or(0);
    let mut axis = 0;
    while rank > max_rank && axis + 1 < rank {
        let inner_extent = layouts.last().map(|l| l.shape[axis + 1]).unwrap_or(1);
        let runs = |l: &Layout| {
            l.strides[axis] == l.strides[axis + 1] * inner_extent as isize
                || (l.strides[axis] == 0 && l.strides[axis + 1] == 0)
        };
        if !layouts.iter().all(runs) {
            axis += 1;
            continue;
        }
        for l in layouts.iter_mut() {
            // A layout broadcast along both halves is broadcast along the merged
            // axis, and says so with an extent of one.
            l.shape[axis] = if l.strides[axis] == 0 && l.strides[axis + 1] == 0 {
                1
            } else {
                l.shape[axis] * l.shape[axis + 1]
            };
            l.shape.remove(axis + 1);
            l.strides[axis] = l.strides[axis + 1];
            l.strides.remove(axis + 1);
        }
        rank -= 1;
    }
    rank
}

/// Highest rank the backends' `copy_nd` kernels address.
pub const MAX_COPY_RANK: usize = 6;

/// A `copy_nd` zone: the output's extents, and the strides each side walks it by.
pub struct CopyGeometry {
    pub shape: TVec<usize>,
    pub input_strides: TVec<isize>,
    pub output_strides: TVec<isize>,
}

/// Merges a copy's adjacent axes with [`merge_axes_to_fit`] until it fits
/// [`MAX_COPY_RANK`]. The rank answered still exceeds it when the two layouts
/// leave no pair to merge, which a caller must check before dispatching.
pub fn merge_copy_axes(
    input_strides: &[isize],
    output_shape: &[usize],
    output_strides: &[isize],
) -> CopyGeometry {
    let mut in_shape: TVec<usize> = output_shape.into();
    let mut geo = CopyGeometry {
        shape: output_shape.into(),
        input_strides: input_strides.into(),
        output_strides: output_strides.into(),
    };
    merge_axes_to_fit(
        &mut [
            Layout { shape: &mut in_shape, strides: &mut geo.input_strides },
            Layout { shape: &mut geo.shape, strides: &mut geo.output_strides },
        ],
        MAX_COPY_RANK,
    );
    geo
}

/// Threads per group for a copy. Bandwidth-bound, a copy wants every thread
/// slot of a compute unit filled, which a group of the maximum size can leave
/// a third of on CUDA.
pub const COPY_THREADS: usize = 256;

/// What a launch may ask of the grid's `y` and `z`, each, on every backend.
pub const MAX_COPY_GRID_YZ: usize = 65535;

/// The groups a `copy_ndN` kernel runs on, and the threads of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyLaunch {
    pub groups: [usize; 3],
    pub threads: usize,
}

/// The grid a `copy_ndN` kernel runs on, for a `shape` already through
/// [`merge_copy_axes`] and with no zero extent. A group covers whole rows of the innermost axis, `x`
/// the rows a single group does not hold, and `y` and `z` together every axis
/// beyond the innermost two. The rows take `x` because a tensor has far more of
/// them, and `y` and `z` hold [`MAX_COPY_GRID_YZ`] each where `x` holds 2^31;
/// the kernel reads the pair back as `y + z * groups.y`, so `z` carries only
/// what does not fit in `y` and the last of its groups can run past the axes
/// and has to return. Two dimensions bound those axes at 65535 squared between
/// them, which no tensor a device holds reaches -- it would take 4 GB of a
/// one-byte type with both innermost axes at one -- and a shape past it is
/// refused rather than launched.
///
/// `threads` stays a whole multiple of the innermost extent -- the kernel maps
/// a thread to a row by dividing by it -- and only a row wider than
/// `max_threads` breaks that, in which case one group takes one row and strides
/// through it. A rank-1 copy is a plain grid-stride loop over `x`.
pub fn copy_launch(shape: &[usize], max_threads: usize) -> CopyLaunch {
    let rank = shape.len();
    assert!((1..=MAX_COPY_RANK).contains(&rank), "Unsupported rank {rank} for a copy launch");
    if rank == 1 {
        let threads = shape[0].clamp(1, max_threads);
        return CopyLaunch { groups: [shape[0].div_ceil(threads), 1, 1], threads };
    }
    let inner = shape[rank - 1];
    let prev = shape[rank - 2];
    let outer: usize = shape[..rank - 2].iter().product();
    assert!(
        outer <= MAX_COPY_GRID_YZ * MAX_COPY_GRID_YZ,
        "A copy of {shape:?} spans {outer} of the axes beyond its innermost two, over the {} two grid dimensions hold",
        MAX_COPY_GRID_YZ * MAX_COPY_GRID_YZ
    );
    let (threads, x) = if inner > max_threads {
        (max_threads, prev)
    } else {
        let rows = (max_threads / inner).min(prev);
        (inner * rows, prev.div_ceil(rows))
    };
    CopyLaunch {
        groups: [x, outer.min(MAX_COPY_GRID_YZ), outer.div_ceil(MAX_COPY_GRID_YZ)],
        threads,
    }
}

/// Edge of the square tile a transposing copy stages through group memory.
pub const TRANSPOSE_TILE: usize = 32;

/// Rows of threads in a transposing copy's group, `TRANSPOSE_TILE` wide: each
/// thread moves `TRANSPOSE_TILE / TRANSPOSE_ROWS` elements of the tile.
pub const TRANSPOSE_ROWS: usize = 8;

/// A copy whose input runs contiguously along the output's second-innermost
/// axis while the output runs contiguously along its innermost one: a batch of
/// 2D transposes. Walked element by element, one of the two sides strides
/// through memory; tiled, each group reads a `TRANSPOSE_TILE` square along the
/// input's run and writes it back along the output's.
///
/// The batch is `groups[2]`, the output's rows `groups[1]` tiles, its columns
/// `groups[0]`. Strides are in elements; a row of the input and a column of
/// the output are implied unit strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransposeLaunch {
    pub rows: usize,
    pub cols: usize,
    pub in_batch_stride: isize,
    pub in_col_stride: isize,
    pub out_batch_stride: isize,
    pub out_row_stride: isize,
    pub groups: [usize; 3],
}

/// The tiled launch for a copy, when it is a batch of transposes as
/// [`TransposeLaunch`] describes once its unit axes are dropped and every
/// mergeable pair merged: at most one batch axis, both transposed extents at
/// least a tile wide, and the grid within [`MAX_COPY_GRID_YZ`]. Anything else
/// goes through [`copy_launch`].
pub fn transpose_launch(
    input_strides: &[isize],
    output_shape: &[usize],
    output_strides: &[isize],
) -> Option<TransposeLaunch> {
    let kept = |strides: &[isize]| -> TVec<isize> {
        strides.iter().zip(output_shape).filter(|(_, d)| **d != 1).map(|(s, _)| *s).collect()
    };
    let mut in_shape: TVec<usize> = output_shape.iter().copied().filter(|d| *d != 1).collect();
    let mut geo = CopyGeometry {
        shape: in_shape.clone(),
        input_strides: kept(input_strides),
        output_strides: kept(output_strides),
    };
    let rank = merge_axes_to_fit(
        &mut [
            Layout { shape: &mut in_shape, strides: &mut geo.input_strides },
            Layout { shape: &mut geo.shape, strides: &mut geo.output_strides },
        ],
        1,
    );
    if !(2..=3).contains(&rank)
        || geo.input_strides[rank - 2] != 1
        || geo.output_strides[rank - 1] != 1
        || geo.input_strides[rank - 1] == 1
    {
        return None;
    }
    let (rows, cols) = (geo.shape[rank - 2], geo.shape[rank - 1]);
    let (batch, in_batch_stride, out_batch_stride) = if rank == 3 {
        (geo.shape[0], geo.input_strides[0], geo.output_strides[0])
    } else {
        (1, 0, 0)
    };
    let groups = [cols.div_ceil(TRANSPOSE_TILE), rows.div_ceil(TRANSPOSE_TILE), batch];
    if rows < TRANSPOSE_TILE
        || cols < TRANSPOSE_TILE
        || groups[1] > MAX_COPY_GRID_YZ
        || groups[2] > MAX_COPY_GRID_YZ
    {
        return None;
    }
    Some(TransposeLaunch {
        rows,
        cols,
        in_batch_stride,
        in_col_stride: geo.input_strides[rank - 1],
        out_batch_stride,
        out_row_stride: geo.output_strides[rank - 2],
        groups,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The moves around an NHWC conv are what the tiled copy is for, and a
    /// fallback to the element walk would still be correct, so only the
    /// launch shows it.
    #[test]
    fn transpose_launch_takes_layout_moves() {
        let (c, h, w) = (128isize, 64isize, 48isize);
        let to_nhwc = transpose_launch(
            &[c * h * w, w, 1, h * w],
            &[1, h as usize, w as usize, c as usize],
            &[h * w * c, w * c, c, 1],
        );
        assert_eq!(
            to_nhwc.map(|t| (t.rows, t.cols, t.in_col_stride, t.out_row_stride, t.groups)),
            Some((3072, 128, 3072, 128, [4, 96, 1]))
        );
        let to_nchw = transpose_launch(
            &[h * w * c, 1, w * c, c],
            &[1, c as usize, h as usize, w as usize],
            &[c * h * w, h * w, w, 1],
        );
        assert_eq!(
            to_nchw.map(|t| (t.rows, t.cols, t.in_col_stride, t.out_row_stride, t.groups)),
            Some((128, 3072, 128, 3072, [96, 4, 1]))
        );
        let batched = transpose_launch(&[40 * 70, 1, 70], &[3, 70, 40], &[70 * 40, 40, 1]);
        assert_eq!(batched.map(|t| (t.in_batch_stride, t.groups)), Some((2800, [2, 3, 3])));
        assert_eq!(transpose_launch(&[1, 100], &[100, 4], &[4, 1]), None);
        assert_eq!(transpose_launch(&[64, 1], &[64, 64], &[64, 1]), None);
    }

    /// Every element of a copy has to be reached, through groups of no more
    /// threads than asked and a grid inside [`MAX_COPY_GRID_YZ`].
    #[test]
    fn copy_launch_covers_every_element() {
        const EXTENTS: [usize; 9] = [1, 2, 3, 17, 255, 256, 257, 1024, 70000];
        for max_threads in [32, 256, 1024] {
            for rank in 1..=4 {
                for ix in 0..EXTENTS.len().pow(rank as u32) {
                    let shape: TVec<usize> = (0..rank)
                        .map(|axis| EXTENTS[ix / EXTENTS.len().pow(axis as u32) % EXTENTS.len()])
                        .collect();
                    let outer: usize = shape[..rank.max(2) - 2].iter().product();
                    if shape
                        .iter()
                        .try_fold(1usize, |a, d| a.checked_mul(*d))
                        .is_none_or(|n| n > 1 << 36)
                        || outer > MAX_COPY_GRID_YZ * MAX_COPY_GRID_YZ
                    {
                        continue;
                    }
                    let CopyLaunch { groups, threads } = copy_launch(&shape, max_threads);
                    assert!(threads >= 1 && threads <= max_threads, "{shape:?}: {threads}");
                    assert!(groups[1] <= MAX_COPY_GRID_YZ && groups[2] <= MAX_COPY_GRID_YZ);
                    if rank == 1 {
                        assert!(groups[0] * threads >= shape[0], "{shape:?}");
                        continue;
                    }
                    let inner = shape[rank - 1];
                    let rows = (threads / inner).max(1);
                    assert!(inner > threads || threads % inner == 0, "{shape:?}: {threads}");
                    assert!(groups[0] * rows >= shape[rank - 2], "{shape:?}: {groups:?}");
                    assert!(groups[1] * groups[2] >= outer, "{shape:?}: {groups:?}");
                }
            }
        }
    }
}
