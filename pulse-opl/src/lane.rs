use std::ops::Range;

/// The byte ranges holding `range` along `axis` of one lane of a tensor whose
/// axis 0 is the lane axis, or of the whole tensor when `lane` is `None`.
///
/// Tensors are contiguous, so the region is one run per coordinate of the axes
/// sitting between the lane axis and `axis`. A laned tensor needs `axis` to be
/// at least 1, leaving axis 0 to the lane.
pub(crate) fn lane_runs(
    shape: &[usize],
    dt_size: usize,
    axis: usize,
    lane: Option<usize>,
    range: Range<usize>,
) -> impl Iterator<Item = Range<usize>> {
    let post = shape[axis + 1..].iter().product::<usize>() * dt_size;
    let block = shape[axis] * post;
    let (runs, lane_start) = match lane {
        Some(lane) => {
            let runs: usize = shape[1..axis].iter().product();
            (runs, lane * runs * block)
        }
        None => (shape[..axis].iter().product(), 0),
    };
    let start = lane_start + range.start * post;
    let len = range.len() * post;
    (0..runs).map(move |run| start + run * block..start + run * block + len)
}
