use super::*;
use crate::internal::*;

/// Whether every bit pattern of `dt`'s storage is a valid value, so a view can be built
/// over arbitrary bytes of that type.
fn any_bit_pattern(dt: DatumType) -> bool {
    dt.is_copy() && dt != DatumType::Bool
}

/// Views may keep a shape that runs past their bytes (a batch origin with the full
/// tensor shape, walked along a couple of axes only): extent is checked on typed access.
fn check_strides(shape: &[usize], strides: &[isize]) -> TractResult<()> {
    ensure!(
        shape.len() == strides.len(),
        "TensorView shape {shape:?} and strides {strides:?} ranks differ"
    );
    ensure!(strides.iter().all(|s| *s >= 0), "TensorView strides {strides:?} must not be negative");
    Ok(())
}

fn natural_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn typed_slice<D: Datum>(dt: DatumType, bytes: &[u8], len: usize) -> TractResult<&[D]> {
    check_typed::<D>(dt, bytes, len)?;
    if len == 0 {
        return Ok(&[]);
    }
    // Datum type, alignment and length were checked against `bytes` above.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const D, len) })
}

fn typed_slice_mut<D: Datum>(dt: DatumType, bytes: &mut [u8], len: usize) -> TractResult<&mut [D]> {
    check_typed::<D>(dt, bytes, len)?;
    if len == 0 {
        return Ok(&mut []);
    }
    // Datum type, alignment and length were checked against `bytes` above.
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut D, len) })
}

fn check_typed<D: Datum>(dt: DatumType, bytes: &[u8], len: usize) -> TractResult<()> {
    ensure!(
        dt.unquantized() == D::datum_type().unquantized(),
        "Tensor datum type error: tensor is {:?}, accessed as {:?}",
        dt,
        D::datum_type(),
    );
    ensure!(
        len.checked_mul(size_of::<D>()).is_some_and(|l| l <= bytes.len()),
        "TensorView access to {len} {dt:?} overruns its {} bytes",
        bytes.len()
    );
    ensure!(
        len == 0 || bytes.as_ptr().cast::<D>().is_aligned(),
        "TensorView data is not aligned for {dt:?}"
    );
    Ok(())
}

fn item_offset(shape: &[usize], strides: &[isize], coords: &[usize]) -> TractResult<usize> {
    ensure!(
        coords.len() == shape.len() && coords.iter().zip(shape).all(|(&x, &dim)| x < dim),
        "Can't access coordinates {coords:?} of TensorView of shape {shape:?}",
    );
    Ok(coords.iter().zip(strides).map(|(c, s)| c * *s as usize).sum())
}

fn axis_offset_bytes(
    dt: DatumType,
    shape: &[usize],
    strides: &[isize],
    axis: usize,
    pos: usize,
) -> TractResult<usize> {
    ensure!(axis < shape.len(), "Axis {axis} out of TensorView of rank {}", shape.len());
    Ok(pos * strides[axis] as usize * dt.size_of())
}

/// A read-only, type-erased window into a tensor's plain storage: the bytes from the
/// view's origin to the end of the storage, read through `shape` and `strides` (in
/// items, never negative).
///
/// Typed access checks the datum type, the alignment and the extent against those
/// bytes. `as_slice` reads `len()` contiguous items from the origin, which only
/// matches the view's logical content when its strides are natural.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    dt: DatumType,
    bytes: &'a [u8],
    shape: &'a [usize],
    strides: &'a [isize],
}

impl<'a> TensorView<'a> {
    pub(crate) fn of_tensor(tensor: &'a Tensor) -> TensorView<'a> {
        TensorView {
            dt: tensor.dt,
            bytes: tensor.plain_ram_storage().as_bytes(),
            shape: &tensor.shape,
            strides: &tensor.strides,
        }
    }

    /// A view of `tensor`'s storage starting `offset_bytes` in, read with `shape` and
    /// `strides`. Errors if the offset is past the end of the storage.
    pub fn from_bytes(
        tensor: &'a Tensor,
        offset_bytes: usize,
        shape: &'a [usize],
        strides: &'a [isize],
    ) -> TractResult<TensorView<'a>> {
        let bytes = tensor
            .try_as_plain_ram()?
            .as_bytes()
            .get(offset_bytes..)
            .with_context(|| format!("TensorView offset {offset_bytes} out of storage"))?;
        check_strides(shape, strides)?;
        Ok(TensorView { dt: tensor.dt, bytes, shape, strides })
    }

    /// A view over raw `bytes` holding `dt` items. Only for datum types where every bit
    /// pattern is a valid value.
    pub fn from_pod_bytes(
        dt: DatumType,
        bytes: &'a [u8],
        shape: &'a [usize],
        strides: &'a [isize],
    ) -> TractResult<TensorView<'a>> {
        ensure!(any_bit_pattern(dt), "TensorView over raw bytes of {dt:?}");
        check_strides(shape, strides)?;
        Ok(TensorView { dt, bytes, shape, strides })
    }

    pub(crate) fn at_prefix(tensor: &'a Tensor, prefix: &[usize]) -> TractResult<TensorView<'a>> {
        ensure!(
            prefix.len() <= tensor.rank() && prefix.iter().zip(tensor.shape()).all(|(p, d)| p < d),
            "Invalid prefix {:?} for shape {:?}",
            prefix,
            tensor.shape()
        );
        let mut view = Self::of_tensor(tensor);
        for (axis, &pos) in prefix.iter().enumerate() {
            view.offset_axis(axis, pos)?;
        }
        view.shape = &tensor.shape[prefix.len()..];
        view.strides = &tensor.strides[prefix.len()..];
        Ok(view)
    }

    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    /// Number of items in the view's shape.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        natural_len(self.shape)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Moves the view's origin `pos` steps along `axis`, keeping its shape and strides.
    pub fn offset_axis(&mut self, axis: usize, pos: usize) -> TractResult<()> {
        let offset = axis_offset_bytes(self.dt, self.shape, self.strides, axis, pos)?;
        self.bytes = self.bytes.get(offset..).context("TensorView offset out of storage")?;
        Ok(())
    }

    /// The view's origin, typed. Errors on a datum type mismatch.
    #[inline]
    pub fn as_ptr<D: Datum>(&self) -> TractResult<*const D> {
        check_typed::<D>(self.dt, self.bytes, 0)?;
        Ok(self.bytes.as_ptr() as *const D)
    }

    /// The view's origin, cast to `D` without checking the datum type.
    #[inline]
    pub unsafe fn as_ptr_unchecked<D: Datum>(&self) -> *const D {
        self.bytes.as_ptr() as *const D
    }

    /// `len()` contiguous items from the view's origin.
    #[inline]
    pub fn as_slice<D: Datum>(&self) -> TractResult<&'a [D]> {
        typed_slice(self.dt, self.bytes, self.len())
    }

    /// Like [`Self::as_slice`], without checking the datum type, alignment or extent.
    #[inline]
    pub unsafe fn as_slice_unchecked<D: Datum>(&self) -> &'a [D] {
        if self.len() == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.as_ptr_unchecked(), self.len()) }
    }

    /// The item at `coords`.
    pub fn at<D: Datum>(&self, coords: impl AsRef<[usize]>) -> TractResult<&'a D> {
        let offset = item_offset(self.shape, self.strides, coords.as_ref())?;
        let bytes = self.bytes.get(offset * self.dt.size_of()..).unwrap_or(&[]);
        Ok(&typed_slice::<D>(self.dt, bytes, 1)?[0])
    }
}

/// The mutable counterpart of [`TensorView`], obtained from a `&mut Tensor` (or split
/// out of raw bytes of a type where any bit pattern is valid), so writes through it never
/// alias another live view.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    dt: DatumType,
    bytes: &'a mut [u8],
    shape: &'a [usize],
    strides: &'a [isize],
}

impl<'a> TensorViewMut<'a> {
    pub(crate) fn of_tensor(tensor: &'a mut Tensor) -> TensorViewMut<'a> {
        let Tensor { dt, shape, strides, storage, .. } = tensor;
        let bytes = storage.as_plain_ram_mut().expect("Non-plain storage").as_bytes_mut();
        TensorViewMut { dt: *dt, bytes, shape, strides }
    }

    /// A mutable view over raw `bytes` holding `dt` items. Only for datum types where
    /// every bit pattern is a valid value.
    pub fn from_pod_bytes(
        dt: DatumType,
        bytes: &'a mut [u8],
        shape: &'a [usize],
        strides: &'a [isize],
    ) -> TractResult<TensorViewMut<'a>> {
        ensure!(any_bit_pattern(dt), "TensorViewMut over raw bytes of {dt:?}");
        check_strides(shape, strides)?;
        Ok(TensorViewMut { dt, bytes, shape, strides })
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        TensorView { dt: self.dt, bytes: self.bytes, shape: self.shape, strides: self.strides }
    }

    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    /// Number of items in the view's shape.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        natural_len(self.shape)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Moves the view's origin `pos` steps along `axis`, keeping its shape and strides.
    pub fn offset_axis(&mut self, axis: usize, pos: usize) -> TractResult<()> {
        let offset = axis_offset_bytes(self.dt, self.shape, self.strides, axis, pos)?;
        ensure!(offset <= self.bytes.len(), "TensorViewMut offset out of storage");
        self.bytes = &mut std::mem::take(&mut self.bytes)[offset..];
        Ok(())
    }

    /// The view's origin, typed. Errors on a datum type mismatch.
    #[inline]
    pub fn as_ptr_mut<D: Datum>(&mut self) -> TractResult<*mut D> {
        check_typed::<D>(self.dt, self.bytes, 0)?;
        Ok(self.bytes.as_mut_ptr() as *mut D)
    }

    /// The view's origin, cast to `D` without checking the datum type.
    #[inline]
    pub unsafe fn as_ptr_mut_unchecked<D: Datum>(&mut self) -> *mut D {
        self.bytes.as_mut_ptr() as *mut D
    }

    /// `len()` contiguous items from the view's origin.
    #[inline]
    pub fn as_slice<D: Datum>(&self) -> TractResult<&[D]> {
        typed_slice(self.dt, self.bytes, self.len())
    }

    /// `len()` contiguous items from the view's origin.
    #[inline]
    pub fn as_slice_mut<D: Datum>(&mut self) -> TractResult<&mut [D]> {
        let len = self.len();
        typed_slice_mut(self.dt, self.bytes, len)
    }

    /// Like [`Self::as_slice_mut`], without checking the datum type, alignment or extent.
    #[inline]
    pub unsafe fn as_slice_mut_unchecked<D: Datum>(&mut self) -> &mut [D] {
        let len = self.len();
        if len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr_mut_unchecked(), len) }
    }

    /// The item at `coords`.
    pub fn at_mut<D: Datum>(&mut self, coords: impl AsRef<[usize]>) -> TractResult<&mut D> {
        let offset = item_offset(self.shape, self.strides, coords.as_ref())?;
        let start = (offset * self.dt.size_of()).min(self.bytes.len());
        Ok(&mut typed_slice_mut::<D>(self.dt, &mut self.bytes[start..], 1)?[0])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_at_prefix() {
        let a = Tensor::from_shape(&[2, 2], &[1, 2, 3, 4]).unwrap();
        let a_view = a.view_at_prefix(&[1]).unwrap();
        assert_eq!(a_view.shape(), &[2]);
        assert_eq!(a_view.as_slice::<i32>().unwrap(), &[3, 4]);
    }

    #[test]
    fn offsetting_view_stays_in_storage() {
        let a = Tensor::from_shape(&[4], &[1f32, 2., 3., 4.]).unwrap();
        let view = a.view_offsetting(&[2]).unwrap();
        assert!(view.as_slice::<f32>().is_err());
        assert_eq!(*view.at::<f32>([1]).unwrap(), 4.);
        assert!(view.at::<f32>([2]).is_err());
        assert!(a.view_offsetting(&[4]).is_err());
        assert!(a.view_offsetting(&[]).is_err());
    }

    #[test]
    fn empty_view_slices() {
        let a = Tensor::from_shape::<f32>(&[0, 3], &[]).unwrap();
        assert!(a.view().as_slice::<f32>().unwrap().is_empty());
    }

    #[test]
    fn from_bytes_checks_extent() {
        let a = Tensor::from_shape(&[4], &[1f32, 2., 3., 4.]).unwrap();
        assert_eq!(
            TensorView::from_bytes(&a, 8, &[2], &[1]).unwrap().as_slice::<f32>().unwrap(),
            &[3., 4.]
        );
        assert!(TensorView::from_bytes(&a, 8, &[3], &[1]).unwrap().as_slice::<f32>().is_err());
        assert!(TensorView::from_bytes(&a, 20, &[0], &[1]).is_err());
    }

    #[test]
    fn pod_bytes_refuse_non_pod_types() {
        let mut bytes = [1u8; 8];
        assert!(TensorViewMut::from_pod_bytes(DatumType::Bool, &mut bytes, &[8], &[1]).is_err());
        assert!(TensorViewMut::from_pod_bytes(DatumType::U8, &mut bytes, &[8], &[1]).is_ok());
    }

    #[test]
    fn mut_view_writes() {
        let mut a = Tensor::from_shape(&[2, 2], &[1i32, 2, 3, 4]).unwrap();
        let mut v = a.view_mut();
        *v.at_mut::<i32>([1, 0]).unwrap() = 30;
        v.offset_axis(0, 1).unwrap();
        assert!(v.as_slice::<i32>().is_err());
        assert_eq!(*v.at_mut::<i32>([0, 1]).unwrap(), 4);
        assert_eq!(a.try_as_plain_ram().unwrap().as_slice::<i32>().unwrap(), &[1, 2, 30, 4]);
    }
}
