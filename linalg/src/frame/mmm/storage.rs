use std::fmt;
use std::fmt::Debug;
use tract_data::internal::*;

use super::MMMInputValue;

/// Non-plain tensor storage for packed matrices.
///
/// Holds one or more `Box<dyn MMMInputValue>` values with an optional batch
/// shape, replacing the previous `Tensor` + double-downcast pattern.
#[derive(Clone)]
pub struct PackedMatrixStorage {
    values: Vec<Box<dyn MMMInputValue>>,
    batch_shape: TVec<usize>,
    batch_strides: TVec<isize>,
}
impl PartialEq for PackedMatrixStorage {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}
impl Eq for PackedMatrixStorage {}

impl PackedMatrixStorage {
    /// Scalar storage (one value, empty shape).
    pub fn new(value: Box<dyn MMMInputValue>) -> Self {
        PackedMatrixStorage { values: vec![value], batch_shape: tvec![], batch_strides: tvec![] }
    }

    /// Batched storage (shape like `[batch, group]`).
    pub fn new_batched(shape: &[usize], values: Vec<Box<dyn MMMInputValue>>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected, "values length must match shape product");
        let strides = Self::compute_strides(shape);
        PackedMatrixStorage { values, batch_shape: shape.into(), batch_strides: strides }
    }

    fn compute_strides(shape: &[usize]) -> TVec<isize> {
        let mut strides: TVec<isize> = tvec![0; shape.len()];
        if !shape.is_empty() {
            strides[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as isize;
            }
        }
        strides
    }

    /// Scalar access (asserts single value).
    #[inline]
    pub fn value(&self) -> &dyn MMMInputValue {
        debug_assert_eq!(self.values.len(), 1);
        &*self.values[0]
    }

    /// Batched access by coordinates.
    pub fn value_at(&self, coords: &[usize]) -> &dyn MMMInputValue {
        let idx = self.flat_index(coords);
        &*self.values[idx]
    }

    /// Batched access by flat (pre-computed) index.
    #[inline]
    pub fn value_at_flat(&self, idx: usize) -> &dyn MMMInputValue {
        &*self.values[idx]
    }

    pub fn values(&self) -> &[Box<dyn MMMInputValue>] {
        &self.values
    }

    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    pub fn batch_strides(&self) -> &[isize] {
        &self.batch_strides
    }

    /// Convert to a Tensor with the given logical datum type.
    pub fn into_tensor(self, dt: DatumType) -> Tensor {
        let shape: TVec<usize> = self.batch_shape.clone();
        Tensor::from_storage(dt, &shape, self)
    }

    fn flat_index(&self, coords: &[usize]) -> usize {
        coords.iter().zip(self.batch_strides.iter()).map(|(c, s)| *c as isize * s).sum::<isize>()
            as usize
    }
}

impl fmt::Debug for PackedMatrixStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedMatrixStorage({} values, shape={:?})", self.values.len(), self.batch_shape)
    }
}

impl fmt::Display for PackedMatrixStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedMatrixStorage({} values, shape={:?})", self.values.len(), self.batch_shape)
    }
}

impl TensorStorage for PackedMatrixStorage {
    fn byte_len(&self) -> usize {
        // Approximate: sum of individual value sizes isn't precise but gives a ballpark
        self.values.len() * std::mem::size_of::<Box<dyn MMMInputValue>>()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(self.clone())
    }

    fn as_plain(&self) -> Option<&PlainStorage> {
        None
    }

    fn as_plain_mut(&mut self) -> Option<&mut PlainStorage> {
        None
    }

    fn into_plain(self: Box<Self>) -> Option<PlainStorage> {
        None
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        for v in &self.values {
            v.dyn_hash(state);
        }
    }

    fn exotic_fact(&self, _shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        if self.values.len() == 1 {
            Ok(Some(dyn_clone::clone_box(self.values[0].exotic_fact())))
        } else {
            let facts: TVec<Box<dyn ExoticFact>> =
                self.values.iter().map(|v| dyn_clone::clone_box(v.exotic_fact())).collect();
            Ok(Some(Box::new(facts)))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutputStoreSpec {
    View { m_axis: Option<usize>, n_axis: Option<usize>, mr: usize, nr: usize },
    Strides { row_byte_stride: isize, col_byte_stride: isize, mr: usize, nr: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct OutputStore {
    pub(crate) ptr: *mut u8,
    pub(crate) row_byte_stride: isize,
    pub(crate) col_byte_stride: isize,
    pub(crate) panel_row_byte_stride: isize,
    pub(crate) panel_col_byte_stride: isize,
    pub(crate) item_size: usize,
    pub(crate) item_count: usize,
    pub(crate) mr: usize,
}

unsafe impl Send for OutputStore {}
unsafe impl Sync for OutputStore {}

impl OutputStoreSpec {
    #[inline]
    pub unsafe fn wrap(&self, tensor: &TensorView) -> OutputStore {
        let (mr, nr, row_byte_stride, col_byte_stride) = unsafe { self.compute_strides(tensor) };
        OutputStore {
            ptr: unsafe { tensor.as_ptr_unchecked::<u8>() } as _,
            row_byte_stride,
            col_byte_stride,
            panel_row_byte_stride: row_byte_stride * mr as isize,
            panel_col_byte_stride: col_byte_stride * nr as isize,
            item_size: tensor.datum_type().size_of(),
            mr,
            item_count: tensor.len(),
        }
    }

    #[inline]
    unsafe fn compute_strides(&self, tensor: &TensorView) -> (usize, usize, isize, isize) {
        let size_of = tensor.datum_type().size_of() as isize;
        match self {
            OutputStoreSpec::View { m_axis, n_axis, mr, nr, .. } => {
                let tensor_strides = tensor.strides();
                let row_item_stride =
                    m_axis.map(|ax| *unsafe { tensor_strides.get_unchecked(ax) }).unwrap_or(0);
                let col_item_stride =
                    n_axis.map(|ax| *unsafe { tensor_strides.get_unchecked(ax) }).unwrap_or(0);
                let row_byte_stride = row_item_stride * size_of;
                let col_byte_stride = col_item_stride * size_of;
                (*mr, *nr, row_byte_stride, col_byte_stride)
            }
            OutputStoreSpec::Strides { row_byte_stride, col_byte_stride, mr, nr, .. } => {
                (*mr, *nr, *row_byte_stride, *col_byte_stride)
            }
        }
    }
}

impl OutputStore {
    #[inline]
    pub(super) unsafe fn tile_c(&self, down: usize, right: usize) -> OutputStoreKer {
        unsafe {
            let (down, right) = (down as isize, right as isize);
            OutputStoreKer {
                ptr: self
                    .ptr
                    .offset(self.panel_row_byte_stride * down + self.panel_col_byte_stride * right)
                    as *mut _,
                row_byte_stride: self.row_byte_stride,
                col_byte_stride: self.col_byte_stride,
                item_size: self.item_size,
            }
        }
    }

    #[inline]
    pub fn item_size(&self) -> usize {
        self.item_size
    }

    #[inline]
    pub(super) unsafe fn set_from_tile(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        unsafe {
            if self.item_size() == 1 {
                self.set_from_tile_t::<i8>(down, right, height, width, tile)
            } else if self.item_size() == 2 {
                self.set_from_tile_t::<i16>(down, right, height, width, tile)
            } else if self.item_size() == 4 {
                self.set_from_tile_t::<i32>(down, right, height, width, tile)
            } else {
                self.set_from_tile_t::<i64>(down, right, height, width, tile)
            }
        }
    }

    #[inline]
    unsafe fn set_from_tile_t<T: Datum + Copy>(
        &self,
        down: usize,
        right: usize,
        height: usize,
        width: usize,
        tile: &OutputStoreKer,
    ) {
        unsafe {
            let tile = tile.ptr as *mut T;
            let dst = self.ptr.add(
                self.panel_row_byte_stride as usize * down
                    + self.panel_col_byte_stride as usize * right,
            );
            for y in 0..height as isize {
                for x in 0..width as isize {
                    let value = tile.offset(y + x * self.mr as isize);
                    let dst = dst.offset(y * self.row_byte_stride + x * self.col_byte_stride);
                    *(dst as *mut T) = *value;
                }
            }
        }
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct OutputStoreKer {
    pub ptr: *mut u8,
    pub row_byte_stride: isize,
    pub col_byte_stride: isize,
    pub item_size: usize,
}
