use ndarray::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Matrix {
    F32(ArrayD<f32>),
    I32(ArrayD<i32>),
    U8(ArrayD<u8>),
}

impl Matrix {
    pub fn from_pb(t: &::tfpb::tensor::TensorProto) -> ::Result<Matrix> {
        use tfpb::types::DataType::*;
        use ndarray::*;
        use Matrix::*;
        let dtype = t.get_dtype();
        let shape = t.get_tensor_shape();
        let mut dims = shape.get_dim();
        let dims = if dims.len() == 0 {
            vec!(1)
        } else {
            dims.iter().map(|d| d.size as usize).collect()
        };
        let content = t.get_tensor_content();
        if content.len() == 0 {
            match dtype {
                DT_INT32 => Ok(Matrix::I32(Array1::from_iter(t.get_int_val().iter().cloned()).into_dyn())),
                DT_FLOAT => Ok(Matrix::F32(Array1::from_iter(t.get_float_val().iter().cloned()).into_dyn())),
                DT_STRING => {
                    if t.get_string_val().len() != 1 {
                        Err(format!("Multiple string tensor not supported"))?
                    }
                    Ok(Matrix::U8(Array1::from_iter(t.get_string_val()[0].iter().cloned()).into_dyn()))
            },
                _ => Err(format!("Missing simple tensor parser: type:{:?}", dtype))?
            }
        } else {
            match dtype {
                DT_FLOAT => Ok(Matrix::F32(Self::from_content(dims, content)?)),
                DT_INT32 => Ok(Matrix::I32(Self::from_content(dims, content)?)),
                _ => Err(format!("Missing tensor parser: dims:{:?} type:{:?}, content.len:{}", dims, dtype, content.len()))?
            }
        }
    }

    pub fn from_content<T: Copy>(dims: Vec<usize>, content:&[u8]) -> ::Result<ArrayD<T>> {
        let value: &[T] = unsafe {
            ::std::slice::from_raw_parts(
                content.as_ptr() as _,
                content.len() / ::std::mem::size_of::<T>(),
            )
        };
        Ok(Array1::from_iter(value.iter().cloned()).into_shape(dims)?.into_dyn())
    }

    pub fn take_f32s(self) -> Option<ArrayD<f32>> {
        if let Matrix::F32(it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn as_f32s(&self) -> Option<&ArrayD<f32>> {
        if let &Matrix::F32(ref it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn as_i32s(&self) -> Option<&ArrayD<i32>> {
        if let &Matrix::I32(ref it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn take_i32s(self) -> Option<ArrayD<i32>> {
        if let Matrix::I32(it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn as_u8s(&self) -> Option<&ArrayD<u8>> {
        if let &Matrix::U8(ref it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn take_u8s(self) -> Option<ArrayD<u8>> {
        if let Matrix::U8(it) = self {
            Some(it)
        } else {
            None
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            &Matrix::I32(ref it) => it.shape(),
            &Matrix::F32(ref it) => it.shape(),
            &Matrix::U8(ref it) => it.shape(),
            _ => unimplemented!(),
        }
    }
}
