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
        let dims = shape.get_dim();
        if dims.len() == 0 {
            if t.get_tensor_content().len() != 0 {
                Err(format!("content with no dim"))?
            }
            match dtype {
                DT_INT32 => Ok(I32(Array1::from_iter(t.get_int_val().iter().cloned()).into_dyn())),
                DT_STRING => {
                    let s = t.get_string_val()[0].to_vec();
                    let r = Ok(U8(Array1::from_vec(s).into_dyn()));
                    r
                }
                _ => {
                    Err(format!(
                        "Unimplemented case (trivial matrix, {:?} dtype)",
                        dtype
                    ))?
                }
            }
        } else {
            if t.get_tensor_content().len() == 0 {
                Err(format!("some dim, no content"))?
            }
            let dims: Vec<usize> = dims.iter().map(|d| d.size as usize).collect();
            let d = IxDyn(&*dims);
            match dtype {
                DT_FLOAT => {
                    let value: &[f32] = unsafe {
                        ::std::slice::from_raw_parts(
                            t.get_tensor_content().as_ptr() as _,
                            t.get_tensor_content().len() / 4,
                        )
                    };
                    Ok(F32(ArrayD::from_shape_vec(d, value.to_vec())?))
                }
                _ => {
                    Err(format!(
                        "Unimplemented loading case (non trivial matrix, {:?} dtype)",
                        dtype
                    ))?
                }
            }
        }
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
