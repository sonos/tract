use crate::internal::*;

#[derive(Serialize, Deserialize)]
struct TensorSerdeProxy<'a> {
    dt: DatumType,
    align: usize,
    shape: TVec<usize>,
    le_data: Option<&'a [u8]>,
}

impl serde::Serialize for Tensor {
    #[cfg(target_endian = "big")]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        unimplemented!("Serialization unsupported on big-endian platforms");
    }

    #[cfg(target_endian = "little")]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let proxy = match self.dt {
            DatumType::String | DatumType::Blob | DatumType::TDim => unimplemented!(),
            _ => TensorSerdeProxy {
                dt: self.dt,
                align: self.layout.align(),
                shape: self.shape.clone(),
                le_data: Some(unsafe { std::slice::from_raw_parts(self.data, self.len()) }),
            },
        };
        proxy.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Tensor {
    #[cfg(target_endian = "big")]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        unimplemented!("Serialization unsupported on big-endian platforms");
    }

    #[cfg(target_endian = "little")]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let proxy = TensorSerdeProxy::deserialize(deserializer)?;
        match proxy.dt {
            DatumType::String | DatumType::Blob | DatumType::TDim => unimplemented!(),
            _ => unsafe {
                Tensor::from_raw_dt_align(
                    proxy.dt,
                    proxy.align,
                    &*proxy.shape,
                    proxy.le_data.unwrap(),
                )
                .map_err(serde::de::Error::custom)
            },
        }
    }
}
