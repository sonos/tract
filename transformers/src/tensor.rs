use std::{fmt::Display, sync::Arc};
use tract_data::internal::OpaquePayload;
use tract_nnef::prelude::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ChunkedTensor(pub TVec<Arc<Tensor>>);

impl ChunkedTensor {
    pub fn new() -> Self {
        Self(tvec!())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, tensor: Arc<Tensor>) {
        self.0.push(tensor)
    }

    pub fn into_opaque_tensor(self) -> Tensor {
        tensor0(Opaque::from(self))
    }
}

impl Default for ChunkedTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for ChunkedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let infos: Vec<String> =
            self.0.iter().map(|t| format!("[{:?}, {:?}]", t.shape(), t.datum_type())).collect();
        write!(f, "ChunkedTensor([{}])", infos.join(", "))
    }
}

impl From<ChunkedTensor> for Opaque {
    fn from(value: ChunkedTensor) -> Self {
        Opaque(Arc::new(value))
    }
}

impl OpaquePayload for ChunkedTensor {
    fn same_as(&self, other: &dyn OpaquePayload) -> bool {
        other.downcast_ref::<Self>().map_or(false, |other| self.0 == other.0)
    }

    fn clarify_to_tensor(&self) -> TractResult<Option<Arc<Tensor>>> {
        if self.0.is_empty() {
            return Ok(None);
        }
        if self.0.len() == 1 {
            return Ok(Some(self.0[0].clone()));
        }

        let tensors: Vec<&Tensor> = self.0.iter().map(|t| t.as_ref()).collect();
        // TODO: might require some generalization to stack columns instead of rows
        let concatenated = Tensor::stack_tensors(0, &tensors)?;
        Ok(Some(Arc::new(concatenated)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_tensor() -> TractResult<()> {
        let t1 = Arc::new(tensor2(&[[1.0f32, 2.0], [3.0, 4.0]]));
        let t2 = Arc::new(tensor2(&[[5.0f32, 6.0]]));

        let mut chunked_tensor = ChunkedTensor::new();
        chunked_tensor.push(t1.clone());
        chunked_tensor.push(t2.clone());

        let opaque_tensor = chunked_tensor.into_opaque_tensor();
        let extracted_chunked_tensor = opaque_tensor
            .to_scalar::<Opaque>()
            .unwrap()
            .downcast_ref::<ChunkedTensor>()
            .unwrap();
        assert_eq!(extracted_chunked_tensor.len(), 2);
        assert_eq!(extracted_chunked_tensor.0[0], t1);
        assert_eq!(extracted_chunked_tensor.0[1], t2);

        Ok(())
    }
}
