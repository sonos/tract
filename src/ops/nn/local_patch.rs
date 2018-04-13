use Result;
use ndarray::prelude::*;

#[derive(Debug)]
pub enum DataFormat {
    NHWC,
}

#[derive(Debug, PartialEq)]
pub enum Padding {
    Valid,
    Same,
}

pub struct ImageWrapper<'a, T: 'a>(ArrayView3<'a, T>);

impl<'a, T> ImageWrapper<'a, T> {
    pub fn height(&self) -> usize {
        self.0.shape()[0]
    }
    pub fn width(&self) -> usize {
        self.0.shape()[1]
    }
    pub fn depth(&self) -> usize {
        self.0.shape()[2]
    }
}

pub struct BatchImageWrapper<'a, T: 'a>(pub ArrayView4<'a, T>);

#[allow(dead_code)]
impl<'a, T> BatchImageWrapper<'a, T> {
    pub fn count(&self) -> usize {
        self.0.shape()[0]
    }
    pub fn n(&self) -> usize {
        self.0.shape()[0]
    }
    pub fn height(&self) -> usize {
        self.0.shape()[1]
    }
    pub fn h(&self) -> usize {
        self.0.shape()[1]
    }
    pub fn width(&self) -> usize {
        self.0.shape()[2]
    }
    pub fn w(&self) -> usize {
        self.0.shape()[2]
    }
    pub fn depth(&self) -> usize {
        self.0.shape()[3]
    }
    pub fn d(&self) -> usize {
        self.0.shape()[3]
    }
}

#[derive(Debug)]
pub struct LocalPatch {
    pub _data_format: DataFormat,
    pub padding: Padding,
    pub strides: Vec<usize>,
}

impl LocalPatch {
    pub fn build(pb: &::tfpb::node_def::NodeDef) -> Result<LocalPatch> {
        if let Some(data_format) = pb.get_attr().get("data_format") {
            if data_format.get_s() == b"NCHW" {
                Err("NCHW data_format not implemented")?
            }
        }
        let strides = pb.get_attr()
            .get("strides")
            .ok_or("expect strides in Conv2D args")?
            .get_list()
            .get_i()
            .iter()
            .map(|a| *a as usize)
            .collect();
        let padding = pb.get_attr()
            .get("padding")
            .ok_or("expect padding in Conv2D args")?;
        let padding = match padding.get_s() {
            b"VALID" => Padding::Valid,
            b"SAME" => Padding::Same,
            s => Err(format!(
                "unsupported Padding {}",
                String::from_utf8_lossy(s)
            ))?,
        };
        Ok(LocalPatch {
            _data_format: DataFormat::NHWC,
            padding,
            strides,
        })
    }

    pub fn adjusted_dim(
        &self,
        in_rows: usize,
        in_cols: usize,
        (filter_rows, filter_cols): (usize, usize),
    ) -> (usize, usize) {
        let stride = self.strides[1];
        match self.padding {
            Padding::Same => (
                (in_rows as f32 / stride as f32).ceil() as usize,
                (in_cols as f32 / stride as f32).ceil() as usize,
            ),
            Padding::Valid => (
                ((in_rows - filter_rows + 1) as f32 / stride as f32).ceil() as usize,
                ((in_cols - filter_cols + 1) as f32 / stride as f32).ceil() as usize,
            ),
        }
    }

    pub fn pad<T>(
        &self,
        data: ArrayView4<T>,
        shape: (usize, usize),
        item: T,
    ) -> Result<Option<Array4<T>>>
    where
        T: Copy + ::num_traits::Zero + ::std::fmt::Debug,
    {
        let img = BatchImageWrapper(data);
        let stride = self.strides[1];
        let (filter_rows, filter_cols) = shape;

        if self.padding == Padding::Same {
            // https://www.tensorflow.org/api_guides/python/nn#Convolution
            let v_padding = ::std::cmp::max(
                0,
                filter_rows - if img.height() % stride == 0 {
                    stride
                } else {
                    img.height() % stride
                },
            );
            let h_padding = ::std::cmp::max(
                0,
                filter_cols - if img.width() % stride == 0 {
                    stride
                } else {
                    img.width() % stride
                },
            );
            let left_padding = h_padding / 2;
            let right_padding = h_padding - left_padding;
            let top_padding = v_padding / 2;
            let bottom_padding = v_padding - top_padding;
            let left_padding = ::ndarray::Array4::<T>::from_elem(
                (img.count(), img.height(), left_padding, img.depth()),
                item,
            );
            let right_padding = ::ndarray::Array4::<T>::from_elem(
                (img.count(), img.height(), right_padding, img.depth()),
                item,
            );
            let tmp = ::ndarray::stack(
                ::ndarray::Axis(2),
                &[left_padding.view(), data.view(), right_padding.view()],
            )?;
            let top_padding = ::ndarray::Array4::<T>::from_elem(
                (img.count(), top_padding, tmp.shape()[2], img.depth()),
                item,
            );
            let bottom_padding = ::ndarray::Array4::<T>::from_elem(
                (img.count(), bottom_padding, tmp.shape()[2], img.depth()),
                item,
            );
            let a = ::ndarray::stack(
                ::ndarray::Axis(1),
                &[top_padding.view(), tmp.view(), bottom_padding.view()],
            )?;
            Ok(Some(a))
        } else {
            Ok(None)
        }
    }

    // data is expected in HWC
    pub fn mk_patches<T: Copy + ::num_traits::Zero + ::std::fmt::Debug>(
        &self,
        data: ArrayView<T, Ix3>,
        shape: (usize, usize),
    ) -> Result<Array2<T>> {
        if self.strides.len() != 4 || self.strides[0] != 1 && self.strides[3] != 1
            || self.strides[1] != self.strides[2]
        {
            Err(format!(
                "strides must be of the form [1, s, s, 1], found {:?}",
                self.strides
            ))?
        }
        let img = ImageWrapper(data);
        let stride = self.strides[1];
        let (filter_rows, filter_cols) = shape;

        let (out_height, out_width) =
            self.adjusted_dim(img.height(), img.width(), (filter_rows, filter_cols));

        let patches_size = (
            (out_height * out_width) as usize,
            filter_rows * filter_cols * img.depth(),
        );

        let mut patches = unsafe { ::ndarray::Array2::<T>::uninitialized(patches_size) };
        let data = data.into_shape((1, img.height(), img.width(), img.depth()))?;
        let padded = self.pad(data, (filter_rows, filter_cols), T::zero())?;
        let data = padded.as_ref().map(|a| a.view()).unwrap_or(data.view());
        for i_x in 0..out_width {
            for i_y in 0..out_height {
                let mut patch_row = patches.row_mut(i_y * out_width + i_x);
                for f_x in 0..filter_cols {
                    for f_y in 0..filter_rows {
                        for d in 0..img.depth() {
                            let loc = &mut patch_row
                                [f_y * img.depth() * filter_cols + f_x * img.depth() + d];
                            *loc = data[(0, i_y * stride + f_y, i_x * stride + f_x, d)];
                        }
                    }
                }
            }
        }
        Ok(patches)
    }
}

pub fn into_4d<T>(data: ArrayD<T>) -> Result<Array4<T>> {
    if data.shape().len() != 4 {
        Err(format!("Expeted 4D shape, found: {:?}", data.shape()))?
    }
    let shape = (
        data.shape()[0],
        data.shape()[1],
        data.shape()[2],
        data.shape()[3],
    );
    Ok(data.into_shape(shape)?)
}

#[cfg(all(test, feature = "tensorflow"))]
pub mod proptests {
    #![allow(non_snake_case)]
    use tfpb;
    use tfpb::types::DataType::DT_FLOAT;

    pub fn placeholder(name: &str) -> tfpb::node_def::NodeDef {
        tfpb::node()
            .name(name)
            .op("Placeholder")
            .attr("dtype", DT_FLOAT)
    }
}
