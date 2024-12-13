use foreign_types::{foreign_type, ForeignType};
use metal::mps::{Kernel, KernelRef};
use metal::{Buffer, CommandBuffer, Device, NSUInteger};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use paste::paste;

use crate::command_buffer::TCommandBuffer;

// From https://github.com/gfx-rs/metal-rs

#[inline]
unsafe fn obj_drop<T>(p: *mut T) {
    msg_send![(p as *mut Object), release]
}

#[inline]
unsafe fn obj_clone<T: 'static>(p: *mut T) -> *mut T {
    msg_send![(p as *mut Object), retain]
}

fn nsstring_as_str(nsstr: &objc::runtime::Object) -> &str {
    let bytes = unsafe {
        let bytes: *const std::os::raw::c_char = msg_send![nsstr, UTF8String];
        bytes as *const u8
    };
    let len: NSUInteger = unsafe { msg_send![nsstr, length] };
    unsafe {
        let bytes = std::slice::from_raw_parts(bytes, len as usize);
        std::str::from_utf8(bytes).unwrap()
    }
}

/// Define a Rust wrapper for an Objective-C opaque type.
///
/// This macro adapts the `foreign-types` crate's [`foreign_type!`]
/// macro to Objective-C, defining Rust types that represent owned and
/// borrowed forms of some underlying Objective-C type, using
/// Objective-C's reference counting to manage its lifetime.
///
/// Given a use of the form:
///
/// ```ignore
/// foreign_obj_type! {
///     type CType = MTLBuffer;   // underlying Objective-C type
///     pub struct Buffer;        // owned Rust type
///     pub struct BufferRef;     // borrowed Rust type
///     type ParentType = ResourceRef;  // borrowed parent class
/// }
/// ```
///
/// This defines the types `Buffer` and `BufferRef` as owning and
/// non-owning types, analogous to `String` and `str`, that manage
/// some underlying `*mut MTLBuffer`:
///
/// - Both `Buffer` and `BufferRef` implement [`obj::Message`], indicating
///   that they can be sent Objective-C messages.
///
/// - Dropping a `Buffer` sends the underlying `MTLBuffer` a `release`
///   message, and cloning a `BufferRef` sends a `retain` message and
///   returns a new `Buffer`.
///
/// - `Buffer` dereferences to `BufferRef`.
///
/// - `BufferRef` dereferences to its parent type `ResourceRef`. The
///   `ParentType` component is optional; if omitted, the `Ref` type
///   doesn't implement `Deref` or `DerefMut`.
///
/// - Both `Buffer` and `BufferRef` implement `std::fmt::Debug`,
///   sending an Objective-C `debugDescription` message to the
///   underlying `MTLBuffer`.
///
/// Following the `foreign_types` crate's nomenclature, the `Ref`
/// suffix indicates that `BufferRef` and `ResourceRef` are non-owning
/// types, used *by reference*, like `&BufferRef` or `&ResourceRef`.
/// These types are not, themselves, references.
macro_rules! foreign_obj_type {
    {
        type CType = $raw_ident:ident;
        pub struct $owned_ident:ident;
        type ParentType = $parent_ident:ident;
    } => {
        foreign_obj_type! {
            type CType = $raw_ident;
            pub struct $owned_ident;
        }

        impl ::std::ops::Deref for paste!{[<$owned_ident Ref>]} {
            type Target = paste!{[<$parent_ident Ref>]};

            #[inline]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self as *const Self::Target)  }
            }
        }

        impl ::std::convert::From<$owned_ident> for $parent_ident {
            fn from(item: $owned_ident) -> Self {
                unsafe { Self::from_ptr(::std::mem::transmute(item.into_ptr())) }
            }
        }
    };
    {
        type CType = $raw_ident:ident;
        pub struct $owned_ident:ident;
    } => {
        foreign_type! {
            pub unsafe type $owned_ident: Sync + Send {
                type CType = $raw_ident;
                fn drop = obj_drop;
                fn clone = obj_clone;
            }
        }

        unsafe impl ::objc::Message for $raw_ident {
        }
        unsafe impl ::objc::Message for paste!{[<$owned_ident Ref>]} {
        }

        impl ::std::fmt::Debug for paste!{[<$owned_ident Ref>]} {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                unsafe {
                    let string: *mut ::objc::runtime::Object = msg_send![self, debugDescription];
                    write!(f, "{}", nsstring_as_str(&*string))
                }
            }
        }

        impl ::std::fmt::Debug for $owned_ident {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::ops::Deref::deref(self).fmt(f)
            }
        }
    };
}

#[allow(non_upper_case_globals)]
/// A common bit for all floating point data types.
const MPSDataTypeFloatBit: isize = 0x10000000;
#[allow(non_upper_case_globals)]
const MPSDataTypeSignedBit: isize = 0x20000000;
#[allow(non_upper_case_globals)]
const MPSDataTypeNormalizedBit: isize = 0x40000000;

/// See <https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype>
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum MPSDataType {
    Invalid = 0,

    Float32 = MPSDataTypeFloatBit | 32,
    Float16 = MPSDataTypeFloatBit | 16,

    // Signed integers.
    Int8 = MPSDataTypeSignedBit | 8,
    Int16 = MPSDataTypeSignedBit | 16,
    Int32 = MPSDataTypeSignedBit | 32,

    // Unsigned integers. Range: [0, UTYPE_MAX]
    UInt8 = 8,
    UInt16 = 16,
    UInt32 = 32,

    // Unsigned normalized. Range: [0, 1.0]
    Unorm1 = MPSDataTypeNormalizedBit | 1,
    Unorm8 = MPSDataTypeNormalizedBit | 8,
}

pub enum MPSMatrixDescriptor {}

foreign_obj_type! {
    type CType = MPSMatrixDescriptor;
    pub struct MatrixDescriptor;
}

impl MatrixDescriptor {
    /// Build a MPS matrix descriptor
    /// - rows: The number of rows in the matrix.
    /// - columns: The number of columns in the matrix.
    /// - row_bytes: The stride, in bytes, between corresponding elements of consecutive rows in the matrix.
    /// - data_type: The type of the data to be stored in the matrix.
    pub fn new(
        rows: NSUInteger,
        columns: NSUInteger,
        row_bytes: NSUInteger,
        data_type: MPSDataType,
    ) -> Self {
        unsafe {
            msg_send![class!(MPSMatrixDescriptor), matrixDescriptorWithRows:rows
                                        columns:columns
                                        rowBytes: row_bytes
                                        dataType: data_type]
        }
    }

    pub fn row_bytes_for_colums(columns: NSUInteger, data_type: MPSDataType) -> usize {
        unsafe {
            msg_send![class!(MPSMatrixDescriptor), rowBytesForColumns: columns
										dataType: data_type]
        }
    }
}

pub enum MPSMatrixMultiplication {}

foreign_obj_type! {
    type CType = MPSMatrixMultiplication;
    pub struct MatrixMultiplication;
    type ParentType = Kernel;
}

impl MatrixMultiplication {
    /// Initializes a matrix multiplication kernel.
    ///
    /// - device: The device on which the matrix multiplication kernel will run.
    ///
    /// - transpose_left: A boolean value that indicates if the left input matrix
    ///   should be used in its transposed form. If the value is YES, then op(A) = A**T; otherwise, op(A) = A.
    ///
    /// - transpose_right: A boolean value that indicates if the right input matrix
    ///   should be used in its transposed form. If the value is YES, then op(B) = B**T; otherwise, op(B) = B.
    ///
    /// - result_rows: The number of rows in the result matrix (M in the BLAS GEMM description).
    ///
    /// - result_columns: The number of columns in the result matrix (N in the BLAS GEMM description).
    ///
    /// - interior_columns: The number of columns of the left input matrix after the appropriate transpose operation has been applied (K in the BLAS GEMM description).
    ///
    /// - alpha: The scale factor to apply to the product, specified in double precision. This value will be converted to the appropriate precision in the implementation itself, subject to rounding and/or clamping as necessary.
    ///
    /// - beta: The scale factor to apply to the initial values of C, specified in double precision. This value will be converted to the appropriate precision in the implementation itself, subject to rounding and/or clamping as necessary.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Device,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: NSUInteger,
        result_columns: NSUInteger,
        interior_columns: NSUInteger,
        alpha: f64,
        beta: f64,
    ) -> Option<Self> {
        unsafe {
            let instance: MatrixMultiplication = msg_send![class!(MPSMatrixMultiplication), alloc];
            let ptr: *mut Object = msg_send![instance.as_ref(), initWithDevice: device
										               transposeLeft: transpose_left
										               transposeRight: transpose_right
										               resultRows: result_rows
										               resultColumns: result_columns
										               interiorColumns: interior_columns
										               alpha: alpha
										               beta: beta];
            if ptr.is_null() {
                None
            } else {
                Some(instance)
            }
        }
    }
}

impl MatrixMultiplicationRef {
    /// Encodes a matrix multiplication kernel to a command buffer.
    ///
    /// - commandBuffer: The command buffer that will receive the encoded kernel.
    ///
    /// - left: The left input matrix.
    ///
    /// - right: The right input matrix.
    ///
    /// - result: The addend matrix which will also be overwritten by the operation result.
    ///
    pub fn encode(
        &self,
        command_buffer: TCommandBuffer,
        left: Matrix,
        right: Matrix,
        result: Matrix,
    ) {
        let cmd_buffer: CommandBuffer = command_buffer.as_ref().to_owned();
        unsafe {
            msg_send![self, encodeToCommandBuffer: cmd_buffer
                                  leftMatrix: left
                                  rightMatrix: right
                                  resultMatrix: result]
        }
    }
}

pub enum MPSMatrixVectorMultiplication {}

foreign_obj_type! {
    type CType = MPSMatrixVectorMultiplication;
    pub struct MatrixVectorMultiplication;
    type ParentType = Kernel;
}

impl MatrixVectorMultiplication {
    /// Initializes a matrix multiplication kernel.
    ///
    /// - device: The device on which the matrix vector multiplication kernel will run.
    ///
    /// - transpose: A boolean value.
    ///
    /// - rows: The number of rows.
    ///
    /// - columns: The number of columns.
    ///
    /// - alpha: The scale factor to apply to the product, specified in double precision. This value will be converted to the appropriate precision in the implementation itself, subject to rounding and/or clamping as necessary.
    ///
    /// - beta: The scale factor to apply to the initial values of C, specified in double precision. This value will be converted to the appropriate precision in the implementation itself, subject to rounding and/or clamping as necessary.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Device,
        transpose: bool,
        rows: NSUInteger,
        columns: NSUInteger,
        alpha: f64,
        beta: f64,
    ) -> Option<Self> {
        unsafe {
            let instance: MatrixVectorMultiplication =
                msg_send![class!(MPSMatrixVectorMultiplication), alloc];
            let ptr: *mut Object = msg_send![instance.as_ref(), initWithDevice: device
                                                       transpose: transpose
                                                       rows: rows
                                                       columns: columns
                                                       alpha: alpha
                                                       beta: beta];
            if ptr.is_null() {
                None
            } else {
                Some(instance)
            }
        }
    }
}

impl MatrixVectorMultiplicationRef {
    /// Encodes a matrix multiplication kernel to a command buffer.
    ///
    /// - commandBuffer: The command buffer that will receive the encoded kernel.
    ///
    /// - left: The left input matrix.
    ///
    /// - right: The right input matrix.
    ///
    /// - result: The addend matrix which will also be overwritten by the operation result.
    ///
    pub fn encode(
        &self,
        command_buffer: TCommandBuffer,
        left: Matrix,
        right: Vector,
        result: Vector,
    ) {
        let cmd_buffer: CommandBuffer = command_buffer.as_ref().to_owned();
        unsafe {
            msg_send![self, encodeToCommandBuffer: cmd_buffer
                                  inputMatrix: left
                                  inputVector: right
                                  resultVector: result]
        }
    }
}

pub enum MPSVectorDescriptor {}

foreign_obj_type! {
    type CType = MPSVectorDescriptor;
    pub struct VectorDescriptor;
}

impl VectorDescriptor {
    /// Build a MPS vector descriptor
    /// - lenght: The length of the vector.
    /// - data_type: The type of the data to be stored in the matrix.
    pub fn new(length: NSUInteger, data_type: MPSDataType) -> Self {
        unsafe {
            msg_send![class!(MPSVectorDescriptor), vectorDescriptorWithLength:length
                                        dataType: data_type]
        }
    }
}

pub enum MPSVector {}

foreign_obj_type! {
    type CType = MPSVector;
    pub struct Vector;
}

impl Vector {
    pub fn new_with_descriptor(
        buffer: Buffer,
        offset: NSUInteger,
        descriptor: VectorDescriptor,
    ) -> Option<Self> {
        unsafe {
            let instance: Vector = msg_send![class!(MPSVector), alloc];
            let ptr: *mut Object = msg_send![instance.as_ref(), initWithBuffer: buffer
                                                       offset: offset
                                                       descriptor: descriptor];
            if ptr.is_null() {
                None
            } else {
                Some(instance)
            }
        }
    }

    pub fn new(
        buffer: Buffer,
        offset: NSUInteger,
        data_type: MPSDataType,
        length: NSUInteger,
    ) -> Option<Self> {
        let descriptor = VectorDescriptor::new(length, data_type);
        Self::new_with_descriptor(buffer, offset, descriptor)
    }
}

pub enum MPSMatrix {}

foreign_obj_type! {
    type CType = MPSMatrix;
    pub struct Matrix;
}

impl Matrix {
    pub fn new_with_descriptor(
        buffer: Buffer,
        offset: NSUInteger,
        descriptor: MatrixDescriptor,
    ) -> Option<Self> {
        unsafe {
            let instance: Matrix = msg_send![class!(MPSMatrix), alloc];
            let ptr: *mut Object = msg_send![instance.as_ref(), initWithBuffer: buffer
                                                       offset: offset
                                                       descriptor: descriptor];
            if ptr.is_null() {
                None
            } else {
                Some(instance)
            }
        }
    }

    pub fn new(
        buffer: Buffer,
        offset: NSUInteger,
        data_type: MPSDataType,
        rows: NSUInteger,
        columns: NSUInteger,
    ) -> Option<Self> {
        // The API suggest we use Self::row_bytes_for_colums(columns, data_type) but an error is raised
        // depending on the size of the matrix.
        let row_bytes = match data_type {
            MPSDataType::Float32 => 4 * columns,
            MPSDataType::Float16 => 2 * columns,
            _ => return None,
        };
        let descriptor = MatrixDescriptor::new(rows, columns, row_bytes as _, data_type);
        Self::new_with_descriptor(buffer, offset, descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{anyhow, Result};

    #[test]
    fn test_mps_matrix_descriptor() -> Result<()> {
        let columns = 11;
        assert_eq!(48, MatrixDescriptor::row_bytes_for_colums(columns, MPSDataType::Float32));
        Ok(())
    }

    #[test]
    fn test_mps_matrix_multiplication() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let lhs = vec![1f32; 4 * 4];
                let lhs_buffer = context.buffer_from_slice(&lhs);

                let lhs_matrix = Matrix::new(lhs_buffer, 0, MPSDataType::Float32, 4, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating LHS matrix"))?;

                let rhs = vec![1f32; 4 * 4];
                let rhs_buffer = context.buffer_from_slice(&rhs);

                let rhs_matrix = Matrix::new(rhs_buffer, 0, MPSDataType::Float32, 4, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating RHS matrix"))?;

                let output = vec![0f32; 4 * 4];
                let output_buffer = context.buffer_from_slice(&output);

                let output_matrix = Matrix::new(output_buffer, 0, MPSDataType::Float32, 4, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating output matrix"))?;

                let matmul = MatrixMultiplication::new(
                    context.device().to_owned(),
                    false,
                    false,
                    4,
                    4,
                    4,
                    1.0,
                    0.0,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MatrixMultiplication"))?;

                let cmd_buffer = context.command_buffer();

                matmul.encode(
                    cmd_buffer,
                    lhs_matrix.to_owned(),
                    rhs_matrix.to_owned(),
                    output_matrix.to_owned(),
                );

                context.wait_until_completed()?;

                assert_eq!(output.as_slice(), &vec![4.0; 4 * 4]);
                Ok(())
            })
        })
    }

    #[test]
    fn test_mps_matrix_vector_multiplication() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let lhs = vec![1f32; 4 * 4];
                let lhs_buffer = context.buffer_from_slice(&lhs);

                let lhs_matrix = Matrix::new(lhs_buffer, 0, MPSDataType::Float32, 4, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating LHS matrix"))?;

                let rhs = vec![1f32; 4];
                let rhs_buffer = context.buffer_from_slice(&rhs);

                let rhs_vector = Vector::new(rhs_buffer, 0, MPSDataType::Float32, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating RHS vector"))?;

                let output = vec![0f32; 4 * 1];
                let output_buffer = context.buffer_from_slice(&output);

                let output_vector = Vector::new(output_buffer, 0, MPSDataType::Float32, 4)
                    .ok_or_else(|| anyhow!("An error occured when creating output vector"))?;

                let matmul = MatrixVectorMultiplication::new(
                    context.device().to_owned(),
                    false,
                    4,
                    4,
                    1.0,
                    0.0,
                )
                .ok_or_else(|| anyhow!("An error occured when creating MatrixMultiplication"))?;

                let cmd_buffer = context.command_buffer();

                matmul.encode(
                    cmd_buffer,
                    lhs_matrix.to_owned(),
                    rhs_vector.to_owned(),
                    output_vector.to_owned(),
                );

                context.wait_until_completed()?;

                assert_eq!(output.as_slice(), &vec![4.0; 4]);
                Ok(())
            })
        })
    }
}
