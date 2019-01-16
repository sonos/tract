//! error_chain generated types

error_chain! {
    types {
        TractError, TractErrorKind, TractResultExt, TractResult;
    }
    foreign_links {
        Image(::image::ImageError) #[cfg(features="image_ops")];
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        StrUtf8(::std::str::Utf8Error);
        NumParseInt(::std::num::ParseIntError);
    }
    errors {
        TFString {}
    }
}
