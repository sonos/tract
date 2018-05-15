//! error_chain generated types

error_chain!{
    foreign_links {
        Image(::image::ImageError) #[cfg(features="image_ops")];
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
    errors {
        TFString {}
    }
}
