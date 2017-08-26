
error_chain!{
    foreign_links {
        Image(::image::ImageError);
        Io(::std::io::Error);
        NdarrayShape(::ndarray::ShapeError);
        Protobuf(::protobuf::ProtobufError);
        StrUtf8(::std::str::Utf8Error);
    }
    errors {
        TFString {}
    }
}

#[cfg(feature = "tensorflow")]
impl ::std::convert::From<::tensorflow::Status> for Error {
    fn from(tfs: ::tensorflow::Status) -> Error {
        format!("Tensorflow error: {:?}", tfs).into()
    }
}
