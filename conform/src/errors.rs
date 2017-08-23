error_chain! {
    foreign_links {
        NdArrayShape(::ndarray::ShapeError);
        Io(::std::io::Error);
        Reqwest(::reqwest::Error);
        TfDeploy(::tfdeploy::Error);
    }

    errors {
        TFString
    }
}

impl ::std::convert::From<::tensorflow::Status> for Error {
    fn from(tfs: ::tensorflow::Status) -> Error {
        format!("Tensorflow error: {:?}", tfs).into()
    }
}
