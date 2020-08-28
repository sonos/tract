#![allow(deprecated)]

use tract_core::ndarray;
use crate::model::Model;

error_chain! {
    types {
        CliError, CliErrorKind, CliResultExt, CliResult;
    }

    links {
        Tract(tract_core::TractError, tract_core::TractErrorKind);
        TractTensorflowConform(tract_tensorflow::conform::Error, tract_tensorflow::conform::ErrorKind) #[cfg(feature="conform")];
    }

    foreign_links {
        Fmt(::std::fmt::Error);
        Io(::std::io::Error);
        NumParseInt(::std::num::ParseIntError);
        NdarrayShape(ndarray::ShapeError);
        NdarrayNpyReadNpz(ndarray_npy::ReadNpzError);
        SerdeJson(serde_json::error::Error);
        Infaillible(std::convert::Infallible);
    }

    errors {
        ModelBuilding(partial: Box<dyn Model>) {
        }
    }
}
