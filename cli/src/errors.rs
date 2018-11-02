#[cfg(feature = "tensorflow")]
use conform;
use ndarray;
use tract_core;

/// Configures error handling for this crate.
error_chain! {
    types {
        CliError, CliErrorKind, CliResultExt, CliResult;
    }
    links {
        Conform(conform::Error, conform::ErrorKind) #[cfg(feature="tensorflow")];
        Tract(tract_core::TractError, tract_core::TractErrorKind);
    }

    foreign_links {
        Io(::std::io::Error);
        NumParseInt(::std::num::ParseIntError);
        NdarrayShape(ndarray::ShapeError);
    }
}
