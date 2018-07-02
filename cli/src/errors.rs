#[cfg(feature = "tensorflow")]
use conform;
use tfdeploy;
use bincode;
use serde_json;
use ndarray;

/// Configures error handling for this crate.
error_chain! {
    links {
        Conform(conform::Error, conform::ErrorKind) #[cfg(feature="tensorflow")];
        Tfdeploy(tfdeploy::Error, tfdeploy::ErrorKind);
    }

    foreign_links {
        Io(::std::io::Error);
        NumParseInt(::std::num::ParseIntError);
        Bincode(bincode::Error);
        SerdeJson(serde_json::Error);
        NdarrayShape(ndarray::ShapeError);
    }
}
