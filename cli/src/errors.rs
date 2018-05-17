#[cfg(feature="tensorflow")]
extern crate conform;
extern crate tfdeploy;


/// Configures error handling for this crate.
error_chain! {
    links {
        Conform(conform::Error, conform::ErrorKind) #[cfg(feature="tensorflow")];
        Tfdeploy(tfdeploy::Error, tfdeploy::ErrorKind);
    }

    foreign_links {
        Io(::std::io::Error);
        NumParseInt(::std::num::ParseIntError);
    }
}