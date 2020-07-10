/// # Operators on array and shapes
mod broadcast;
pub(crate) mod concat;
//mod flatten;
mod gather;
mod pad;
mod reshape;
mod shape;
mod size;
mod slice;
mod tile;

pub use self::broadcast::MultiBroadcastTo;
pub use self::concat::{ConcatSlice, TypedConcat};
//pub use self::flatten::Flatten;
pub use self::gather::Gather;
pub use self::pad::{Pad, PadMode, PulsePad};
pub use self::reshape::FiniteReshape;
pub use self::shape::Shape;
pub use self::size::Size;
pub use self::slice::Slice;
pub use self::tile::Tile;
