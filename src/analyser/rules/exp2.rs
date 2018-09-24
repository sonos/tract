use std::fmt;
use std::marker::PhantomData;
use TfdResult;

use analyser::prelude::*;
use analyser::rules::prelude::*;
use dim::TDim;
use num::Zero;
use std::ops::{Add, Div, Mul, Neg, Sub};
use tensor::{DatumType, Tensor};

