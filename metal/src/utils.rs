pub fn div_ceil(m: usize, b: usize) -> metal::NSUInteger {
    ((m + b - 1) / b) as metal::NSUInteger
}
