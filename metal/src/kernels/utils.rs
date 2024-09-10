pub fn build_metal_size_for_shape(shape: &[usize]) -> metal::MTLSize {
    match shape.len() {
        0 => panic!("Unexpected empty shape while build grid size"),
        1 => metal::MTLSize { width: shape[0] as _, height: 1, depth: 1 },
        2 => metal::MTLSize { width: shape[1] as _, height: shape[0] as _, depth: 1 },
        3.. => metal::MTLSize {
            width: shape[shape.len() - 1] as _,
            height: shape[shape.len() - 2] as _,
            depth: (shape[..shape.len() - 2].iter().product::<usize>()) as _,
        },
    }
}

pub fn build_metal_size_with_ones() -> metal::MTLSize {
    metal::MTLSize { width: 1, height: 1, depth: 1 }
}
