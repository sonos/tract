use tract_hir::internal::*;

pub fn parse_costs(spec: &str) -> TVec<(Cost, usize)> {
    spec.split(",")
        .map(|spec| {
            let mut toks = spec.split("=");
            let name = toks.next().unwrap();
            let n = toks.next().unwrap().parse::<usize>().unwrap();
            let c = match name {
                "FMA(F32)" => Cost::FMA(f32::datum_type()),
                "Div(F32)" => Cost::Div(f32::datum_type()),
                "Buffer(F32)" => Cost::Buffer(f32::datum_type()),
                _ => panic!("Unknown cost specifier {}", name),
            };
            (c, n)
        })
        .collect()
}
