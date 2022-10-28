use crate::TractResult;
use tract_hir::internal::*;

pub fn parse_costs(spec: &str) -> TractResult<Vec<(Cost, usize)>> {
    spec.split(',')
        .map(|spec| {
            let mut toks = spec.split('=');
            let name = toks.next().unwrap();
            let n = toks.next().unwrap().parse::<usize>().unwrap();
            let c = match name {
                "FMA(F32)" => Cost::FMA(f32::datum_type()),
                "Div(F32)" => Cost::Div(f32::datum_type()),
                "Buffer(F32)" => Cost::Buffer(f32::datum_type()),
                "Params(F32)" => Cost::Params(f32::datum_type()),
                _ => bail!("Unknown cost specifier {}", name),
            };
            Ok((c, n))
        })
        .collect()
}
