use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::math::round_ties_to_even;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_grid_sample",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

#[derive(Clone, Debug)]
pub enum InterpolationMode {
    Bilinear,
    Nearest,
    Bicubic,
}

impl InterpolationMode {
    fn as_str(&self) -> &'static str {
        match self {
            InterpolationMode::Bilinear => "bilinear",
            InterpolationMode::Nearest => "nearest",
            InterpolationMode::Bicubic => "bicubic",
        }
    }

    fn from_str(s: &str) -> TractResult<Self> {
        Ok(match s {
            "bilinear" => InterpolationMode::Bilinear,
            "nearest" => InterpolationMode::Nearest,
            "bicubic" => InterpolationMode::Bicubic,
            _ => bail!("Unsupported GridSample mode: {}", s),
        })
    }
}

#[derive(Clone, Debug)]
pub enum PaddingMode {
    Zeros,
    Border,
    Reflection,
}

impl PaddingMode {
    fn as_str(&self) -> &'static str {
        match self {
            PaddingMode::Zeros => "zeros",
            PaddingMode::Border => "border",
            PaddingMode::Reflection => "reflection",
        }
    }

    fn from_str(s: &str) -> TractResult<Self> {
        Ok(match s {
            "zeros" => PaddingMode::Zeros,
            "border" => PaddingMode::Border,
            "reflection" => PaddingMode::Reflection,
            _ => bail!("Unsupported GridSample padding_mode: {}", s),
        })
    }
}

#[derive(Clone, Debug)]
pub struct GridSample {
    pub mode: InterpolationMode,
    pub padding_mode: PaddingMode,
    pub align_corners: bool,
}

impl GridSample {
    fn denormalize(&self, coord: f32, size: usize) -> f32 {
        if self.align_corners {
            (coord + 1.0) / 2.0 * (size as f32 - 1.0)
        } else {
            ((coord + 1.0) * size as f32 - 1.0) / 2.0
        }
    }

    fn bounds(&self, size: usize) -> (f32, f32) {
        if self.align_corners { (0.0, size as f32 - 1.0) } else { (-0.5, size as f32 - 0.5) }
    }

    fn pixel_at_nd(
        &self,
        x: &tract_ndarray::ArrayViewD<'_, f32>,
        batch: usize,
        channel: usize,
        coords: &[isize],
        spatial_sizes: &[usize],
    ) -> f32 {
        match self.padding_mode {
            PaddingMode::Zeros => {
                for (&c, &s) in coords.iter().zip(spatial_sizes.iter()) {
                    if c < 0 || c >= s as isize {
                        return 0.0;
                    }
                }
                let mut idx = vec![batch, channel];
                idx.extend(coords.iter().map(|&c| c as usize));
                x[idx.as_slice()]
            }
            PaddingMode::Border => {
                let mut idx = vec![batch, channel];
                for (&c, &s) in coords.iter().zip(spatial_sizes.iter()) {
                    idx.push((c.max(0) as usize).min(s - 1));
                }
                x[idx.as_slice()]
            }
            PaddingMode::Reflection => {
                let mut idx = vec![batch, channel];
                for (&c, &s) in coords.iter().zip(spatial_sizes.iter()) {
                    let (lo, hi) = self.bounds(s);
                    idx.push(gs_reflect(c as f32, lo, hi) as usize);
                }
                x[idx.as_slice()]
            }
        }
    }

    fn apply_padding(&self, coord: f32, lo: f32, hi: f32) -> f32 {
        match self.padding_mode {
            PaddingMode::Border => coord.clamp(0.0, hi + lo),
            PaddingMode::Reflection => gs_reflect(coord, lo, hi),
            PaddingMode::Zeros => coord,
        }
    }

    fn is_oob(&self, coords: &[f32], bounds: &[(f32, f32)]) -> bool {
        coords.iter().zip(bounds.iter()).any(|(&c, &(lo, hi))| c < lo || c > hi)
    }

    fn pad_coords(&self, coords: &mut [f32], bounds: &[(f32, f32)]) {
        for (c, &(lo, hi)) in coords.iter_mut().zip(bounds.iter()) {
            *c = self.apply_padding(*c, lo, hi);
        }
    }

    fn sample_nd(
        &self,
        x: &tract_ndarray::ArrayViewD<'_, f32>,
        batch: usize,
        channel: usize,
        pixel_coords: &[f32],
        spatial_sizes: &[usize],
    ) -> f32 {
        let ndim = pixel_coords.len();
        let bounds: Vec<(f32, f32)> = spatial_sizes.iter().map(|&s| self.bounds(s)).collect();

        match self.mode {
            InterpolationMode::Nearest => {
                let mut coords: Vec<f32> =
                    pixel_coords.iter().map(|&c| round_ties_to_even(c)).collect();
                if self.is_oob(&coords, &bounds) {
                    self.pad_coords(&mut coords, &bounds);
                }
                let icoords: Vec<isize> = coords.iter().map(|&c| c as isize).collect();
                self.pixel_at_nd(x, batch, channel, &icoords, spatial_sizes)
            }
            InterpolationMode::Bilinear => {
                let mut coords: Vec<f32> = pixel_coords.to_vec();
                if self.is_oob(&coords, &bounds) {
                    self.pad_coords(&mut coords, &bounds);
                }
                let num_corners = 1 << ndim;
                let mut result = 0.0f32;
                for corner in 0..num_corners {
                    let mut weight = 1.0f32;
                    let mut icoords = Vec::with_capacity(ndim);
                    for (d, &c) in coords.iter().enumerate() {
                        let lo = c.floor() as isize;
                        if (corner >> d) & 1 == 0 {
                            icoords.push(lo);
                            weight *= (lo + 1) as f32 - c;
                        } else {
                            icoords.push(lo + 1);
                            weight *= c - lo as f32;
                        }
                    }
                    result += weight * self.pixel_at_nd(x, batch, channel, &icoords, spatial_sizes);
                }
                result
            }
            InterpolationMode::Bicubic => {
                assert!(ndim == 2, "Bicubic interpolation only supports 2D spatial dimensions");
                let (mut px, mut py) = (pixel_coords[0], pixel_coords[1]);
                if self.is_oob(&[px, py], &bounds) {
                    px = self.apply_padding(px, bounds[0].0, bounds[0].1);
                    py = self.apply_padding(py, bounds[1].0, bounds[1].1);
                }
                let x0 = px.floor() as isize - 1;
                let y0 = py.floor() as isize - 1;
                let dx = px - x0 as f32 - 1.0;
                let dy = py - y0 as f32 - 1.0;

                let mut p = [[0.0f32; 4]; 4];
                for (h, row) in p.iter_mut().enumerate() {
                    for (w, val) in row.iter_mut().enumerate() {
                        *val = self.pixel_at_nd(
                            x,
                            batch,
                            channel,
                            &[x0 + w as isize, y0 + h as isize],
                            spatial_sizes,
                        );
                    }
                }
                bicubic_interpolate(&p, dx, dy)
            }
        }
    }
}

fn gs_reflect(x: f32, x_min: f32, x_max: f32) -> f32 {
    let rng = x_max - x_min;
    if rng == 0.0 {
        return x_min;
    }
    if x < x_min {
        let dx = x_min - x;
        let n = (dx / rng) as i32;
        let r = dx - n as f32 * rng;
        if n % 2 == 0 { x_min + r } else { x_max - r }
    } else if x > x_max {
        let dx = x - x_max;
        let n = (dx / rng) as i32;
        let r = dx - n as f32 * rng;
        if n % 2 == 0 { x_max - r } else { x_min + r }
    } else {
        x
    }
}

fn bicubic_interpolate(p: &[[f32; 4]; 4], dx: f32, dy: f32) -> f32 {
    let mut v = [0.0f32; 4];
    let mut coeffs = [0.0f32; 4];
    cubic_coeffs(dx, &mut coeffs);
    for i in 0..4 {
        v[i] =
            coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
    }
    cubic_coeffs(dy, &mut coeffs);
    coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]
}

fn cubic_coeffs(x: f32, coeffs: &mut [f32; 4]) {
    let a = -0.75f32;
    let xp1 = x + 1.0;
    let xm1 = 1.0 - x;
    let xm2 = 2.0 - x;
    coeffs[0] = ((a * xp1 - 5.0 * a) * xp1 + 8.0 * a) * xp1 - 4.0 * a;
    coeffs[1] = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    coeffs[2] = ((a + 2.0) * xm1 - (a + 3.0)) * xm1 * xm1 + 1.0;
    coeffs[3] = ((a * xm2 - 5.0 * a) * xm2 + 8.0 * a) * xm2 - 4.0 * a;
}

impl Op for GridSample {
    fn name(&self) -> StaticName {
        "GridSample".into()
    }

    op_as_typed_op!();
}

impl EvalOp for GridSample {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (x, grid) = args_2!(inputs);
        let input_dt = x.datum_type();
        let x_tensor = x.into_tensor();
        let x_cow = x_tensor.cast_to::<f32>()?;
        let x = x_cow.try_as_dense()?.to_array_view::<f32>()?;
        let grid_tensor = grid.into_tensor();
        let grid_cow = grid_tensor.cast_to::<f32>()?;
        let grid = grid_cow.try_as_dense()?.to_array_view::<f32>()?;

        let x_shape = x.shape();
        let grid_shape = grid.shape();
        let rank = x_shape.len();
        let spatial_rank = rank - 2;

        let n_batch = x_shape[0];
        let n_channel = x_shape[1];
        let spatial_sizes: Vec<usize> = x_shape[2..].to_vec();

        let mut output_shape = vec![n_batch, n_channel];
        output_shape.extend_from_slice(&grid_shape[1..rank - 1]);

        let output = tract_ndarray::ArrayD::from_shape_fn(&*output_shape, |idx| -> f32 {
            let batch = idx[0];
            let channel = idx[1];
            let out_spatial: Vec<usize> = (2..rank).map(|d| idx[d]).collect();

            let mut grid_idx = vec![batch];
            grid_idx.extend_from_slice(&out_spatial);
            grid_idx.push(0);

            let mut pixel_coords = Vec::with_capacity(spatial_rank);
            for (d, &size) in spatial_sizes.iter().enumerate() {
                *grid_idx.last_mut().unwrap() = spatial_rank - 1 - d;
                let norm_coord = grid[grid_idx.as_slice()];
                pixel_coords.push(self.denormalize(norm_coord, size));
            }

            self.sample_nd(&x, batch, channel, &pixel_coords, &spatial_sizes)
        });

        Ok(tvec!(output.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

impl TypedOp for GridSample {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let x_shape = &inputs[0].shape;
        let grid_shape = &inputs[1].shape;
        let rank = x_shape.len();

        let mut output_shape: TVec<TDim> = tvec![x_shape[0].clone(), x_shape[1].clone()];
        for d in 1..rank - 1 {
            output_shape.push(grid_shape[d].clone());
        }

        Ok(tvec!(inputs[0].datum_type.fact(&output_shape)))
    }
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("grid"),
        TypeName::String.named("mode").default("bilinear"),
        TypeName::String.named("padding_mode").default("zeros"),
        TypeName::Logical.named("align_corners").default(false),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &GridSample) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let grid = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_onnx_grid_sample",
        &[input, grid],
        &[
            ("mode", string(op.mode.as_str())),
            ("padding_mode", string(op.padding_mode.as_str())),
            ("align_corners", logical(op.align_corners)),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let grid = invocation.named_arg_as(builder, "grid")?;
    let mode: String = invocation.named_arg_as(builder, "mode")?;
    let padding_mode: String = invocation.named_arg_as(builder, "padding_mode")?;
    let align_corners: bool = invocation.named_arg_as(builder, "align_corners")?;
    let op = GridSample {
        mode: InterpolationMode::from_str(&mode)?,
        padding_mode: PaddingMode::from_str(&padding_mode)?,
        align_corners,
    };
    builder.wire(op, &[input, grid])
}
