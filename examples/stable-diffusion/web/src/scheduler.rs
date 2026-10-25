//! Samplers over SD 1.5's scaled-linear noise schedule, and the seeded noise the latents start
//! from.
//!
//! Both work in the "k-diffusion" convention: the latents are `x = x0 + sigma * eps`, they start
//! at `sigma_max * noise`, and the UNet sees them scaled by `1 / sqrt(sigma^2 + 1)`.

/// How the latents move from one noise level to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// Euler over evenly spaced timesteps, as diffusers' `EulerDiscreteScheduler` with
    /// `timestep_spacing="linspace"`.
    Euler,
    /// DPM-Solver++(2M) over Karras-spaced sigmas, as diffusers' `DPMSolverMultistepScheduler`
    /// with `use_karras_sigmas=True`: second order from the second step on, first order on the
    /// last, which lands on sigma zero.
    DpmPp2mKarras,
}

impl Sampler {
    pub fn parse(name: &str) -> Option<Sampler> {
        match name {
            "euler" => Some(Sampler::Euler),
            "dpm++2m-karras" => Some(Sampler::DpmPp2mKarras),
            _ => None,
        }
    }
}

const TRAIN_STEPS: usize = 1000;

/// The noise level of each training timestep.
fn train_sigmas() -> Vec<f64> {
    let (lo, hi) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let mut cumprod = 1.0f64;
    (0..TRAIN_STEPS)
        .map(|i| {
            let beta = (lo + (hi - lo) * i as f64 / (TRAIN_STEPS - 1) as f64).powi(2);
            cumprod *= 1.0 - beta;
            ((1.0 - cumprod) / cumprod).sqrt()
        })
        .collect()
}

/// The (fractional) training timestep of noise level `sigma`, interpolating in log sigma.
fn sigma_to_t(sigma: f64, log_sigmas: &[f64]) -> f64 {
    let ls = sigma.ln();
    let low = log_sigmas.iter().take_while(|&&l| ls - l >= 0.0).count().saturating_sub(1);
    let low = low.min(log_sigmas.len() - 2);
    let (l, h) = (log_sigmas[low], log_sigmas[low + 1]);
    let w = ((l - ls) / (l - h)).clamp(0.0, 1.0);
    (1.0 - w) * low as f64 + w * (low + 1) as f64
}

pub struct Scheduler {
    pub sampler: Sampler,
    pub timesteps: Vec<i64>,
    /// One per step, then the final level (zero).
    pub sigmas: Vec<f32>,
    previous_x0: Option<Vec<f32>>,
}

impl Scheduler {
    pub fn new(sampler: Sampler, steps: usize) -> Scheduler {
        let train = train_sigmas();
        let (timesteps, mut sigmas): (Vec<i64>, Vec<f32>) = match sampler {
            Sampler::Euler => (0..steps)
                .map(|i| {
                    let t = (TRAIN_STEPS - 1) as f64 * (1.0 - i as f64 / (steps - 1).max(1) as f64);
                    let t = t.round() as i64;
                    (t, train[t as usize] as f32)
                })
                .unzip(),
            Sampler::DpmPp2mKarras => {
                let rho = 7.0;
                let (min, max) = (train[0].powf(1.0 / rho), train[TRAIN_STEPS - 1].powf(1.0 / rho));
                let log_sigmas: Vec<f64> = train.iter().map(|s| s.ln()).collect();
                (0..steps)
                    .map(|i| {
                        let ramp = i as f64 / (steps - 1).max(1) as f64;
                        let sigma = (max + ramp * (min - max)).powf(rho);
                        (sigma_to_t(sigma, &log_sigmas).round() as i64, sigma as f32)
                    })
                    .unzip()
            }
        };
        sigmas.push(0.0);
        Scheduler { sampler, timesteps, sigmas, previous_x0: None }
    }

    pub fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    /// The factor the latents are scaled by before step `step`'s UNet call.
    pub fn input_scale(&self, step: usize) -> f32 {
        let sigma = self.sigmas[step];
        1.0 / (sigma * sigma + 1.0).sqrt()
    }

    /// Moves `latents` from step `step`'s noise level to the next, given the guided noise
    /// prediction `eps`.
    pub fn step(&mut self, step: usize, latents: &mut [f32], eps: &[f32]) {
        let (sigma, next) = (self.sigmas[step], self.sigmas[step + 1]);
        match self.sampler {
            Sampler::Euler => {
                for (x, e) in latents.iter_mut().zip(eps) {
                    *x += e * (next - sigma);
                }
            }
            Sampler::DpmPp2mKarras => {
                let x0: Vec<f32> = latents.iter().zip(eps).map(|(x, e)| x - sigma * e).collect();
                if next == 0.0 {
                    latents.copy_from_slice(&x0);
                } else {
                    let (t, t_next) = (-(sigma as f64).ln(), -(next as f64).ln());
                    let h = t_next - t;
                    let ratio = next / sigma;
                    let decay = (-(-h).exp_m1()) as f32;
                    match &self.previous_x0 {
                        Some(old) => {
                            let r = (t - -(self.sigmas[step - 1] as f64).ln()) / h;
                            let (a, b) = ((1.0 + 1.0 / (2.0 * r)) as f32, (1.0 / (2.0 * r)) as f32);
                            for ((x, d), o) in latents.iter_mut().zip(&x0).zip(old) {
                                *x = ratio * *x + decay * (a * d - b * o);
                            }
                        }
                        None => {
                            for (x, d) in latents.iter_mut().zip(&x0) {
                                *x = ratio * *x + decay * d;
                            }
                        }
                    }
                }
                self.previous_x0 = Some(x0);
            }
        }
    }
}

/// Standard normal samples from a seed: SplitMix64 through Box-Muller.
pub struct Noise(u64);

impl Noise {
    pub fn new(seed: u64) -> Noise {
        Noise(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn samples(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let (u1, u2) = (self.uniform(), self.uniform());
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}
