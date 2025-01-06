// from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/random/philox_random.h

use tract_hir::internal::*;

#[derive(Copy, Clone)]
pub struct Philox4x32x10 {
    key: u64,
    counter: u128,
}

fn mul_hilo(a: u32, b: u32) -> (u32, u32) {
    ((((a as u64) * (b as u64)) >> 32) as u32, ((a as u64) * (b as u64)) as u32)
}

#[allow(non_upper_case_globals)]
impl Philox4x32x10 {
    pub fn weird_tf_constructor(seed_lo: u64, seed_hi: u64) -> Philox4x32x10 {
        let mut ph = Self::for_seed(seed_lo);
        ph.skip_fast((seed_hi as u128) << 64);
        ph
    }

    #[allow(unused)]
    pub fn for_seeds(seed1: u32, seed2: u32) -> Philox4x32x10 {
        Self::for_seed(((seed2 as u64) << 32) | seed1 as u64)
    }

    pub fn for_seed(seed: u64) -> Philox4x32x10 {
        Philox4x32x10 { key: seed, counter: 0 }
    }

    pub fn skip_fast(&mut self, n: u128) {
        self.counter = self.counter.wrapping_add(n);
    }

    #[allow(unused)]
    pub fn next_as_u32s(&mut self) -> [u32; 4] {
        let v = self.next();
        [v as u32, (v >> 32) as u32, (v >> 64) as u32, (v >> 96) as u32]
    }

    pub fn next(&mut self) -> u128 {
        let mut key = self.key;
        let mut counter = self.counter;

        // 0
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 1
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 2
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 3
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 4
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 5
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 6
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 7
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 8
        Self::compute_one(&mut counter, key);
        Self::raise_key(&mut key);
        // 9
        Self::compute_one(&mut counter, key);

        self.counter = self.counter.wrapping_add(1);
        counter
    }

    fn raise_key(key: &mut u64) {
        const kPhiloxW32A: u32 = 0x9E3779B9;
        const kPhiloxW32B: u32 = 0xBB67AE85;

        let k0 = *key as u32;
        let k1 = (*key >> 32) as u32;
        let k0 = k0.wrapping_add(kPhiloxW32A) as u64;
        let k1 = k1.wrapping_add(kPhiloxW32B) as u64;

        *key = (k1 << 32) | k0;
    }

    fn compute_one(counter: &mut u128, key: u64) {
        const kPhiloxM4x32A: u32 = 0xD2511F53;
        const kPhiloxM4x32B: u32 = 0xCD9E8D57;

        let c0 = *counter as u32;
        let c1 = (*counter >> 32) as u32;
        let c2 = (*counter >> 64) as u32;
        let c3 = (*counter >> 96) as u32;

        let (hi0, lo0) = mul_hilo(kPhiloxM4x32A, c0);
        let (hi1, lo1) = mul_hilo(kPhiloxM4x32B, c2);

        let r0 = (hi1 ^ c1 ^ (key as u32)) as u128;
        let r1 = lo1 as u128;
        let r2 = (hi0 ^ c3 ^ ((key >> 32) as u32)) as u128;
        let r3 = lo0 as u128;

        *counter = (r3 << 96) | (r2 << 64) | (r1 << 32) | r0
    }

    pub fn u32_iter(self) -> impl Iterator<Item = u32> {
        self.flat_map(|big| {
            tvec![big as u32, (big >> 32) as u32, (big >> 64) as u32, (big >> 96) as u32]
                .into_iter()
        })
    }
}

impl Iterator for Philox4x32x10 {
    type Item = u128;
    fn next(&mut self) -> Option<u128> {
        Some(Philox4x32x10::next(self))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // checked against https://github.com/dominikwerder/philox
    // https://github.com/dominikwerder/philox/blob/master/src/test.rs#L62
    #[test]
    fn seed() {
        let mut ph = Philox4x32x10::for_seeds(1, 2);
        assert_eq!(ph.next_as_u32s(), [0x598de3a, 0x98d2802e, 0x270f8f9e, 0xeab709d3]);
    }

    #[test]
    fn zeros() {
        let mut ph = Philox4x32x10::for_seeds(0, 0);
        assert_eq!(ph.next_as_u32s(), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
    }

    #[test]
    fn ffff() {
        let mut ph = Philox4x32x10::for_seeds(0xffffffff, 0xffffffff);
        ph.skip_fast(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff);
        assert_eq!(ph.next_as_u32s(), [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
    }

    #[test]
    fn x243f6a88() {
        let mut ph = Philox4x32x10::for_seeds(0xa4093822, 0x299f31d0);
        ph.skip_fast(0x0370_7344_1319_8a2e_85a3_08d3_243f_6a88);
        assert_eq!(ph.next_as_u32s(), [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]);
    }
}
