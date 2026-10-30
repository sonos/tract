#![cfg(feature = "unstable-autobatch")]
use tract::prelude::*;

#[test]
fn a_saturated_pool_fails_as_overloaded() -> anyhow::Result<()> {
    tract_data::knobs::set_config_global("TRACT_POOL_WORKERS", "1")?;
    tract_data::knobs::set_config_global("TRACT_POOL_QUEUE", "0")?;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../harness/nnef-test-cases/deploy-pool");
    let model = tract::nnef()?.load(path)?;
    let runnable = tract::runtime_for_name("deploy")?.prepare(model)?;
    let x = Tensor::from_slice(&[5, 4], &[0.5f32; 20])?;
    let scale = Tensor::from_slice(&[1, 3], &[1f32; 3])?;
    let overloaded = std::thread::scope(|scope| {
        let callers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    (0..1000).any(|_| match runnable.run([x.clone(), scale.clone()]) {
                        Err(e) => e.downcast_ref::<tract::Overloaded>().is_some(),
                        Ok(_) => false,
                    })
                })
            })
            .collect();
        callers.into_iter().map(|caller| caller.join().unwrap()).collect::<Vec<bool>>()
    });
    assert!(overloaded.iter().any(|o| *o));
    Ok(())
}
