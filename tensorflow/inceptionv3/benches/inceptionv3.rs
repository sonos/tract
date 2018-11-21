#[macro_use]
extern crate criterion;
extern crate dinghy_test;
extern crate inceptionv3;
#[macro_use]
extern crate tract_core as tract;
extern crate tract_tensorflow;

use self::dinghy_test::test_project_path;
use criterion::Criterion;

use std::path;

const HOPPER: &str = "grace_hopper.jpg";
pub fn hopper() -> path::PathBuf {
    test_project_path().join(HOPPER)
}

#[cfg(feature = "conform")]
fn dummy(_bencher: &mut Criterion) {
    tract_tensorflow::conform::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
}

#[cfg(feature = "conform")]
fn tf(bencher: &mut Criterion) {
    let mut tf =
        tract_tensorflow::conform::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen())
            .unwrap();
    let input = inceptionv3::load_image(hopper());
    bencher.bench_function("tensorflow", move |b| {
        b.iter(|| {
            tf.run(
                vec![("input", input.clone())],
                "InceptionV3/Predictions/Reshape_1",
            ).unwrap()
        })
    });
}

fn tract(bencher: &mut Criterion) {
    let tfd = ::tract_tensorflow::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(hopper());
    let plan = tract::SimplePlan::new(tfd).unwrap();
    bencher.bench_function("tract", move |b| {
        b.iter(|| plan.run(tvec![input.clone()]).unwrap())
    });
}

pub fn benches() {
    let mut criterion: Criterion = Criterion::default().sample_size(3).configure_from_args();
    #[cfg(feature = "conform")]
    {
        dummy(&mut criterion);
        tf(&mut criterion);
    }
    tract(&mut criterion);
}
criterion_main!(benches);
