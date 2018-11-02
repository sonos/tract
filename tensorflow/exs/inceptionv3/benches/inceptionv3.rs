#[macro_use]
extern crate criterion;
#[cfg(feature = "tensorflow")]
extern crate conform;
extern crate dinghy_test;
extern crate inceptionv3;
#[macro_use]
extern crate tract;
extern crate tract_tensorflow;

use self::dinghy_test::test_project_path;
use criterion::Criterion;

use std::path;

const HOPPER: &str = "grace_hopper.jpg";
pub fn hopper() -> path::PathBuf {
    test_project_path().join(HOPPER)
}

#[cfg(feature = "tensorflow")]
fn dummy(_bencher: &mut Criterion) {
    ::conform::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
}

#[cfg(feature = "tensorflow")]
fn tf(bencher: &mut Criterion) {
    let mut tf = ::conform::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(hopper());
    bencher.bench_function("TF", move |b| {
        b.iter(|| {
            tf.run(
                vec![("input", input.clone())],
                "InceptionV3/Predictions/Reshape_1",
            ).unwrap()
        })
    });
}

fn tfd(bencher: &mut Criterion) {
    let tfd = ::tract_tensorflow::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(hopper());
    let plan = ::tract::SimplePlan::new(tfd).unwrap();
    bencher.bench_function("TFD", move |b| {
        b.iter(|| plan.run(tvec![input.clone()]).unwrap())
    });
}

pub fn benches() {
    let mut criterion: Criterion = Criterion::default().sample_size(3).configure_from_args();
    #[cfg(feature = "tensorflow")]
    {
        dummy(&mut criterion);
        tf(&mut criterion);
    }
    tfd(&mut criterion);
}
criterion_main!(benches);
