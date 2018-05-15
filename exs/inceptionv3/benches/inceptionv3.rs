#[macro_use]
extern crate criterion;
#[cfg(feature="tensorflow")]
extern crate conform;
extern crate dinghy_test;
extern crate tfdeploy;
extern crate inceptionv3;

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
    bencher.bench_function("TF",
        move |b| b.iter(||
            tf.run(
                vec![("input", input.clone())],
                "InceptionV3/Predictions/Reshape_1",
            ).unwrap()
    ));
}

fn tfd(bencher: &mut Criterion) {
    let tfd = ::tfdeploy::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(hopper());
    let input_id = tfd.node_id_by_name("input").unwrap();
    let output_id = tfd.node_id_by_name("InceptionV3/Predictions/Reshape_1")
        .unwrap();
    bencher.bench_function("TFD",
        move |b| b.iter(|| tfd.run(vec![(input_id, input.clone())], output_id).unwrap())
    );
}

pub fn benches() {
    let mut criterion: Criterion = Criterion::default().sample_size(10).configure_from_args();
    #[cfg(feature = "tensorflow")] {
        dummy(&mut criterion);
        tf(&mut criterion);
    }
    tfd(&mut criterion);
}
criterion_main!(benches);

