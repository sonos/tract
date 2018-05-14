#[macro_use]
extern crate bencher;
extern crate dinghy_test;
extern crate tfdeploy;
extern crate inceptionv3;

use self::dinghy_test::test_project_path;

use std::path;

const HOPPER: &str = "grace_hopper.jpg";
pub fn hopper() -> path::PathBuf {
    test_project_path().join(HOPPER)
}

#[cfg(feature = "tensorflow")]
fn dummy(_bencher: &mut bencher::Bencher) {
    ::tfdeploy::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
}

#[cfg(feature = "tensorflow")]
fn tf(bencher: &mut bencher::Bencher) {
    let mut tf = ::tfdeploy::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(inceptionv3::hopper());
    bencher.iter(|| {
        tf.run(
            vec![("input", input.clone())],
            "InceptionV3/Predictions/Reshape_1",
        ).unwrap()
    });
}

fn tfd(bencher: &mut bencher::Bencher) {
    let tfd = ::tfdeploy::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(hopper());
    let input_id = tfd.node_id_by_name("input").unwrap();
    let output_id = tfd.node_id_by_name("InceptionV3/Predictions/Reshape_1")
        .unwrap();
    bencher.iter(|| {
        tfd.run(vec![(input_id, input.clone())], output_id).unwrap();
    });
}

#[cfg(feature = "tensorflow")]
benchmark_group!(benches, dummy, tf, tfd);
#[cfg(not(feature = "tensorflow"))]
benchmark_group!(benches, tfd);
benchmark_main!(benches);

