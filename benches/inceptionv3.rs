#[macro_use]
extern crate bencher;
extern crate dinghy_test;
extern crate image;
extern crate itertools;
extern crate flate2;
extern crate ndarray;
extern crate mio_httpc;
extern crate tar;
extern crate tfdeploy;

#[path = "../examples/inceptionv3.rs"]
mod inceptionv3;

#[cfg(features = "tensorflow")]
fn dummy(_bencher: &mut bencher::Bencher) {
    ::tfdeploy::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
}

#[cfg(features = "tensorflow")]
fn tf(bencher: &mut bencher::Bencher) {
    let mut tf = ::tfdeploy::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(inceptionv3::HOPPER);
    bencher.iter(|| {
        tf.run(
            vec![("input", input.clone())],
            "InceptionV3/Predictions/Reshape_1",
        ).unwrap()
    });
}

fn tfd(bencher: &mut bencher::Bencher) {
    let tfd = ::tfdeploy::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(inceptionv3::HOPPER);
    bencher.iter(|| {
        tfd.run(
            vec![("input", input.clone())],
            "InceptionV3/Predictions/Reshape_1",
        ).unwrap();
    });
}

#[cfg(features = "tensorflow")]
benchmark_group!(benches, dummy, tf, tfd);
#[cfg(not(features = "tensorflow"))]
benchmark_group!(benches, tfd);
benchmark_main!(benches);
