#![cfg(feature = "tensorflow")]
extern crate colored;
extern crate dinghy_test;
extern crate flate2;
extern crate image;
extern crate itertools;
extern crate mio_httpc;
extern crate ndarray;
extern crate tar;
extern crate tfdeploy;

#[path = "../examples/inceptionv3.rs"]
mod inceptionv3;
mod utils;

#[test]
fn test_tf() {
    let mut tf = ::tfdeploy::tf::for_path(inceptionv3::inception_v3_2016_08_28_frozen()).unwrap();
    let input = inceptionv3::load_image(inceptionv3::hopper());
    let mut output = tf.run(vec![("input", input)], "InceptionV3/Predictions/Reshape_1")
        .unwrap();
    let labels = inceptionv3::load_labels();
    for (ix, c) in output.remove(0).take_f32s().unwrap().iter().enumerate() {
        if *c >= 0.01 {
            println!("{}: {} {}", ix, c, labels[ix]);
        }
    }
}

#[test]
fn test_compare_all() {
    utils::compare_all(
        inceptionv3::inception_v3_2016_08_28_frozen(),
        vec![("input", inceptionv3::load_image(inceptionv3::hopper()))],
        "InceptionV3/Predictions/Reshape_1",
    ).unwrap();
}
