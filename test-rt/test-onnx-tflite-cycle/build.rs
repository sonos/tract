fn main() {
    let mut suite = suite_onnx::suite().clone();
    suite.ignore(&ignore);
    suite.test_runtime("tflite_cycle", "suite_onnx::suite()", "tflite_cycle()");
}

fn ignore(t: &[String]) -> bool {
    let name = t.last().unwrap();
    !name.contains("_conv_")
}
