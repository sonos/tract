#[path="suite.rs"]
mod suite;

fn main() {
    suite::suite().test_runtime("tflite_cycle", "suite::suite()", "tflite_cycle()");
    suite::suite().test_runtime("tflite_predump", "suite::suite()", "tflite_predump()");
}

