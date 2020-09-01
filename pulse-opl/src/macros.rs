
#[macro_export]
macro_rules! op_pulse {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["pulse"]
        }
    };
}

