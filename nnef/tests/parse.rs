

#[test]
fn parse_alexnet() {
    let content = std::fs::read_to_string("tests/alexnet.nnef").unwrap();
    tract_nnef::parser::parse_document(&content).unwrap();
}

#[test]
fn parse_stdlib() {
    let content = std::fs::read_to_string("tests/stdlib.nnef").unwrap();
    tract_nnef::parser::parse_fragments(&content).unwrap();
}
