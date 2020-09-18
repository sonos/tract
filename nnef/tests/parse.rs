use tract_nnef::ast::dump;
use tract_nnef::ast::parse;

#[test]
fn parse_alexnet() {
    let content = std::fs::read_to_string("tests/alexnet.nnef").unwrap();
    parse::parse_document(&content).unwrap();
}

#[test]
fn parse_dump_parse_alexnet() {
    let content = std::fs::read_to_string("tests/alexnet.nnef").unwrap();
    let ast = parse::parse_document(&content).unwrap();
    let mut dumped = vec!();
    dump::Dumper::new(&mut dumped).document(&ast).unwrap();

    let dumped = String::from_utf8(dumped).unwrap();
    let ast2 = parse::parse_document(&dumped).unwrap();

    assert_eq!(ast, ast2);
}

#[test]
fn parse_stdlib() {
    let content = std::fs::read_to_string("stdlib.nnef").unwrap();
    parse::parse_fragments(&content).unwrap();
}

#[test]
fn parse_dump_parse_stdlib() {
    let content = std::fs::read_to_string("stdlib.nnef").unwrap();
    let ast = parse::parse_fragments(&content).unwrap();
    let mut dumped = vec!();
    dump::Dumper::new(&mut dumped).fragments(&ast).unwrap();

    let dumped = String::from_utf8(dumped).unwrap();
    println!("{}", dumped);
    let ast2 = parse::parse_fragments(&dumped).unwrap();

    assert_eq!(ast.len(), ast2.len());
    for (a, b) in ast.iter().zip(ast2.iter()) {
        assert_eq!(a, b);
    }
}
