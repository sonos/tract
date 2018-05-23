use super::*;

#[test]
fn new_abstract_tensor() {
    assert_eq!(
        ATensor::new(),
        ATensor {
            datatype: AType::Any,
            shape: AShape::Closed(vec![]),
            value: AValue::Any,
        }
    );
}

#[test]
fn unify_same_datatype() {
    let dt = AType::Only(DataType::DT_FLOAT);
    assert_eq!(unify_datatype(&dt, &dt).unwrap(), dt);
}

#[test]
fn unify_different_datatypes_only() {
    let dt1 = AType::Only(DataType::DT_FLOAT);
    let dt2 = AType::Only(DataType::DT_DOUBLE);
    assert!(unify_datatype(&dt1, &dt2).is_err());
}

#[test]
fn unify_different_datatypes_any_left() {
    let dt = AType::Only(DataType::DT_FLOAT);
    assert_eq!(unify_datatype(&AType::Any, &dt).unwrap(), dt);
}

#[test]
fn unify_different_datatypes_any_right() {
    let dt = AType::Only(DataType::DT_FLOAT);
    assert_eq!(unify_datatype(&dt, &AType::Any).unwrap(), dt);
}

#[test]
fn unify_same_shape_1() {
    let s = AShape::Closed(vec![]);
    assert_eq!(unify_shape(&s, &s).unwrap(), s);
}

#[test]
fn unify_same_shape_2() {
    use super::ADimension::*;
    let s = AShape::Closed(vec![Any]);
    assert_eq!(unify_shape(&s, &s).unwrap(), s);
}

#[test]
fn unify_same_shape_3() {
    use super::ADimension::*;
    let s = AShape::Closed(vec![Only(1), Only(2)]);
    assert_eq!(unify_shape(&s, &s).unwrap(), s);
}

#[test]
fn unify_different_shapes_1() {
    use super::ADimension::*;
    let s1 = AShape::Closed(vec![Only(1), Only(2)]);
    let s2 = AShape::Closed(vec![Only(1)]);
    assert!(unify_shape(&s1, &s2).is_err());
}

#[test]
fn unify_different_shapes_2() {
    use super::ADimension::*;
    let s1 = AShape::Closed(vec![Only(1), Only(2)]);
    let s2 = AShape::Closed(vec![Any]);
    assert!(unify_shape(&s1, &s2).is_err());
}

#[test]
fn unify_different_shapes_3() {
    use super::ADimension::*;
    let s1 = AShape::Open(vec![Only(1), Only(2)]);
    let s2 = AShape::Closed(vec![Any]);
    assert!(unify_shape(&s1, &s2).is_err());
}

#[test]
fn unify_different_shapes_4() {
    use super::ADimension::*;
    let s1 = AShape::Closed(vec![Any]);
    let s2 = AShape::Closed(vec![Any]);
    let sr = AShape::Closed(vec![Any]);
    assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
}

#[test]
fn unify_different_shapes_5() {
    use super::ADimension::*;
    let s1 = AShape::Closed(vec![Any]);
    let s2 = AShape::Closed(vec![Only(1)]);
    let sr = AShape::Closed(vec![Only(1)]);
    assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
}

#[test]
fn unify_different_shapes_6() {
    use super::ADimension::*;
    let s1 = AShape::Open(vec![]);
    let s2 = AShape::Closed(vec![Only(1)]);
    let sr = AShape::Closed(vec![Only(1)]);
    assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
}

#[test]
fn unify_different_shapes_7() {
    use super::ADimension::*;
    let s1 = AShape::Open(vec![Any, Only(2)]);
    let s2 = AShape::Closed(vec![Only(1), Any, Any]);
    let sr = AShape::Closed(vec![Only(1), Only(2), Any]);
    assert_eq!(unify_shape(&s1, &s2).unwrap(), sr);
}

#[test]
fn unify_same_value() {
    use ndarray::prelude::*;
    let dt = AValue::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
    assert_eq!(unify_value(&dt, &dt).unwrap(), dt);
}

#[test]
fn unify_different_values_only() {
    use ndarray::prelude::*;
    let dt1 = AValue::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
    let dt2 = AValue::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[2]))));
    assert!(unify_value(&dt1, &dt2).is_err());
}

#[test]
fn unify_different_values_any_left() {
    use ndarray::prelude::*;
    let dt = AValue::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
    assert_eq!(unify_value(&AValue::Any, &dt).unwrap(), dt);
}

#[test]
fn unify_different_values_any_right() {
    use ndarray::prelude::*;
    let dt = AValue::Only(Matrix::F32(ArrayD::zeros(IxDyn(&[1]))));
    assert_eq!(unify_value(&dt, &AValue::Any).unwrap(), dt);
}