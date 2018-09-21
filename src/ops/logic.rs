use ops::prelude::*;

element_map!(Not, [bool], |a| !a );

element_bin!(And, [bool], |mut a, b| { a &= &b; a });
element_bin!(Or, [bool], |mut a, b| { a |= &b; a });
element_bin!(Xor, [bool], |mut a, b| { a ^= &b; a });
