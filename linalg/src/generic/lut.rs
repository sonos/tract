use crate::frame::lut::LutKer;

#[derive(Clone, Debug, Hash)]
pub struct GenericLut8;

impl LutKer for GenericLut8 {
    fn name() -> &'static str {
        "generic"
    }

    fn input_alignment_bytes() -> usize {
        1
    }

    fn table_alignment_bytes() -> usize {
        1
    }

    fn n() -> usize {
        8
    }

    unsafe fn run(buf: *mut u8, len: usize, table: *const u8) {
        debug_assert!(len % Self::n() == 0);
        debug_assert!(buf as usize % Self::input_alignment_bytes() == 0);
        debug_assert!(table as usize % Self::table_alignment_bytes() == 0);
        for i in 0..((len / 8) as isize) {
            let ptr = buf.offset(8 * i);
            *ptr.offset(0) = *table.offset(*ptr.offset(0) as isize);
            *ptr.offset(1) = *table.offset(*ptr.offset(1) as isize);
            *ptr.offset(2) = *table.offset(*ptr.offset(2) as isize);
            *ptr.offset(3) = *table.offset(*ptr.offset(3) as isize);
            *ptr.offset(4) = *table.offset(*ptr.offset(4) as isize);
            *ptr.offset(5) = *table.offset(*ptr.offset(5) as isize);
            *ptr.offset(6) = *table.offset(*ptr.offset(6) as isize);
            *ptr.offset(7) = *table.offset(*ptr.offset(7) as isize);
        }
    }
}

#[cfg(test)]
#[macro_use]
pub mod test {
    lut_frame_tests!(true, crate::generic::GenericLut8);
}
