use std::fmt;
use std::fmt::Debug;

use num_traits::Zero;

use super::MatMatMulKer;

#[derive(PartialEq, Clone)]
pub enum FusedSpec<TI: Copy + Debug, TC: Copy + Debug> {
    Min(TI),
    Max(TI),
    AddC,
    PerRowMul(Vec<TI>),
    PerRowAdd(Vec<TI>),
    PerColMul(Vec<TI>),
    PerColAdd(Vec<TI>),
    AddRowColProducts(Vec<TI>, Vec<TI>),
    QI8Even(TI, u8, TC),
}

impl<TI: Copy + Debug, TC: Copy + Debug> Debug for FusedSpec<TI, TC> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FusedSpec::Min(t) => write!(fmt, "Min({:?})", t),
            FusedSpec::Max(t) => write!(fmt, "Max({:?})", t),
            FusedSpec::AddC => write!(fmt, "AddC"),
            FusedSpec::PerRowMul(_) => write!(fmt, "PerRowMul"),
            FusedSpec::PerRowAdd(_) => write!(fmt, "PerRowAdd"),
            FusedSpec::PerColMul(_) => write!(fmt, "PerColMul"),
            FusedSpec::PerColAdd(_) => write!(fmt, "PerColAdd"),
            FusedSpec::AddRowColProducts(_, _) => write!(fmt, "AddRowColProducts"),
            FusedSpec::QI8Even(_, _, _) => write!(fmt, "QI8Even"),
        }
    }
}

#[repr(C, usize)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub enum FusedKerSpec<TI: Copy, TC:Copy> {
    Done,
    Min(TI),
    Max(TI),
    AddC,
    PerRowMul(*const TI),
    PerRowAdd(*const TI),
    PerColMul(*const TI),
    PerColAdd(*const TI),
    AddRowColProducts(*const TI, *const TI),
    QI8Even(TI, u8, TC),
}

pub struct ScratchSpaceFusedNonLinear<TI: Copy, TC:Copy> {
    uspecs: Vec<FusedKerSpec<TI, TC>>,
    non_linear_buffers: Vec<Vec<TI>>,
}

impl<TI: Copy, TC: Copy> Default for ScratchSpaceFusedNonLinear<TI, TC> {
    fn default() -> ScratchSpaceFusedNonLinear<TI, TC> {
        ScratchSpaceFusedNonLinear { uspecs: vec![], non_linear_buffers: vec![] }
    }
}

impl<TI: Copy, TC: Copy> ScratchSpaceFusedNonLinear<TI, TC> {
    pub unsafe fn for_tile<TA, TB, K: MatMatMulKer<TA, TB, TC, TI>>(
        &mut self,
        specs: &[FusedSpec<TI, TC>],
        down: usize,
        right: usize,
    ) -> *const FusedKerSpec<TI, TC>
    where
        TA: Copy,
        TB: Copy,
        TC: Copy + Debug,
        TI: Copy + Debug + Zero,
    {
        self.uspecs.clear();
        for spec in specs {
            let s = match spec {
                FusedSpec::Min(m) => FusedKerSpec::Min(*m),
                FusedSpec::Max(m) => FusedKerSpec::Max(*m),
                FusedSpec::AddC => FusedKerSpec::AddC,
                FusedSpec::PerRowMul(v) => {
                    let have = v.len() - down * K::mr();
                    let ptr = if have < K::mr() {
                        let mut buf = vec![TI::zero(); K::mr()];
                        buf[..have].copy_from_slice(&v[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(down * K::mr())
                    };
                    FusedKerSpec::PerRowMul(ptr)
                }
                FusedSpec::PerRowAdd(v) => {
                    let have = v.len() - down * K::mr();
                    let ptr = if have < K::mr() {
                        let mut buf = vec![TI::zero(); K::mr()];
                        buf[..have].copy_from_slice(&v[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(down * K::mr())
                    };
                    FusedKerSpec::PerRowAdd(ptr)
                }
                FusedSpec::PerColMul(v) => {
                    let have = v.len() - right * K::nr();
                    let ptr = if have < K::nr() {
                        let mut buf = vec![TI::zero(); K::nr()];
                        buf[..have].copy_from_slice(&v[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(right * K::nr())
                    };
                    FusedKerSpec::PerColMul(ptr)
                }
                FusedSpec::PerColAdd(v) => {
                    let have = v.len() - right * K::nr();
                    let ptr = if have < K::nr() {
                        let mut buf = vec![TI::zero(); K::nr()];
                        buf[..have].copy_from_slice(&v[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        v.as_ptr().add(right * K::nr())
                    };
                    FusedKerSpec::PerColAdd(ptr)
                }
                FusedSpec::AddRowColProducts(rows, cols) => {
                    let have = rows.len() - down * K::mr();
                    let row_ptr = if have < K::mr() {
                        let mut buf = vec![TI::zero(); K::mr()];
                        buf[..have].copy_from_slice(&rows[down * K::mr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        rows.as_ptr().add(down * K::mr())
                    };
                    let have = cols.len() - right * K::nr();
                    let col_ptr = if have < K::nr() {
                        let mut buf = vec![TI::zero(); K::nr()];
                        buf[..have].copy_from_slice(&cols[right * K::nr()..][..have]);
                        let ptr = buf.as_ptr();
                        self.non_linear_buffers.push(buf);
                        ptr
                    } else {
                        cols.as_ptr().add(right * K::nr())
                    };
                    FusedKerSpec::AddRowColProducts(row_ptr, col_ptr)
                }
                FusedSpec::QI8Even(scale, shift, zero) => {
                    FusedKerSpec::QI8Even(*scale, *shift, *zero)
                }
            };
            self.uspecs.push(s);
        }
        self.uspecs.push(FusedKerSpec::Done);
        self.uspecs.as_ptr()
    }
}
