//! Regression test for CWE-770 in NNEF string-tensor loading.
//!
//! `read_tensor` used to read a per-item `len: u32` straight from the (untrusted) NNEF file and
//! then `Vec::with_capacity(len)` + `unsafe { set_len(len) }` — a malicious file advertising a
//! huge string length forced an unbounded *upfront* allocation (abort on OOM / memory exhaustion)
//! before any byte was read. This test crafts such a file (declares a 128 MiB string length but
//! ships only 3 bytes) and asserts the largest *single* heap allocation stays bounded.
//!
//! A counting `#[global_allocator]` (test-only) records the largest single allocation. It is
//! process-global, so this is the only test target in the binary (no parallel pollution).

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

use tract_nnef::tensors::read_tensor;

struct CountingAllocator;
static MAX_SINGLE_ALLOC: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let _ = MAX_SINGLE_ALLOC
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| Some(c.max(size)));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Build a minimal NNEF tensor of rank 1 with a single STRING item whose declared byte length is
/// `DECLARED_LEN` but whose body carries only 3 bytes ("abc"). Layout matches `Header` (128 bytes).
fn malicious_nnef_string_tensor(declared_len: u32) -> Vec<u8> {
    let mut buf = vec![0u8; 128];
    buf[0] = 0x4e;
    buf[1] = 0xef; // magic NNEF
    buf[2] = 1;
    buf[3] = 0; // version 1.0
    buf[4] = 3; // data_size_bytes = 3 (LE)
    buf[8] = 1; // rank = 1
    buf[12] = 1; // dims[0] = 1 (one string item)
    // bits_per_item at offset 44 = u32::MAX (String uses this sentinel)
    buf[44] = 0xff;
    buf[45] = 0xff;
    buf[46] = 0xff;
    buf[47] = 0xff;
    // item_type at offset 48 = 0x1000 (LE)
    buf[48] = 0x00;
    buf[49] = 0x10;
    // item_type_vendor at offset 50 = 0x5452 ("TR", LE)
    buf[50] = 0x52;
    buf[51] = 0x54;
    // remaining header bytes are zero
    // body: per string item, len: u32 (LE) then that many bytes
    buf.extend_from_slice(&declared_len.to_le_bytes());
    buf.extend_from_slice(b"abc"); // only 3 bytes present
    buf
}

// Malicious server advertises 128 MiB but ships only 3 bytes.
const DECLARED_LEN: u32 = 128 * 1024 * 1024;
// We tolerate single allocations up to this (covers tract's own tensor/string bookkeeping); the
// vulnerability is a single allocation on the order of DECLARED_LEN.
const MAX_ACCEPTABLE_SINGLE_ALLOC: usize = 16 * 1024 * 1024;

#[test]
fn string_item_len_must_not_drive_unbounded_alloc() {
    let bytes = malicious_nnef_string_tensor(DECLARED_LEN);
    MAX_SINGLE_ALLOC.store(0, Ordering::Relaxed);

    // The call is expected to fail (length mismatch) — we only care about the allocation it made.
    let _ = read_tensor(Cursor::new(bytes));

    let max_alloc = MAX_SINGLE_ALLOC.load(Ordering::Relaxed);
    println!(
        "largest single allocation during read_tensor = {max_alloc} bytes (malicious declared len = {DECLARED_LEN})"
    );
    assert!(
        max_alloc <= MAX_ACCEPTABLE_SINGLE_ALLOC,
        "read_tensor pre-allocated {max_alloc} bytes from an untrusted string length (CWE-770)"
    );
}
