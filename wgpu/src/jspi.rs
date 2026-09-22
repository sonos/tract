//! Optional JSPI hybrid fallback.
//!
//! Mid-graph GPU→CPU readback needs the wasm stack to suspend across
//! `mapAsync`. JSPI does that; Asyncify is unavailable (`wasm32-unknown-emscripten`
//! only) and must not become a load-time dependency:
//!
//! | Engine | WebGPU | JSPI |
//! |---|---|---|
//! | Chrome / Edge | 113+ | 137+ |
//! | Firefox | Win 141+, macOS 147+ | 139+ |
//! | Safari | 26.0 | 27 |
//!
//! Default wasm builds carry **no** `WebAssembly.Suspending` imports, so Safari
//! 26 still loads a fully-covered model. Enable Cargo feature `jspi` for the
//! hybrid binary (uncovered ops fall back to tract CPU kernels). Native always
//! has blocking poll, so hybrid is on there with no feature flag.

#[cfg(all(target_arch = "wasm32", feature = "jspi"))]
use tract_core::internal::*;

/// True when a GPU→CPU sync can complete in the middle of `eval`.
///
/// Native: always (blocking `PollType::Wait`).
/// Wasm: only if this crate was built with `--features jspi` **and** the
/// browser exposes `WebAssembly.Suspending` / `WebAssembly.promising`.
pub fn hybrid_fallback_available() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
    #[cfg(all(target_arch = "wasm32", feature = "jspi"))]
    {
        jspi_in_browser()
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "jspi")))]
    {
        false
    }
}

/// `WebAssembly.Suspending` + `WebAssembly.promising` present on the global.
#[cfg(target_arch = "wasm32")]
pub fn jspi_in_browser() -> bool {
    let Ok(wa) =
        js_sys::Reflect::get(&js_sys::global(), &wasm_bindgen::JsValue::from_str("WebAssembly"))
    else {
        return false;
    };
    if wa.is_undefined() || wa.is_null() {
        return false;
    }
    let sus = js_sys::Reflect::get(&wa, &wasm_bindgen::JsValue::from_str("Suspending")).ok();
    let prom = js_sys::Reflect::get(&wa, &wasm_bindgen::JsValue::from_str("promising")).ok();
    matches!(sus, Some(ref v) if v.is_function()) && matches!(prom, Some(ref v) if v.is_function())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn jspi_in_browser() -> bool {
    false
}

/// Block on [`WgpuContext::download_async`] via JSPI. Must run inside a
/// `#[wasm_bindgen(jspi)]` export (or a task spawned from one). Does not hold
/// the `WGPU_QUEUE` TLS borrow across the suspend.
#[cfg(all(target_arch = "wasm32", feature = "jspi"))]
pub fn download_jspi(buffer: &wgpu::Buffer, offset: u64, len: u64) -> TractResult<Vec<u8>> {
    use wasm_bindgen::JsValue;

    crate::with_wgpu_queue(|q| q.flush())?;
    let ctx = crate::wgpu_context();
    let buffer = buffer.clone();
    let promise = wasm_bindgen_futures::future_to_promise(async move {
        match ctx.download_async(&buffer, offset, len).await {
            Ok(bytes) => Ok(js_sys::Uint8Array::from(bytes.as_slice()).into()),
            Err(e) => Err(JsValue::from_str(&format!("{e:#}"))),
        }
    });
    #[allow(deprecated)]
    let val = js_sys::futures::jspi_block_on_promise(&promise)
        .map_err(|e| anyhow::anyhow!("JSPI GPU readback failed: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&val).to_vec())
}
