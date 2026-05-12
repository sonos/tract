// C ABI for the macOS demo app — see coreml-demo-rs/src/lib.rs.
//
// Stable as long as the Rust signatures don't change. If you alter a function
// signature in lib.rs, mirror it here AND in TractDemo/Bridging-Header.h.

#ifndef TRACT_DEMO_H
#define TRACT_DEMO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ModelHandle ModelHandle;

void tract_demo_init(void);

int32_t tract_demo_create_model(
    uint32_t model_kind,            // 0=MODNet, 1=RVM
    uint32_t h,
    uint32_t w,
    uint32_t backend,               // 0=TractCpu, 1=TractMetal, 2=TractCoreML
    uint32_t coreml_compute_units,  // 0=CPUOnly, 1=CPUAndGPU, 2=CPUAndANE, 3=All
    ModelHandle **out_handle);

int32_t tract_demo_run_frame(
    ModelHandle *handle,
    const uint16_t *src_rgb_f16,    // [3 * H * W] half-floats, CHW order
    uint16_t *out_alpha_f16,        // caller-allocated [1 * H * W]
    double *out_ms_elapsed);

void tract_demo_destroy_model(ModelHandle *handle);

size_t tract_demo_get_last_error(uint8_t *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif  // TRACT_DEMO_H
