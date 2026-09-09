/* tslint:disable */
/* eslint-disable */

/**
 * RGBA8 packed pixels → NCHW f32 `[1, 3, H, W]` (alpha dropped), then readback.
 */
export function ingest_rgba8(width: number, height: number, rgba: Uint8Array): Promise<Float32Array>;

export function run_hybrid_logsoftmax(_input: Float32Array): Float32Array;

/**
 * Mul-by-2 then sigmoid on a length-8 f32 vector, shape `[2, 4]`.
 */
export function run_two_op(input: Float32Array): Promise<Float32Array>;

/**
 * Runs the model on one RGBA frame and writes the mask into the texture above.
 * Nothing is read back: the mask stays on the GPU for the compositor.
 */
export function segment_into_texture(width: number, height: number, rgba: Uint8Array): number;

/**
 * The `GPUDevice` the model runs on. A compositor built against this device
 * can sample the mask where the model leaves it.
 */
export function segmentation_device(): any;

/**
 * The `GPUTexture` the mask is written into, created on first use.
 */
export function segmentation_mask_texture(): any;

/**
 * Whole selfie-segmentation model on the GPU: `iters` frames, each ending in
 * the one readback the covered graph allows. Returns a one-line report.
 */
export function selfie_seg_bench(iters: number): Promise<string>;

/**
 * The comparator the GPU path has to beat: the same model on tract's CPU
 * kernels, single-threaded, in the same browser.
 */
export function selfie_seg_bench_cpu(iters: number): Promise<string>;

/**
 * Splits a frame into the phases that make up its cost: recording the graph's
 * dispatches, submitting them, and the readback's map and copy.
 */
export function selfie_seg_phases(iters: number): Promise<string>;

/**
 * `requestAdapter` / `requestDevice`. Call once before anything else.
 */
export function start(): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly ingest_rgba8: (a: number, b: number, c: number, d: number) => any;
    readonly run_hybrid_logsoftmax: (a: number, b: number) => [number, number, number, number];
    readonly run_two_op: (a: number, b: number) => any;
    readonly segment_into_texture: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly segmentation_device: () => [number, number, number];
    readonly segmentation_mask_texture: () => [number, number, number];
    readonly selfie_seg_bench: (a: number) => any;
    readonly selfie_seg_bench_cpu: (a: number) => any;
    readonly selfie_seg_phases: (a: number) => any;
    readonly start: () => any;
    readonly wasm_bindgen_c5cdf803200c69ee___convert__closures_____invoke___js_sys_6a2197ff0da30551___Function_fn_wasm_bindgen_c5cdf803200c69ee___JsValue_____wasm_bindgen_c5cdf803200c69ee___sys__Undefined___js_sys_6a2197ff0da30551___Function_fn_wasm_bindgen_c5cdf803200c69ee___JsValue_____wasm_bindgen_c5cdf803200c69ee___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_c5cdf803200c69ee___convert__closures_____invoke___wasm_bindgen_c5cdf803200c69ee___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_c5cdf803200c69ee___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_c5cdf803200c69ee___convert__closures_____invoke___wasm_bindgen_c5cdf803200c69ee___sys__JsNullable_wgpu_7873fc23d7c79807___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_c5cdf803200c69ee___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_c5cdf803200c69ee___convert__closures_____invoke___wasm_bindgen_c5cdf803200c69ee___sys__JsNullable_wgpu_7873fc23d7c79807___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_c5cdf803200c69ee___JsError___true__13: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_c5cdf803200c69ee___convert__closures_____invoke___wasm_bindgen_c5cdf803200c69ee___sys__JsNullable_wgpu_7873fc23d7c79807___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_c5cdf803200c69ee___JsError___true__14: (a: number, b: number, c: any) => [number, number];
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure_command_export: (a: number, b: number) => void;
    readonly __externref_table_dealloc_command_export: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
