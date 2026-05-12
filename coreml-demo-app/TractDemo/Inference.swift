import Foundation

enum InferenceError: LocalizedError {
    case createFailed(String)
    case runFailed(String)
    case notImplemented(String)

    var errorDescription: String? {
        switch self {
        case .createFailed(let m): return "create_model failed: \(m)"
        case .runFailed(let m): return "run_frame failed: \(m)"
        case .notImplemented(let m): return "not implemented: \(m)"
        }
    }
}

enum ModelKind: UInt32, CaseIterable, Hashable {
    case modnet = 0
    case rvm = 1

    var label: String {
        switch self {
        case .modnet: return "MODNet"
        case .rvm: return "RVM"
        }
    }

    var availableResolutions: [ResolutionChoice] {
        switch self {
        case .modnet: return [.modnet256, .modnet384, .modnet512]
        case .rvm: return [.rvm480x640]
        }
    }

    var defaultResolution: ResolutionChoice {
        switch self {
        case .modnet: return .modnet512
        case .rvm: return .rvm480x640
        }
    }
}

enum ResolutionChoice: Hashable, CaseIterable {
    case modnet256
    case modnet384
    case modnet512
    case rvm480x640

    var h: Int {
        switch self {
        case .modnet256: return 256
        case .modnet384: return 384
        case .modnet512: return 512
        case .rvm480x640: return 480
        }
    }

    var w: Int {
        switch self {
        case .modnet256: return 256
        case .modnet384: return 384
        case .modnet512: return 512
        case .rvm480x640: return 640
        }
    }

    var label: String { "\(w)×\(h)" }
}

enum Backend: UInt32, CaseIterable, Hashable {
    case tractCpu = 0
    case tractMetal = 1
    case tractCoreML = 2
    /// Apple's `MLModel` directly, no tract. Needs a bundled `.mlmodel` —
    /// not implemented in M4 first cut; surfaces an error so the demo
    /// reviewer can see the picker entry exists for future work.
    case directCoreML = 99

    var label: String {
        switch self {
        case .tractCpu: return "tract CPU"
        case .tractMetal: return "tract Metal"
        case .tractCoreML: return "tract CoreML"
        case .directCoreML: return "direct CoreML"
        }
    }
}

enum CoreMLComputeUnits: UInt32, CaseIterable, Hashable {
    case cpuOnly = 0
    case cpuAndGpu = 1
    case cpuAndAne = 2
    case all = 3

    var label: String {
        switch self {
        case .cpuOnly: return "CPU only"
        case .cpuAndGpu: return "CPU+GPU"
        case .cpuAndAne: return "CPU+ANE"
        case .all: return "All"
        }
    }
}

/// Owns one tract `RunnableModel` behind the Rust FFI. We mark it
/// `@unchecked Sendable` because the pipeline pins each instance to a single
/// dispatch queue (no cross-thread mutation in practice).
final class TractInference: @unchecked Sendable {
    let kind: ModelKind
    let h: Int
    let w: Int
    let backend: Backend
    private var handle: OpaquePointer?

    init(
        kind: ModelKind,
        h: Int,
        w: Int,
        backend: Backend,
        computeUnits: CoreMLComputeUnits
    ) throws {
        self.kind = kind
        self.h = h
        self.w = w
        self.backend = backend

        if backend == .directCoreML {
            throw InferenceError.notImplemented(
                "direct CoreML — bundle a .mlmodel and wire MLModel.prediction(from:)"
            )
        }

        var ptr: OpaquePointer?
        let rc = tract_demo_create_model(
            kind.rawValue,
            UInt32(h),
            UInt32(w),
            backend.rawValue,
            computeUnits.rawValue,
            &ptr
        )
        if rc != 0 {
            throw InferenceError.createFailed(Self.lastError())
        }
        self.handle = ptr
    }

    deinit {
        if let h = handle {
            tract_demo_destroy_model(h)
        }
    }

    /// `rgbCHW` is `[3 * H * W]` half-floats in CHW order (R plane, G, B).
    /// Returns the alpha matte `[H * W]` plus elapsed inference time in ms.
    func runFrame(_ rgbCHW: [Float16]) throws -> (alpha: [Float16], ms: Double) {
        precondition(rgbCHW.count == 3 * h * w, "rgbCHW length mismatch")
        var alpha = [Float16](repeating: 0, count: h * w)
        var ms: Double = 0
        let rc = rgbCHW.withUnsafeBufferPointer { srcF16 in
            srcF16.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: srcF16.count) { srcU16 in
                alpha.withUnsafeMutableBufferPointer { dstF16 in
                    dstF16.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: dstF16.count) { dstU16 in
                        tract_demo_run_frame(handle, srcU16, dstU16, &ms)
                    }
                }
            }
        }
        if rc != 0 {
            throw InferenceError.runFailed(Self.lastError())
        }
        return (alpha, ms)
    }

    static func lastError() -> String {
        let len = tract_demo_get_last_error(nil, 0)
        guard len > 0 else { return "(empty error)" }
        var buf = [UInt8](repeating: 0, count: len)
        _ = buf.withUnsafeMutableBufferPointer { p in
            tract_demo_get_last_error(p.baseAddress, p.count)
        }
        return String(decoding: buf, as: UTF8.self)
    }
}
