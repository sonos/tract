import CoreImage
import CoreVideo
import Foundation
import SwiftUI

/// Owns the inference model and runs it on a dedicated background queue.
/// The capture-side `submit(pixelBuffer:)` drops frames when the queue is
/// busy — that minimizes latency between camera and display.
@MainActor
final class InferencePipeline: ObservableObject {
    @Published private(set) var status: String = "idle"
    @Published private(set) var ready: Bool = false
    @Published private(set) var composite: CGImage?
    @Published private(set) var alphaMatte: CGImage?
    @Published private(set) var prepFrame: CGImage?
    @Published private(set) var lastInferenceMs: Double = 0
    @Published private(set) var fps: Double = 0

    private let worker = Worker()
    private var fpsAccumulator: [CFAbsoluteTime] = []

    /// One in-flight frame at a time. If a new frame arrives while the worker
    /// is still busy, it's dropped at the camera boundary — preventing the
    /// pixel-buffer / submit-closure queue from growing under heavy backends
    /// like CPU (where inference is much slower than 30 fps).
    private let workSlot = DispatchSemaphore(value: 1)

    func load(
        model: ModelKind,
        h: Int,
        w: Int,
        backend: Backend,
        provider: CoreMLComputeUnits
    ) async {
        ready = false
        composite = nil
        alphaMatte = nil
        prepFrame = nil
        fpsAccumulator.removeAll()
        fps = 0
        status = "loading \(model.label) \(w)×\(h) on \(backend.label)…"

        let started = Date()
        do {
            let inf = try TractInference(
                kind: model,
                h: h,
                w: w,
                backend: backend,
                computeUnits: provider
            )
            await worker.setInference(inf)
            let dt = Int(Date().timeIntervalSince(started) * 1000)
            ready = true
            let providerSuffix = backend == .tractCoreML ? " (\(provider.label))" : ""
            status = "\(model.label) \(w)×\(h) on \(backend.label)\(providerSuffix) ready (\(dt) ms)"
        } catch {
            ready = false
            status = "load failed: \(error.localizedDescription)"
        }
    }

    /// Called from any actor context — safe from the camera queue.
    /// Frames are dropped immediately (no allocation, no enqueue) if a
    /// previous frame is still in flight.
    nonisolated func submit(pixelBuffer: CVPixelBuffer) {
        guard workSlot.wait(timeout: .now()) == .success else {
            return  // worker still busy; drop this frame
        }
        worker.submit(pixelBuffer: pixelBuffer) { [weak self] result in
            self?.workSlot.signal()
            Task { @MainActor in
                self?.apply(result)
            }
        }
    }

    private func apply(_ result: WorkerResult) {
        switch result {
        case .success(let composite, let alpha, let prep, let ms):
            self.composite = composite
            self.alphaMatte = alpha
            self.prepFrame = prep
            self.lastInferenceMs = ms
            recordFrame()
        case .dropped:
            break
        case .failure(let message):
            status = "frame failed: \(message)"
        }
    }

    private func recordFrame() {
        let now = CFAbsoluteTimeGetCurrent()
        fpsAccumulator.append(now)
        while let oldest = fpsAccumulator.first, now - oldest > 1.0 {
            fpsAccumulator.removeFirst()
        }
        fps = Double(fpsAccumulator.count)
    }
}

enum WorkerResult: Sendable {
    case success(CGImage, CGImage?, CGImage?, Double)
    case dropped
    case failure(String)
}

private final class Worker: @unchecked Sendable {
    private let queue = DispatchQueue(label: "tract-coreml.demo.inference", qos: .userInitiated)
    private var inference: TractInference?

    func setInference(_ new: TractInference) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                self.inference = new
                cont.resume()
            }
        }
    }

    func submit(
        pixelBuffer: CVPixelBuffer,
        completion: @escaping @Sendable (WorkerResult) -> Void
    ) {
        queue.async { [weak self] in
            guard let self else { return }
            guard let inf = self.inference else {
                completion(.dropped)
                return
            }

            let ci = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cg = sharedCIContext.createCGImage(ci, from: ci.extent) else {
                completion(.failure("createCGImage returned nil"))
                return
            }
            let prep = ImageConversion.prepareFrame(cg, targetH: inf.h, targetW: inf.w)

            let alpha: [Float16]
            let ms: Double
            do {
                let out = try inf.runFrame(prep.chwF16)
                alpha = out.alpha
                ms = out.ms
            } catch {
                completion(.failure(error.localizedDescription))
                return
            }

            guard let composite = ImageConversion.compositeOverColor(
                frame: prep,
                alpha: alpha,
                bg: (r: 16, g: 200, b: 96)
            ) else {
                completion(.failure("composite returned nil"))
                return
            }
            let matte = ImageConversion.alphaToCGImage(alpha, h: inf.h, w: inf.w)
            let prepImage = ImageConversion.rgbaToCGImage(prep.rgbaU8, h: inf.h, w: inf.w)
            completion(.success(composite, matte, prepImage, ms))
        }
    }
}
