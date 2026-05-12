import AVFoundation
import CoreImage
import CoreVideo
import SwiftUI

let sharedCIContext = CIContext(options: nil)

// CVPixelBuffer is a CoreFoundation type; safe to ferry across actor boundaries
// as long as we don't mutate the contents.
extension CVPixelBuffer: @unchecked @retroactive Sendable {}

@MainActor
final class CameraSession: NSObject, ObservableObject {
    enum State: Equatable {
        case idle
        case requestingPermission
        case denied
        case running
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    let session = AVCaptureSession()

    /// Each new captured pixel buffer goes here; pipeline drops if busy.
    nonisolated let pipeline: InferencePipeline

    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "tract-coreml.demo.video", qos: .userInitiated)

    init(pipeline: InferencePipeline) {
        self.pipeline = pipeline
        super.init()
    }

    func start() async {
        state = .requestingPermission

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted {
                state = .denied
                return
            }
        case .denied, .restricted:
            state = .denied
            return
        @unknown default:
            state = .denied
            return
        }

        do {
            try configure()
        } catch {
            state = .failed(error.localizedDescription)
            return
        }

        session.startRunning()
        state = .running
    }

    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .unspecified
        ) ?? AVCaptureDevice.default(for: .video) else {
            throw CaptureError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CaptureError.cannotAddInput
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(videoOutput) else {
            throw CaptureError.cannotAddOutput
        }
        session.addOutput(videoOutput)
    }
}

extension CameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = sampleBuffer.imageBuffer else { return }
        pipeline.submit(pixelBuffer: pixelBuffer)
    }
}

enum CaptureError: LocalizedError {
    case noDevice
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noDevice: return "No video capture device found."
        case .cannotAddInput: return "Could not attach the camera input."
        case .cannotAddOutput: return "Could not attach the video data output."
        }
    }
}
