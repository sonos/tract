import SwiftUI

struct ContentView: View {
    @StateObject private var pipeline = InferencePipeline()
    @StateObject private var camera: CameraSession

    @State private var model: ModelKind = .modnet
    @State private var resolution: ResolutionChoice = .modnet512
    @State private var backend: Backend = .tractCoreML
    @State private var provider: CoreMLComputeUnits = .all

    init() {
        let p = InferencePipeline()
        _pipeline = StateObject(wrappedValue: p)
        _camera = StateObject(wrappedValue: CameraSession(pipeline: p))
    }

    var body: some View {
        VStack(spacing: 0) {
            controlsBar

            HStack(spacing: 0) {
                ZStack {
                    Color.black.ignoresSafeArea()
                    switch camera.state {
                    case .idle, .requestingPermission:
                        ProgressView("requesting camera permission…")
                            .foregroundStyle(.white)
                    case .denied:
                        VStack(spacing: 8) {
                            Text("Camera access denied.").font(.title3)
                            Text("Enable in System Settings → Privacy & Security → Camera.")
                                .foregroundStyle(.secondary)
                        }
                        .foregroundStyle(.white)
                    case .running:
                        CameraPreview(session: camera.session)
                    case .failed(let message):
                        Text("Capture error: \(message)").foregroundStyle(.red)
                    }
                }
                paneWith(image: pipeline.prepFrame, label: "prep input")
                paneWith(image: pipeline.alphaMatte, label: "alpha matte")
                paneWith(image: pipeline.composite, label: "composite over green")
            }

            statsBar
        }
        .frame(minWidth: 1800, minHeight: 600)
        .task { await camera.start() }
        .task { await reload() }
        .onChange(of: model) {
            // Reset resolution to a valid one for the new model.
            resolution = model.defaultResolution
            Task { await reload() }
        }
        .onChange(of: resolution) { Task { await reload() } }
        .onChange(of: backend) { Task { await reload() } }
        .onChange(of: provider) {
            if backend == .tractCoreML { Task { await reload() } }
        }
    }

    // MARK: - controls

    @ViewBuilder
    private var controlsBar: some View {
        HStack(spacing: 18) {
            labelled("Model") {
                Picker("", selection: $model) {
                    ForEach(ModelKind.allCases, id: \.self) { m in
                        Text(m.label).tag(m)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .fixedSize()
            }

            labelled("Resolution") {
                Picker("", selection: $resolution) {
                    ForEach(model.availableResolutions, id: \.self) { r in
                        Text(r.label).tag(r)
                    }
                }
                .labelsHidden()
                .fixedSize()
            }

            labelled("Backend") {
                Picker("", selection: $backend) {
                    ForEach(Backend.allCases, id: \.self) { b in
                        Text(b.label).tag(b)
                    }
                }
                .labelsHidden()
                .fixedSize()
            }

            if backend == .tractCoreML {
                labelled("Provider") {
                    Picker("", selection: $provider) {
                        ForEach(CoreMLComputeUnits.allCases, id: \.self) { p in
                            Text(p.label).tag(p)
                        }
                    }
                    .labelsHidden()
                    .fixedSize()
                }
            }

            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.thinMaterial)
    }

    @ViewBuilder
    private func labelled<V: View>(_ label: String, @ViewBuilder content: () -> V) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            content()
        }
    }

    @ViewBuilder
    private var statsBar: some View {
        HStack(spacing: 24) {
            Label(String(format: "%.1f ms", pipeline.lastInferenceMs),
                  systemImage: "stopwatch")
            Label(String(format: "%.0f fps", pipeline.fps),
                  systemImage: "speedometer")
            Spacer()
            Text(pipeline.status)
                .lineLimit(1)
                .truncationMode(.middle)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .font(.system(.callout, design: .monospaced))
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.regularMaterial)
    }

    @ViewBuilder
    private func paneWith(image: CGImage?, label: String) -> some View {
        ZStack(alignment: .topLeading) {
            Color.black.opacity(0.85).ignoresSafeArea()
            if let image {
                Image(decorative: image, scale: 1.0)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack {
                    ProgressView()
                    Text("waiting…")
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Text(label)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .padding(6)
                .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 4))
                .padding(8)
        }
    }

    private func reload() async {
        await pipeline.load(
            model: model,
            h: resolution.h,
            w: resolution.w,
            backend: backend,
            provider: provider
        )
    }
}
