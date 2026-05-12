import SwiftUI

@main
struct TractDemoApp: App {
    var body: some Scene {
        WindowGroup("tract-coreml demo") {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 720)
    }
}
