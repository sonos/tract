import CoreGraphics
import CoreImage
import Foundation

/// One MODNet/RVM input rendered out of a `CGImage`: both the CHW f16 tensor
/// to feed into tract, and the same pixels as packed RGBA u8 (so we can
/// composite the result without re-rasterizing).
struct PreparedFrame {
    let chwF16: [Float16]   // [3 * H * W], normalized to [-1, 1]
    let rgbaU8: [UInt8]     // [H * W * 4], premultiplied-last RGBA
    let h: Int
    let w: Int
}

enum ImageConversion {
    /// Center-crop + resize a `CGImage` to `targetH × targetW`, returning
    /// both representations needed downstream.
    static func prepareFrame(_ image: CGImage, targetH: Int, targetW: Int) -> PreparedFrame {
        let bytesPerRow = targetW * 4
        var rgba = [UInt8](repeating: 0, count: bytesPerRow * targetH)

        rgba.withUnsafeMutableBufferPointer { buf in
            // Pin R, G, B, A byte order in memory — premultipliedLast alone
            // is ambiguous on Apple Silicon; pair with byteOrder32Big.
            let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
                | CGBitmapInfo.byteOrder32Big.rawValue
            guard let ctx = CGContext(
                data: buf.baseAddress,
                width: targetW,
                height: targetH,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: bitmapInfo
            ) else { return }

            // Center-crop ("fill"): pick the scale that fills the target rect.
            let srcW = CGFloat(image.width)
            let srcH = CGFloat(image.height)
            let scale = max(CGFloat(targetW) / srcW, CGFloat(targetH) / srcH)
            let drawW = srcW * scale
            let drawH = srcH * scale
            let dx = (CGFloat(targetW) - drawW) / 2
            let dy = (CGFloat(targetH) - drawH) / 2
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: dx, y: dy, width: drawW, height: drawH))
        }

        let n = targetH * targetW
        var chw = [Float16](repeating: 0, count: 3 * n)
        for y in 0..<targetH {
            for x in 0..<targetW {
                let idx = (y * targetW + x) * 4
                let pix = y * targetW + x
                let r = (Float(rgba[idx]) / 127.5) - 1.0
                let g = (Float(rgba[idx + 1]) / 127.5) - 1.0
                let b = (Float(rgba[idx + 2]) / 127.5) - 1.0
                chw[pix] = Float16(r)
                chw[n + pix] = Float16(g)
                chw[2 * n + pix] = Float16(b)
            }
        }
        return PreparedFrame(chwF16: chw, rgbaU8: rgba, h: targetH, w: targetW)
    }

    /// `out = fg * a + bg * (1 - a)`, where `a` is the model alpha re-mapped
    /// through a piecewise-linear sharpener: values below `lo` clamp to 0,
    /// above `hi` to 1, in-between linearly. Default tuning (`0.4..0.7`)
    /// cuts MODNet's wide soft halo without nuking hair detail.
    static func compositeOverColor(
        frame: PreparedFrame,
        alpha: [Float16],
        bg: (r: UInt8, g: UInt8, b: UInt8),
        sharpen: (lo: Float, hi: Float) = (0.4, 0.7)
    ) -> CGImage? {
        let n = frame.h * frame.w
        precondition(alpha.count == n)
        var out = [UInt8](repeating: 0, count: n * 4)
        let span = max(1e-6, sharpen.hi - sharpen.lo)
        for i in 0..<n {
            let raw = max(Float(0), min(Float(1), Float(alpha[i])))
            let a = max(Float(0), min(Float(1), (raw - sharpen.lo) / span))
            let inv = 1 - a
            let base = i * 4
            let fg = i * 4
            out[base] = UInt8((Float(frame.rgbaU8[fg]) * a + Float(bg.r) * inv).rounded())
            out[base + 1] = UInt8((Float(frame.rgbaU8[fg + 1]) * a + Float(bg.g) * inv).rounded())
            out[base + 2] = UInt8((Float(frame.rgbaU8[fg + 2]) * a + Float(bg.b) * inv).rounded())
            out[base + 3] = 255
        }
        return makeRGBACGImage(from: out, h: frame.h, w: frame.w)
    }

    /// Render the prepared RGBA bytes as a `CGImage` so we can debug what
    /// the model sees vs what gets composited.
    static func rgbaToCGImage(_ bytes: [UInt8], h: Int, w: Int) -> CGImage? {
        makeRGBACGImage(from: bytes, h: h, w: w)
    }

    /// `alpha` is `[h * w]` half-floats in `[0, 1]`. Renders to grayscale `CGImage`.
    static func alphaToCGImage(_ alpha: [Float16], h: Int, w: Int) -> CGImage? {
        let n = h * w
        precondition(alpha.count == n)
        var bytes = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            let v = max(Float(0), min(Float(1), Float(alpha[i])))
            bytes[i] = UInt8(v * 255)
        }
        return makeGrayCGImage(from: bytes, h: h, w: w)
    }

    private static func makeGrayCGImage(from bytes: [UInt8], h: Int, w: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    private static func makeRGBACGImage(from bytes: [UInt8], h: Int, w: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        let info = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        return CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
