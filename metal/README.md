# tract-metal

## Updating Metal Flash Attention library

```
git clone https://github.com/philipturner/metal-flash-attention.git
cd metal-flash-attention

# for iOS
swift build.swift --platform iOS --xcode-path /Applications/Xcode.app
cp build/lib/libMetalFlashAttention.metallib path/to/tract/metal/src/kernels/libMetalFlashAttention-ios.metallib

# for MacOS
swift build.swift --platform macOS --xcode-path /Applications/Xcode.app
cp build/lib/libMetalFlashAttention.metallib path/to/tract/metal/src/kernels/libMetalFlashAttention-macos.metallib
```