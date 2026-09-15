# Intel SDE Setup for ACE Testing

This document describes how to set up Intel SDE (Software Development Emulator) for testing ACE (AI Compute Extensions) instructions in tract.

## Background

ACE is a new x86 instruction set extension for AI matrix operations, jointly developed by Intel and AMD through the x86 Ecosystem Advisory Group. Hardware supporting ACE is not expected before ~2028, but Intel SDE 10.13.1+ includes ACE emulation via the `-future-ag` flag.

## Current Limitations

As of 2026-09-15, there are significant limitations for ACE SDE testing:

1. **No macOS SDE with ACE**: Intel SDE 10.13.1 (with ACE support) is only available for Linux and Windows, not macOS
2. **ARM64 Mac complication**: The development environment is an ARM64 Mac, but Intel SDE requires x86_64
3. **No assembler support**: No current toolchain (binutils, LLVM) can encode ACE instructions
4. **Build probe fails**: The `cfg(tract_ace)` build probe intentionally fails on all current toolchains

## Recommended Setup Process

### Option 1: x86_64 Linux VM (Recommended for Full Testing)

1. **Install virtualization software**:
   - UTM (free, open-source) - already installed on the development machine
   - Parallels Desktop (commercial)
   - VMware Fusion (commercial)

2. **Create x86_64 Linux VM**:
   - Ubuntu 22.04 LTS recommended
   - Allocate sufficient resources (4GB+ RAM, 20GB+ disk)
   - Enable AVX-512 support in VM settings

3. **Install build tools in VM**:
   ```bash
   sudo apt update
   sudo apt install build-essential gcc nasm binutils
   ```

4. **Download Intel SDE**:
   ```bash
   wget https://downloadmirror.intel.com/813591/sde-external-10.13.1-2026-07-28-lin.tar.xz
   tar xf sde-external-10.13.1-2026-07-28-lin.tar.xz
   export PATH=$PATH:$PWD/sde-external-10.13.1-2026-07-28-lin
   ```

5. **Verify SDE installation**:
   ```bash
   sde --version
   sde -future-ag -- echo "ACE support available"
   ```

### Option 2: Docker Container (Alternative)

For x86_64 hosts with Docker:

```bash
docker pull ubuntu:22.04
docker run -it ubuntu:22.04
# Then follow steps 3-5 from Option 1 inside the container
```

## ACE SDE Testing Workflow

Once SDE is set up:

1. **Write ACE test programs**:
   - Create small assembly files with ACE mnemonics
   - Use the specification-defined instruction names (TOP4BSSD, TOP2BF16PS, etc.)

2. **Compile with ACE-enabled assembler**:
   - Currently not possible (no assembler support)
   - Future: Use binutils with ACE support or LLVM with ACE intrinsics

3. **Run under SDE**:
   ```bash
   sde -future-ag -- ./ace_test_program
   ```

4. **Compare with software model**:
   - Use the differential testing framework in `src/ace/sde_tests.rs`
   - Compare SDE output against the portable software model in `src/ace/isa.rs`

## Current Status

The tract ACE implementation includes:

✅ **Ready for SDE validation**:
- Portable software model of all ACE v1 instructions
- Comprehensive test suite (157 tests)
- Differential testing infrastructure (`sde_tests.rs`)
- Real assembly kernel files with ACE mnemonics

⏳ **Waiting for toolchain support**:
- Intel SDE ACE emulation available but not accessible on current development platform
- No assembler can encode ACE instructions yet
- Build probe fails intentionally until toolchain support exists

🔧 **Prepared for future**:
- Assembly kernels (`ace_int8.S`, `ace_bf16.S`, `ace_mxfp8.S`, `ace_mxint8.S`)
- Build probe mechanism (`assembler_supports_ace()`)
- Runtime detection (`has_ace()`)
- Swap points in kernels for one-line transition to real instructions

## Integration with CI

Once SDE testing is feasible, consider adding to CI:

```yaml
# Example GitHub Actions step (conceptual)
- name: Test ACE with Intel SDE
  run: |
    wget https://downloadmirror.intel.com/813591/sde-external-10.13.1-2026-07-28-lin.tar.xz
    tar xf sde-external-10.13.1-2026-07-28-lin.tar.xz
    export PATH=$PATH:$PWD/sde-external-10.13.1-2026-07-28-lin
    cargo test -p tract-linalg --test ace_sde_tests
```

## References

- [Intel SDE Download](https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html)
- [Intel SDE Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-software-development-emulator-release-notes.html)
- [ACE Specification v1.15](https://x86ecosystem.org/wp-content/uploads/2026/06/ACE_v1_Specification_public_1_15.pdf)
- [x86 Ecosystem Advisory Group](https://x86ecosystem.org/)

## Notes

- SDE is for functional testing only, not performance measurement
- SDE runs programs much slower than native hardware
- The `-future-ag` flag enables ACE emulation in SDE 10.13.1+
- ACE uses the AMX framework, so requires AVX-512 baseline support