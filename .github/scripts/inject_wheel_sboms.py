"""Inject CycloneDX + SPDX SBOMs into a wheel's `.dist-info/sboms/` per PEP 770.

Reads the wheel, generates SBOMs from its unpacked contents via `syft`
(which understands the `cargo-auditable` section embedded in the
bundled Rust dylib), drops them into the wheel's metadata directory,
and re-packs the wheel.  `wheel pack` regenerates RECORD so the new
files are properly listed and hashed.

Usage:  python inject_wheel_sboms.py wheel-1.whl wheel-2.whl ...
        (in-place; replaces each input wheel with the SBOM-bearing one)

Requires:  `syft` on PATH, and the `wheel` Python package installed.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def inject(wheel_path: Path) -> None:
    wheel_path = wheel_path.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        unpack_root = tmp / "unpacked"
        repack_root = tmp / "repacked"
        unpack_root.mkdir()
        repack_root.mkdir()

        subprocess.check_call(
            [sys.executable, "-m", "wheel", "unpack", "-d", str(unpack_root), str(wheel_path)]
        )

        # `wheel unpack` writes one top-level dir named `<name>-<version>`
        (unpacked,) = list(unpack_root.iterdir())
        (dist_info,) = list(unpacked.glob("*.dist-info"))

        sboms_dir = dist_info / "sboms"
        sboms_dir.mkdir(exist_ok=True)

        # syft scans the unpacked tree.  Its rust-audit-binary cataloger
        # reads the `.dep-v0` ELF/Mach-O section that `cargo-auditable`
        # embedded; the Python cataloger picks up METADATA.
        subprocess.check_call(
            [
                "syft",
                "scan",
                f"dir:{unpacked}",
                "--source-name",
                unpacked.name,
                "-o",
                f"cyclonedx-json={sboms_dir / 'sbom.cdx.json'}",
                "-o",
                f"spdx-json={sboms_dir / 'sbom.spdx.json'}",
            ]
        )

        # `wheel pack` rewrites RECORD with hashes for every file under
        # `unpacked/`, including the two SBOMs we just added.
        subprocess.check_call(
            [sys.executable, "-m", "wheel", "pack", "-d", str(repack_root), str(unpacked)]
        )

        (repacked_wheel,) = list(repack_root.glob("*.whl"))
        # Names should match; if `wheel pack` produced a different
        # filename (e.g. build-tag difference), prefer the new name.
        target = wheel_path.parent / repacked_wheel.name
        if target != wheel_path:
            wheel_path.unlink()
        shutil.move(str(repacked_wheel), str(target))
        print(f"injected SBOMs into {target.name}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for w in argv:
        inject(Path(w))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
