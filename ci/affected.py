#!/usr/bin/env python3
"""Decides which PR CI legs a change can affect.

A leg is a CI job (or matrix entry) seen as what it builds or runs: workspace packages
on a target triple, with the features it exercises. Its inputs are the workspace crates
`cargo tree` finds in that closure, plus some paths of its own. A change affects the leg
when it touches one of them. Anything this script cannot place affects every leg.

linalg arch trees (`linalg/{arm32,arm64,x86_64,riscv64}`, `linalg/src/wasm`) only count
for legs that compile them: legs on that target_arch, and legs whose build enables
`foreign-inventory` (the tract-cli builds on x86_64 and aarch64-apple-darwin compile every
tree, though they only run their own).

Usage: affected.py --files CHANGED [--json OUT] [--summary OUT]
CHANGED lists one changed path per line, relative to the repository root.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CPU = "onnx,tf,pulse,pulse-opl,tflite,transformers,extra,bench-suite"
CROSS_CLI = "onnx,tf,pulse,pulse-opl,tflite,transformers,extra"
DINGHY_CLI = "onnx,tf,pulse,pulse-opl,transformers,extra,bench-suite"

IGNORED = [
    "**/*.md",
    "doc/**",
    "**/LICENSE*",
    "**/.gitignore",
    "**/*.png",
    "**/*.svg",
    "**/*.xlsx",
    "ci/benches-wgpu.toml",
    "ci/llm-*",
    "ci/llm-expectations-*/**",
    "ci/asan.sh",
    "ci/tflite/**",
    "ci/tflite.sh",
    "ci/onnx-tests.sh",
    "ci/feature-configs*.sh",
    "ci/test-published-crates.sh",
    "ci/native.sh",
    "harness/*/ci-full.sh",
    "tools/asset-mirror/**",
]

GLOBAL = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/**",
    ".github/scripts/**",
    ".github/actions/**",
    "ci/**",
]

NOT_GLOBAL = [
    "ci/affected.py",
    "ci/docker-debian-stretch/**",
    "ci/benches.toml",
    "ci/bench-*",
    "ci/cargo-deny-check.sh",
]

SCRIPT_WORKFLOWS = {
    ".github/scripts/ci-feedback.js": ["cli.yml"],
    "ci/cross.sh": ["cli.yml", "cross-platform.yml"],
    "ci/android-ndk.sh": ["cross-platform.yml"],
}

BENCH_PATHS = ["ci/benches.toml", "ci/bench-*", ".github/workflows/bench-comment.yml"]
STRETCH_PATHS = ["ci/docker-debian-stretch/**"]

ARCH_TREES = {
    "linalg/arm32/**": "arm",
    "linalg/src/arm32/**": "arm",
    "linalg/src/arm32.rs": "arm",
    "linalg/arm64/**": "aarch64",
    "linalg/src/arm64/**": "aarch64",
    "linalg/src/arm64.rs": "aarch64",
    "linalg/x86_64/**": "x86_64",
    "linalg/src/x86_64/**": "x86_64",
    "linalg/src/x86_64.rs": "x86_64",
    "linalg/riscv64/**": "riscv64",
    "linalg/src/riscv64/**": "riscv64",
    "linalg/src/riscv64.rs": "riscv64",
    "linalg/src/wasm/**": "wasm32",
}

ALL_ARCHES = "all"


@dataclass(frozen=True)
class Spec:
    package: str | None
    features: str = ""
    default_features: bool = True
    dev: bool = False


@dataclass
class Leg:
    workflow: str
    key: str
    triple: str
    specs: list[Spec]
    paths: list[str] = field(default_factory=list)
    foreign_inventory: bool = False
    path_only: bool = False
    claims: bool = True

    @property
    def arch(self):
        head = self.triple.split("-")[0]
        if head.startswith("arm") or head.startswith("thumb"):
            return "arm"
        if head in ("i686", "i586"):
            return "x86"
        if head.startswith("riscv64"):
            return "riscv64"
        return head


def glob_re(pattern):
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def matches(path, patterns):
    return any(glob_re(p).match(path) for p in patterns)


def harness_runners():
    cases = []
    for script in sorted((ROOT / "harness").glob("*/ci.sh")):
        runners = ["ubuntu-24.04"]
        for line in script.read_text().splitlines():
            if line.startswith("# ci-runners:"):
                runners = line.split(":", 1)[1].split()
                break
        cases.append((script.parent.name, runners))
    return cases


def runner_leg(runner):
    if runner in ("ubuntu-24.04",):
        return "x86_64-unknown-linux-gnu", CPU
    if runner in ("cuda-tests", "bench-i9-11900kb-rtx-4060"):
        return "x86_64-unknown-linux-gnu", f"{CPU},cuda-13000"
    if runner in ("macos-latest", "metal-tests", "bench-apple-m1-max"):
        return "aarch64-apple-darwin", f"{CPU},metal"
    raise ValueError(f"no CLI leg for runner {runner}")


def cli_spec(features, default_features=False):
    return Spec("tract-cli", features, default_features)


def legs():
    out = []

    def leg(workflow, key, triple, specs, **kw):
        out.append(Leg(workflow, key, triple, specs, **kw))

    cli = "cli.yml"
    leg(cli, "build x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu",
        [cli_spec("bench-suite", True)], foreign_inventory=True)
    leg(cli, "build aarch64-unknown-linux-gnu-stretch", "aarch64-unknown-linux-gnu",
        [cli_spec(f"{CROSS_CLI},bench-suite,cuda-12000")], paths=STRETCH_PATHS)
    leg(cli, "build-macos", "aarch64-apple-darwin",
        [cli_spec("bench-suite", True)], foreign_inventory=True)
    for triple, rustc, paths in [
        ("aarch64-unknown-linux-gnu-stretch", "aarch64-unknown-linux-gnu", STRETCH_PATHS),
        ("armv7-unknown-linux-gnueabihf-stretch", "armv7-unknown-linux-gnueabihf", STRETCH_PATHS),
        ("riscv64gc-unknown-linux-musl", "riscv64gc-unknown-linux-musl", []),
    ]:
        leg(cli, f"build {triple} (dinghy)", rustc, [cli_spec(DINGHY_CLI)], paths=paths)
    for case, runners in harness_runners():
        for runner in runners:
            triple, features = runner_leg(runner)
            leg(cli, f"harness/{case}/ci.sh / {runner}", triple, [cli_spec(features)],
                paths=[f"harness/{case}/**"])
    for box, triple, features in [
        ("apple-m1-max", "aarch64-apple-darwin", f"{CPU},metal"),
        ("i9-11900kb_rtx-4060", "x86_64-unknown-linux-gnu", f"{CPU},cuda-13000"),
        ("i9-9900k", "x86_64-unknown-linux-gnu", CPU),
        ("jetson-orin-nx", "aarch64-unknown-linux-gnu", f"{CPU},cuda-12000"),
    ]:
        leg(cli, f"bench {box}", triple, [cli_spec(features)], paths=BENCH_PATHS)
    for device, triple in [
        ("cortex-a55", "aarch64-unknown-linux-gnu"),
        ("cortex-a53", "aarch64-unknown-linux-gnu"),
        ("cortex-a7", "armv7-unknown-linux-gnueabihf"),
        ("cortex-a9", "armv7-unknown-linux-gnueabihf"),
        ("beaglev-ahead", "riscv64gc-unknown-linux-musl"),
        ("orangepi-rv2", "riscv64gc-unknown-linux-musl"),
    ]:
        leg(cli, f"bench {device}", triple, [cli_spec(DINGHY_CLI)], paths=BENCH_PATHS)

    crates = "crates.yml"
    host = "x86_64-unknown-linux-gnu"
    for crate in ["tract-data", "tract-linalg", "tract-core", "tract-nnef", "tract-hir",
                  "tract-onnx", "tract-pulse", "tract-onnx-opl", "tract-pulse-opl", "tract",
                  "test-unit-core", "test-pulse-core", "test-onnx-core", "test-nnef-cycle",
                  "test-f16"]:
        leg(crates, f"ubuntu-24.04 / {crate}", host, [Spec(crate, dev=True)])
    leg(crates, "cuda", host,
        [Spec(p, dev=True) for p in ["tract-gpu", "tract-cuda", "test-cuda", "tract-wgpu", "test-wgpu"]])
    leg(crates, "metal", "aarch64-apple-darwin",
        [Spec(p, dev=True) for p in ["tract-gpu", "tract-metal", "test-metal"]])
    leg(crates, "fmt, clippy, etc", host,
        [Spec(None, dev=True), cli_spec("bench-suite", True)],
        paths=["**/*.rs", "**/Cargo.toml", "rustfmt.toml", "clippy.toml", ".rustfmt.toml"],
        foreign_inventory=True, claims=False)
    leg(crates, "api/proxy/sys/tract.h matches tract-ffi", host, [],
        paths=["api/ffi/**", "api/proxy/sys/tract.h", "api/generate-tract-h.sh", "api/cbindgen.toml"],
        path_only=True)
    leg(crates, "cargo-deny", host, [],
        paths=["**/Cargo.toml", "deny.toml", "ci/cargo-deny-check.sh"], path_only=True)

    cross = "cross-platform.yml"
    leg(cross, "windows-arm64", "aarch64-pc-windows-msvc", [Spec("tract-linalg")])
    qemu = lambda: [Spec("tract-linalg", dev=True), Spec("tract-core", dev=True),
                    Spec("tract-ffi", default_features=False), cli_spec(CROSS_CLI)]
    for platform, rustc in [
        ("aarch64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"),
        ("armv6vfp-unknown-linux-gnueabihf", "arm-unknown-linux-gnueabihf"),
        ("armv7-unknown-linux-gnueabihf", "armv7-unknown-linux-gnueabihf"),
        ("aarch64-unknown-linux-musl", "aarch64-unknown-linux-musl"),
        ("cortexa53-unknown-linux-musl", "aarch64-unknown-linux-musl"),
        ("armv7-unknown-linux-musl", "armv7-unknown-linux-musleabihf"),
        ("riscv64gc-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"),
        ("rvv128-unknown-linux-gnu", "riscv64gc-unknown-linux-gnu"),
    ]:
        leg(cross, f"linux ({platform})", rustc, qemu())
    leg(cross, "linux (raspbian)", "arm-unknown-linux-gnueabihf",
        [Spec("tract-ffi", default_features=False), cli_spec(CROSS_CLI)])
    leg(cross, "linux (aarch64-unknown-linux-gnu-stretch)", "aarch64-unknown-linux-gnu",
        [Spec("tract-ffi", "cuda-12000", False), cli_spec(f"{CROSS_CLI},bench-suite,cuda-12000")],
        paths=STRETCH_PATHS)
    leg(cross, "linux (armv7-unknown-linux-gnueabihf-stretch)", "armv7-unknown-linux-gnueabihf",
        [Spec("tract-ffi", default_features=False), cli_spec(CROSS_CLI)], paths=STRETCH_PATHS)
    leg(cross, "linux (x86_64-unknown-linux-gnu-stretch)", host,
        [cli_spec("", True)], paths=STRETCH_PATHS, foreign_inventory=True)
    for platform, rustc in [
        ("aarch64-linux-android", "aarch64-linux-android"),
        ("armv7-linux-androideabi", "armv7-linux-androideabi"),
        ("i686-linux-android", "i686-linux-android"),
        ("x86_64-linux-android", "x86_64-linux-android"),
    ]:
        leg(cross, f"linux ({platform})", rustc, [Spec("tract-linalg"), Spec("tract-ffi")])
    wasm = [Spec("tract-onnx", "getrandom-js"), Spec("tract-tensorflow")]
    leg(cross, "linux (wasm32-unknown-unknown)", "wasm32-unknown-unknown", wasm)
    leg(cross, "linux (wasm32-unknown-emscripten)", "wasm32-unknown-emscripten", wasm)
    leg(cross, "linux (wasm32-wasi)", "wasm32-wasip1",
        wasm + [Spec(p, dev=True) for p in ["tract-linalg", "tract-core", "test-unit-core"]])
    leg(cross, "apple (aarch64-apple-ios)", "aarch64-apple-ios",
        [Spec("tract-linalg"), Spec("tract-ffi")])
    leg(cross, "apple (aarch64-apple-darwin)", "aarch64-apple-darwin",
        [cli_spec("", True)], foreign_inventory=True)

    return out


_closures = {}


def closure(spec, triple, members):
    key = (spec, triple)
    if key not in _closures:
        cmd = ["cargo", "tree", "--locked", "-q", "--target", triple, "--prefix", "none",
               "--format", "{p}", "-e", "normal,build" + (",dev" if spec.dev else "")]
        if spec.package:
            cmd += ["-p", spec.package]
        if not spec.default_features:
            cmd.append("--no-default-features")
        if spec.features:
            cmd += ["--features", spec.features]
        tree = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
        names = {line.split()[0] for line in tree.splitlines() if line.strip()}
        _closures[key] = names & members
    return _closures[key]


def package_dirs():
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    root = Path(meta["workspace_root"])
    dirs = {}
    for p in meta["packages"]:
        rel = Path(p["manifest_path"]).parent.relative_to(root).as_posix()
        dirs[rel] = p["name"]
    return dirs


def owner(path, dirs):
    best = None
    for d, name in dirs.items():
        if (d == "." or path == d or path.startswith(d + "/")) and (best is None or len(d) > len(best[0])):
            best = (d, name)
    return best[1] if best and best[0] != "." else None


def decide(files):
    dirs = package_dirs()
    members = set(dirs.values())
    all_legs = legs()
    claimed = [p for leg in all_legs if leg.claims for p in leg.paths]
    workflows = {leg.workflow for leg in all_legs}

    touched = []
    unplaced = []
    global_hits = []
    workflow_hits = {}
    for f in files:
        if matches(f, IGNORED):
            continue
        if f in SCRIPT_WORKFLOWS:
            for name in SCRIPT_WORKFLOWS[f]:
                workflow_hits.setdefault(name, []).append(f)
            continue
        if matches(f, GLOBAL) and not matches(f, NOT_GLOBAL):
            global_hits.append(f)
            continue
        if f.startswith(".github/workflows/"):
            name = f.rsplit("/", 1)[1]
            if name in workflows:
                workflow_hits.setdefault(name, []).append(f)
            continue
        pkg = owner(f, dirs)
        if pkg:
            arch = next((a for p, a in ARCH_TREES.items() if matches(f, [p])), None)
            touched.append((f, pkg, arch))
        elif not matches(f, claimed) and f != "ci/affected.py":
            unplaced.append(f)

    results = []
    for leg in all_legs:
        reason = None
        if global_hits:
            reason = f"global: {global_hits[0]}"
        elif unplaced:
            reason = f"unplaced: {unplaced[0]}"
        elif leg.workflow in workflow_hits:
            reason = f"workflow: {workflow_hits[leg.workflow][0]}"
        else:
            hit = next((f for f in files if matches(f, leg.paths)), None)
            if hit:
                reason = f"path: {hit}"
            elif not leg.path_only:
                crates = set()
                try:
                    for spec in leg.specs:
                        crates |= closure(spec, leg.triple, members)
                except subprocess.CalledProcessError as e:
                    print(e.stderr, file=sys.stderr)
                    reason = f"cargo tree failed: {' '.join(e.cmd[2:])}"
                    touched_here = []
                else:
                    touched_here = touched
                for f, pkg, arch in touched_here:
                    if pkg in crates and (arch is None or leg.foreign_inventory or arch == leg.arch
                                          or (arch == "x86_64" and leg.arch == "x86")):
                        reason = f"{pkg}: {f}"
                        break
        results.append({"workflow": leg.workflow, "leg": leg.key, "affected": reason is not None,
                        "reason": reason or ""})
    return results


def summary(results, nfiles):
    run = sum(r["affected"] for r in results)
    lines = [f"### CI scope (dry run): {run} of {len(results)} legs affected by {nfiles} changed files", ""]
    for wf in sorted({r["workflow"] for r in results}):
        rows = [r for r in results if r["workflow"] == wf]
        n = sum(r["affected"] for r in rows)
        lines += [f"<details><summary><b>{wf}</b>: {n}/{len(rows)} would run</summary>", "",
                  "| leg | would | because |", "|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['leg']} | {'run' if r['affected'] else 'skip'} | {r['reason']} |")
        lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True)
    parser.add_argument("--json")
    parser.add_argument("--summary")
    args = parser.parse_args()
    files = sorted({l.strip() for l in Path(args.files).read_text().splitlines() if l.strip()})
    results = decide(files)
    doc = {"files": files, "legs": results}
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1) + "\n")
    else:
        json.dump(doc, sys.stdout, indent=1)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(summary(results, len(files)))


if __name__ == "__main__":
    main()
