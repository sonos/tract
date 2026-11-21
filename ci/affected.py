#!/usr/bin/env python3
"""Decides which PR CI legs a change can affect.

A leg is a CI job (or matrix entry) seen as what it builds or runs: workspace packages
on a target triple, with the features it exercises. Its inputs are the workspace crates
`cargo tree` finds in that closure, plus some paths of its own. A change affects the leg
when it touches one of them. Anything this script cannot place affects every leg.

Cargo.lock and the root Cargo.toml count by content against the base revision: a lockfile
change affects the legs whose dependency tree holds a changed package, and a manifest
change confined to `workspace.members` affects none by itself. Either one affects every
leg when the base cannot be read or the change cannot be explained that way.

linalg arch trees (`linalg/{arm32,arm64,x86_64,riscv64}`, `linalg/src/wasm`) only count
for legs that compile them: legs on that target_arch, and legs whose build enables
`foreign-inventory` (the tract-cli builds on x86_64 and aarch64-apple-darwin compile every
tree, though they only run their own). The GPU bench legs ignore them: they time the GPU
runtime, not the CPU kernels.

Usage: affected.py --files CHANGED [--base REV] [--json OUT] [--summary OUT]
CHANGED lists one changed path per line, relative to the repository root. REV defaults to
HEAD's first parent: the PR base when HEAD is the PR merge commit.
"""

import argparse
import json
import re
import subprocess
import sys
import tomllib
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
    "ci/llm-*",
    "ci/llm-expectations-*/**",
    "ci/asan.sh",
    "ci/cli-tests.sh",
    "ci/regular-tests.sh",
    "ci/setup-sccache.sh",
    "ci/test-harness.sh",
    "ci/test-llm.sh",
    "ci/test-rt.sh",
    "ci/web-e2e/**",
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
    "rust-toolchain.toml",
    ".cargo/**",
    ".github/scripts/**",
    ".github/actions/**",
    "ci/**",
]

NOT_GLOBAL = [
    "ci/affected.py",
    ".github/scripts/pr-tripwire.js",
    "ci/docker-debian-stretch/**",
    "ci/benches.toml",
    "ci/bench-*",
    "ci/cargo-deny-check.sh",
    "ci/android-ndk.sh",
]

SCRIPT_WORKFLOWS = {
    ".github/scripts/ci-feedback.js": ["cli.yml"],
    "ci/cross.sh": ["cli.yml", "cross-platform.yml"],
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

# Sources compiled only into a crate's own tests or benches (out-of-line `#[cfg(test)]`
# modules, plus every package's tests/ and benches/): they count for dev legs only.
TEST_ONLY = [
    "core/src/ops/einsum/proptest.rs",
    "linalg/src/frame/mmm/tests/**",
    "transformers/src/ops/test_utils.rs",
    "transformers/src/ops/moe_ffn/tests.rs",
]
PACKAGE_TEST_DIRS = ["tests/**", "benches/**"]

# The GPU runtimes' crates. tract-cli links some of them in every build, but a CPU bench
# never runs their code.
GPU_CRATES = frozenset(["tract-gpu", "tract-cuda", "tract-metal", "tract-wgpu"])

# The model frontends tract-cli links. A bench only loads the frontends of the models in
# ci/benches.toml, so the others never run in a bench leg.
FRONTENDS = {"onnx": "tract-onnx", "tf": "tract-tensorflow", "tflite": "tract-tflite"}


def bench_format(model):
    """The frontend bench-suite loads `model` with (format_hint in cli/src/bench_suite.rs)."""
    m = model.lower()
    if m.endswith(".onnx"):
        return "onnx"
    if m.endswith(".tflite"):
        return "tflite"
    if m.endswith((".tgz", ".tar", ".tar.gz")):
        return "nnef"
    return "tf"


def unbenched_frontends():
    manifest = tomllib.loads((ROOT / "ci" / "benches.toml").read_text())
    used = {bench_format(b["model"]) for b in manifest.get("bench", []) if "model" in b}
    return frozenset(crate for fmt, crate in FRONTENDS.items() if fmt not in used)


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
    exclude: frozenset = frozenset()
    cpu_kernels: bool = True

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
    if runner == "cuda-tests":
        return "x86_64-unknown-linux-gnu", f"{CPU},cuda-13000,wgpu"
    if runner == "bench-i9-11900kb-rtx-4060":
        return "x86_64-unknown-linux-gnu", f"{CPU},cuda-13000"
    if runner in ("macos-latest", "metal-tests", "bench-apple-m1-max"):
        return "aarch64-apple-darwin", f"{CPU},metal"
    raise ValueError(f"no CLI leg for runner {runner}")


def cli_spec(features, default_features=False):
    return Spec("tract-cli", features, default_features)


def legs():
    out = []
    unbenched = unbenched_frontends()

    def leg(workflow, key, triple, specs, **kw):
        out.append(Leg(workflow, key, triple, specs, **kw))

    cli = "cli.yml"
    leg(cli, "build x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu",
        [cli_spec("bench-suite", True)], foreign_inventory=True)
    leg(cli, "build x86_64-unknown-linux-gnu-wgpu", "x86_64-unknown-linux-gnu",
        [cli_spec("bench-suite,wgpu", True)], foreign_inventory=True)
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
    # apple-m1-max_metal is the metal runtime of the apple-m1-max box, benched in the same leg.
    for box, triple, features in [
        ("apple-m1-max", "aarch64-apple-darwin", CPU),
        ("apple-m1-max_metal", "aarch64-apple-darwin", f"{CPU},metal"),
        ("i9-11900kb_rtx-4060", "x86_64-unknown-linux-gnu", CPU),
        ("i9-11900kb_rtx-4060_cuda", "x86_64-unknown-linux-gnu", f"{CPU},cuda-13000"),
        ("i9-11900kb_rtx-4060_wgpu", "x86_64-unknown-linux-gnu", f"{CPU},wgpu"),
        ("i9-9900k", "x86_64-unknown-linux-gnu", CPU),
        ("jetson-orin-nx", "aarch64-unknown-linux-gnu", f"{CPU},cuda-12000"),
    ]:
        leg(cli, f"bench {box}", triple, [cli_spec(features)], paths=BENCH_PATHS,
            exclude=unbenched | (frozenset() if features != CPU else GPU_CRATES),
            cpu_kernels=features == CPU)
    for device, triple in [
        ("cortex-a55", "aarch64-unknown-linux-gnu"),
        ("cortex-a53", "aarch64-unknown-linux-gnu"),
        ("cortex-a7", "armv7-unknown-linux-gnueabihf"),
        ("cortex-a9", "armv7-unknown-linux-gnueabihf"),
        ("beaglev-ahead", "riscv64gc-unknown-linux-musl"),
        ("orangepi-rv2", "riscv64gc-unknown-linux-musl"),
    ]:
        leg(cli, f"bench {device}", triple, [cli_spec(DINGHY_CLI)], paths=BENCH_PATHS,
            exclude=unbenched | GPU_CRATES)

    crates = "crates.yml"
    host = "x86_64-unknown-linux-gnu"
    for crate in ["tract-data", "tract-linalg", "tract-core", "tract-nnef", "tract-hir",
                  "tract-onnx", "tract-pulse", "tract-onnx-opl", "tract-pulse-opl", "tract",
                  "test-unit-core", "test-pulse-core", "test-onnx-core", "test-nnef-cycle",
                  "test-f16"]:
        leg(crates, f"ubuntu-24.04 / {crate}", host, [Spec(crate, dev=True)])
    leg(crates, "cuda", host, [Spec(p, dev=True) for p in ["tract-gpu", "tract-cuda", "test-cuda"]])
    leg(crates, "wgpu", host, [Spec(p, dev=True) for p in ["tract-wgpu", "test-wgpu"]])
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
        leg(cross, f"linux ({platform})", rustc, [Spec("tract-linalg"), Spec("tract-ffi")],
            paths=["ci/android-ndk.sh"])
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


def closure(spec, triple):
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
        _closures[key] = {line.split()[0] for line in tree.splitlines() if line.strip()}
    return _closures[key]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def first_parent():
    head = git("cat-file", "-p", "HEAD").stdout
    return next((l.split()[1] for l in head.splitlines() if l.startswith("parent ")), None)


def at_base(base, path):
    """`path` as of `base`, fetching `base` into a shallow checkout; None if unreadable."""
    if base is None:
        return None
    shown = git("show", f"{base}:{path}")
    if shown.returncode != 0 and git("fetch", "-q", "--depth=1", "--no-tags", "origin", base).returncode == 0:
        shown = git("show", f"{base}:{path}")
    return shown.stdout if shown.returncode == 0 else None


def lock_changes(base):
    """Names of the Cargo.lock packages added, removed or edited since `base`; None when
    that cannot be told, or when nothing changed (a listed but unchanged lockfile is the
    caller's way of forcing every leg)."""
    old = at_base(base, "Cargo.lock")
    if old is None:
        return None
    try:
        entries = lambda text: {json.dumps(p, sort_keys=True) for p in tomllib.loads(text).get("package", [])}
        changed = entries(old) ^ entries((ROOT / "Cargo.lock").read_text())
    except tomllib.TOMLDecodeError:
        return None
    return {json.loads(e)["name"] for e in changed} or None


def members_only_manifest_change(base):
    old = at_base(base, "Cargo.toml")
    if old is None:
        return False

    def without_members(text):
        doc = tomllib.loads(text)
        doc.get("workspace", {}).pop("members", None)
        return doc

    try:
        return without_members(old) == without_members((ROOT / "Cargo.toml").read_text())
    except tomllib.TOMLDecodeError:
        return False


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
    return best if best and best[0] != "." else None


def decide(files, base):
    dirs = package_dirs()
    members = set(dirs.values())
    all_legs = legs()
    claimed = [p for leg in all_legs if leg.claims for p in leg.paths]
    workflows = {leg.workflow for leg in all_legs}

    touched = []
    unplaced = []
    global_hits = []
    workflow_hits = {}
    locked = set()
    for f in files:
        if matches(f, IGNORED):
            continue
        if f == "Cargo.lock":
            changes = lock_changes(base)
            if changes is None:
                global_hits.append(f)
            else:
                locked = changes
            continue
        if f == "Cargo.toml":
            if not members_only_manifest_change(base):
                global_hits.append(f)
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
        found = owner(f, dirs)
        if found:
            pkg_dir, pkg = found
            arch = next((a for p, a in ARCH_TREES.items() if matches(f, [p])), None)
            test_only = matches(f, TEST_ONLY) or matches(f[len(pkg_dir) + 1:], PACKAGE_TEST_DIRS)
            touched.append((f, pkg, arch, test_only))
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
                tree = set()
                try:
                    for spec in leg.specs:
                        tree |= closure(spec, leg.triple)
                    tree -= leg.exclude
                    crates = tree & members
                except subprocess.CalledProcessError as e:
                    print(e.stderr, file=sys.stderr)
                    reason = f"cargo tree failed: {' '.join(e.cmd[2:])}"
                    touched_here = []
                else:
                    touched_here = touched
                dev = any(spec.dev for spec in leg.specs)
                for f, pkg, arch, test_only in touched_here:
                    if (test_only and not dev) or (arch is not None and not leg.cpu_kernels):
                        continue
                    if pkg in crates and (arch is None or leg.foreign_inventory or arch == leg.arch
                                          or (arch == "x86_64" and leg.arch == "x86")):
                        reason = f"{pkg}: {f}"
                        break
                if reason is None and locked & tree:
                    reason = f"Cargo.lock: {min(locked & tree)}"
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
    parser.add_argument("--base")
    parser.add_argument("--json")
    parser.add_argument("--summary")
    args = parser.parse_args()
    files = sorted({l.strip() for l in Path(args.files).read_text().splitlines() if l.strip()})
    results = decide(files, args.base or first_parent())
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
