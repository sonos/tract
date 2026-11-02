# demo.tract.rs

`index.html` is the landing page at <https://demo.tract.rs/>. Each demo is a static site built by
its own `examples/<example>/web/build.sh` into that directory's `dist/`, and served under a
prefix of the `tract-demo` R2 bucket:

| Prefix        | Demo                                        |
|---------------|---------------------------------------------|
| `/sd15/`      | [`stable-diffusion/web`](../stable-diffusion/web) |
| `/pixel-lab/` | [`pixel-lab/web`](../pixel-lab/web)         |
| `/hey-snips/` | [`hey-snips/web`](../hey-snips/web)         |
| `/nemotron/`  | [`nemo-nemotron35-streaming-asr/web`](../nemo-nemotron35-streaming-asr/web) |
| `/ppocr/`     | [`ppocr/web`](../ppocr/web)                 |

A new demo gets a card in `index.html` and a line here; its page links back to `../`.

## End-to-end checks

A demo with an `examples/<example>/web/e2e.json` is opened in headless Chrome on an Apple M1
runner (label `browser-apple-m1`) by `ci/web-e2e/run.sh`: every night on the commit under test,
from the sites the Examples workflow just built, and every morning on the deployed site
(`web-demos-live.yml`). The spec's `path` is the demo's prefix above; its steps drive the page with
JavaScript expressions and check what it shows. `ci/web-e2e/driver.py` documents the format. To
run them on a Mac, with the built sites in `<dir>/web-demo-<example>/`:

```sh
ci/web-e2e/run.sh local <dir> /tmp/e2e-out
ci/web-e2e/run.sh live https://demo.tract.rs/ /tmp/e2e-out
```

## Deploying

With `$R2` an rclone remote on the bucket's account:

```sh
rclone copy examples/web-demos/index.html $R2:tract-demo/ --checksum
rclone copy examples/<example>/web/dist $R2:tract-demo/<prefix> --checksum
```

Build a demo from a commit on main: the page links to its source at the commit it was built from.
Then run the Web demos live workflow with that commit, which checks every deployed page names it
and works:

```sh
gh workflow run web-demos-live.yml -f commit=<sha>
```
