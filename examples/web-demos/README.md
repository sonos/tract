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

Build a demo from a commit on main: the page links to its source at the commit it was built from.
Deploy with `deploy.sh`, `$R2` being an rclone remote on the bucket's account, and a Cloudflare
token allowed to purge the zone's cache:

```sh
R2=<remote> CF_API_TOKEN=<token> CF_ZONE_ID=<zone> examples/web-demos/deploy.sh <example>...
```

`all` deploys every demo and the landing page, `landing` the landing page alone. Set `DIST_ROOT` to
take the sites from the Examples workflow's `web-demo-<example>` artifacts instead of each `dist/`.

Cloudflare keeps every file of the site at the edge until its cache is purged, and browsers
revalidate each file on every load. An upload that skips the purge stays invisible, so go through
`deploy.sh`: it uploads with `Cache-Control: no-cache`, purges, then checks that every file is
served as built.

Then run the Web demos live workflow with that commit, which checks every deployed page names it
and works:

```sh
gh workflow run web-demos-live.yml -f commit=<sha>
```

## Releasing

A tract release redeploys every demo from its tag. Run the Examples workflow on the tag, deploy its
artifacts, then check the live site names the tag's commit:

```sh
gh workflow run examples.yml --ref v<version>
gh run download <run id> --pattern 'web-demo-*' --dir <dir>
DIST_ROOT=<dir> examples/web-demos/deploy.sh all
gh workflow run web-demos-live.yml -f commit=$(git rev-parse v<version>^{commit})
```
