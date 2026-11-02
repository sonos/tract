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

## Deploying

With `$R2` an rclone remote on the bucket's account:

```sh
rclone copy examples/web-demos/index.html $R2:tract-demo/ --checksum
rclone copy examples/<example>/web/dist $R2:tract-demo/<prefix> --checksum
```

Build a demo from a commit on main: the page links to its source at the commit it was built from.
