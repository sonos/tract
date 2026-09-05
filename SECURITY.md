# Security Policy

## Reporting a vulnerability

Report vulnerabilities through GitHub's private vulnerability reporting:
<https://github.com/sonos/tract/security/advisories/new>.

Do not open an issue, a pull request or a discussion for a suspected
vulnerability. Public reports are moved to a private advisory before they are
looked at, which only delays the fix.

Please include the tract version, the crate and the entry point involved, and a
model or input that reproduces the problem. We aim to acknowledge a report
within ten working days.

If you are not sure which side of the rules below your finding falls on, report
it privately and let us decide.

tract has no bug bounty and offers no payment for reports.

## Supported versions

| Version | Supported                |
| ------- | ------------------------ |
| 0.23.x  | yes                      |
| 0.22.x  | yes, security fixes only |
| 0.21.x  | yes, security fixes only |
| < 0.21  | no                       |

Fixes ship as a new patch release on each supported line.

## Security model

A model file is a program, not data. tract executes the graph it is given, and
loading a model is equivalent to loading a plugin or a shared library the
application chose to ship. In tract's supported deployments the model is
supplied by the developer and travels with the application binary.

Inference inputs are the opposite: tensors handed to a compiled model at run
time are untrusted in essentially every deployment, and tract is expected to
withstand any tensor of a shape and type the model accepts.

## Supported surface

The supported API is the facade in `api/rs/src/lib.rs`, and the C and Python
bindings built on it. That is what applications are expected to use, and it is
the surface this policy covers.

Rust has no visibility level between "this crate" and "the world", so the
implementation crates — `tract-core`, `tract-linalg`, `tract-nnef`,
`tract-onnx`, the `-opl` crates and the rest — expose a large number of `pub`
items purely so that tract's own crates can call each other. They are
implementation detail. They carry no stability contract, they change between
patch releases, and several of them have preconditions that the facade
establishes and that nothing re-checks: unchecked constructors, kernels invoked
after shape and alignment have already been verified, and internal state that
is expected to come out of a validated model.

The `tract` command line tool is a developer and debugging tool. It is not
meant for production use, and findings against it are evaluated on that
premise: it is not excluded from this policy, but it is expected to be pointed
at models and files its operator chose.

Driving one of those items directly into an invalid state is not a supported
use, and doing so is not by itself a vulnerability. It becomes one as soon as
the same state is reachable through the facade, through the bindings, or from a
model file — and then it is scored under the rules below.

## Scope

Scored normally, no cap:

- Memory unsafety, arbitrary code execution or arbitrary file access reachable
  from attacker-controlled tensor input, with a model the developer controls.
  This includes the `unsafe` kernels in `linalg` reached through the ordinary
  inference path.

Accepted, fixed and disclosed, but capped at Medium severity whatever the
technical impact:

- The same classes of bug when they require the attacker to supply the model
  file — malformed or hostile ONNX, NNEF or TFLite input to a loader.

Exploiting the second class requires a deployment in which an untrusted party
provides the model, which is outside tract's documented usage. We do not accept
reporter-supplied CVSS vectors that assume attacker-controlled models, and we
will dispute third-party scores that do. In CVSS 4.0 terms these carry Attack
Requirements: Present.

If your deployment does accept models from untrusted sources, treat model
loading as executing untrusted code: run it in a separate process, with
restricted filesystem access and a memory limit.

## Not vulnerabilities

Report these as ordinary issues, in public:

- Panics, assertion failures and aborts on a malformed or hostile model. tract
  validates models loosely and a crash while loading one is a bug, not a
  security boundary being crossed.
- Memory exhaustion or unbounded run time caused by the shapes and operators a
  model asks for. A model is a program and may legitimately be an expensive one.
- Numerical differences from other runtimes, and accuracy or convergence
  problems.
- Advisories against a dependency with no path reachable from tract.
- Crashes or memory errors produced by calling an implementation crate directly,
  outside the facade described above, with arguments the facade would never
  produce. Show the same result through `api/rs`, through a binding, or with a
  model file, and it is in scope.

## Disclosure

We fix under embargo and disclose when the fix is released, or after 90 days,
whichever comes first. Advisories are published as GitHub Security Advisories,
which propagates them to the RustSec database and to `cargo audit`.

Using an assistant to write up the initial report is fine — the finding still
has to be yours. Everything after that is a conversation between people, and we
expect the author at the other end of it: answering a question about the
report, saying whether a proposed fix addresses it, telling us when what we have
asked does not make sense. Reports whose follow-up is machine-generated are
assessed and fixed on their merits, without credit in the advisory. We would
rather talk to the person who found the bug than prompt their model.
