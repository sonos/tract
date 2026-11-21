// Pre-approval tripwire for fork PRs: greps the added lines of a PR's diff for what a CI
// probe or exfiltration attempt looks like, and reports it as the `tripwire` commit status
// on the PR head plus a table in the run summary. It reads the diff as text through the
// API and never checks out, builds or runs PR code. A clean result is not a review: it
// only means none of the patterns below matched.

const BUILD_TIME_PATHS = [
  [/(^|\/)build\.rs$/, 'build script'],
  [/(^|\/)Cargo\.(toml|lock)$/, 'cargo manifest'],
  [/^\.cargo\//, 'cargo config'],
  [/(^|\/)rust-toolchain(\.toml)?$/, 'toolchain'],
  [/^\.github\//, 'CI'],
  [/^ci\//, 'CI'],
  [/\.(sh|bash|py|pl|rb|js|ps1)$/, 'script'],
  [/(^|\/)(Makefile|Dockerfile|justfile)$/, 'build file'],
];

const ANY_FILE = [
  ['cloud metadata', /169\.254\.169\.254|fd00:ec2::254|metadata\.google\.internal|computeMetadata|metadata\/instance|security-credentials|instance-identity/i],
  ['network fetch', /\b(curl|wget|ncat|socat)\b|\bnc\s+-|\/dev\/(tcp|udp)\/|Invoke-WebRequest|\bInvoke-RestMethod\b/],
  ['credential path', /\.ssh\/|\bid_(rsa|ed25519|ecdsa)\b|\.aws\/|\.docker\/config|\.kube\/config|\.netrc|\.git-credentials|\.npmrc|\.pypirc|\.cargo\/credentials|\.config\/gh\b|\.gnupg/],
  ['runner token', /ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST|ACTIONS_CACHE_URL|ACTIONS_RESULTS_URL|GITHUB_TOKEN|_authToken|\bsecrets\./],
  ['docker access', /docker\.sock|\bdocker\s+(ps|run|exec|cp|inspect|images)\b|\bkubectl\b/],
  ['obfuscation', /base64\s+(-d|--decode)|from_base64|b64decode|\batob\(|(\\x[0-9a-fA-F]{2}){8}|[A-Za-z0-9+/]{160,}={0,2}/],
  ['shell eval', /\b(ba|z)?sh\s+-c\b|\|\s*(ba|z)?sh\b|Command::new\(\s*"[^"]*\b(ba|z)?sh"/],
  ['host recon', /\b(hostname|whoami|uname\s+-a|ifconfig|ip\s+addr|printenv)\b|\$\(\s*id\s*(-|2>)|\/etc\/(passwd|shadow|hosts)|\bls\s+-la\s+["']?\$HOME/],
  ['CI sniffing', /env::var(_os)?\(\s*"(GITHUB_|RUNNER_|CI")|process\.env\.(GITHUB_|RUNNER_)/],
  ['invisible unicode', /[‪-‮⁦-⁩​-‍⁠﻿]/],
];

const BUILD_SCRIPT = [
  ['process spawn', /\bCommand::new\b|std::process/],
  ['networking', /std::net|TcpStream|UdpSocket|\breqwest\b|\bureq\b|\bhyper\b|\bcurl::/],
  ['file read outside the crate', /fs::read\w*\(\s*"\/|env::var(_os)?\(\s*"HOME"|home_dir\(/],
];

const MANIFEST = [
  ['build-time dependency', /^\s*\[(.*\.)?build-dependencies\]/],
  ['proc-macro crate', /proc-macro\s*=\s*true/],
  ['build script entry', /^\s*build\s*=/],
  ['non-registry source', /\bgit\s*=|^\s*\[patch|\bregistry\s*=/],
];

const SHELL_FILE = [
  ['shell eval', /\beval\b/],
];

const LOCKFILE = [
  ['non-crates.io source', /^source\s*=\s*"(?!registry\+https:\/\/github\.com\/rust-lang\/crates\.io-index")/],
];

function rulesFor(path) {
  const rules = [...ANY_FILE];
  if (/(^|\/)build\.rs$/.test(path)) rules.push(...BUILD_SCRIPT);
  if (/(^|\/)Cargo\.toml$/.test(path)) rules.push(...MANIFEST);
  if (/(^|\/)Cargo\.lock$/.test(path)) rules.push(...LOCKFILE);
  if (/\.(sh|bash|ya?ml)$|(^|\/)(Makefile|Dockerfile|justfile)$/.test(path)) rules.push(...SHELL_FILE);
  return rules;
}

// Added lines of a unified diff hunk list, with their line number in the new file.
function addedLines(patch) {
  const out = [];
  let line = 0;
  for (const raw of patch.split('\n')) {
    const hunk = raw.match(/^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunk) { line = Number(hunk[1]); continue; }
    if (raw.startsWith('+')) { out.push([line, raw.slice(1)]); line++; }
    else if (!raw.startsWith('-') && !raw.startsWith('\\')) line++;
  }
  return out;
}

// `files` is the `pulls.listFiles` payload. Returns the build-time paths touched, the
// files without a text diff (binary or too large), and one finding per (added line, rule)
// match. A build-time file without a text diff is itself a finding.
function scan(files) {
  const buildTime = [];
  const opaque = [];
  const findings = [];
  for (const f of files) {
    const kind = BUILD_TIME_PATHS.find(([re]) => re.test(f.filename));
    if (kind) buildTime.push(`${f.filename} (${kind[1]})`);
    if (f.status === 'removed') continue;
    if (typeof f.patch !== 'string') {
      opaque.push(f.filename);
      if (kind) findings.push({ file: f.filename, line: 0, rule: 'build-time file without a text diff', text: '' });
      continue;
    }
    const rules = rulesFor(f.filename);
    for (const [line, text] of addedLines(f.patch)) {
      for (const [rule, re] of rules) {
        if (re.test(text)) findings.push({ file: f.filename, line, rule, text });
      }
    }
  }
  return { buildTime, opaque, findings };
}

function cell(s) {
  const t = s.trim().replace(/[\u0000-\u001f\u007f‪-‮⁦-⁩]/g, '?');
  return '`' + (t.length > 160 ? t.slice(0, 160) + '…' : t).replace(/[`|]/g, "'") + '`';
}

function report({ buildTime, opaque, findings }) {
  const lines = [];
  lines.push(findings.length
    ? `## ⚠️ Tripwire: ${findings.length} finding(s)`
    : '## Tripwire: no pattern matched');
  lines.push('', 'A clean result is not a review. Read build-time code before **Approve and run**.');
  if (buildTime.length) lines.push('', '**Build-time paths touched:**', ...buildTime.map(p => `- \`${p.replace(/`/g, "'")}\``));
  if (opaque.length) lines.push('', '**No text diff (binary or too large), not scanned:**', ...opaque.map(p => `- \`${p.replace(/`/g, "'")}\``));
  if (findings.length) {
    lines.push('', '| file:line | rule | added line |', '|---|---|---|');
    for (const f of findings.slice(0, 200)) lines.push(`| \`${f.file.replace(/[`|]/g, "'")}:${f.line}\` | ${f.rule} | ${cell(f.text)} |`);
    if (findings.length > 200) lines.push('', `…and ${findings.length - 200} more.`);
  }
  return lines.join('\n');
}

function describe({ buildTime, opaque, findings }) {
  const parts = [];
  if (findings.length) parts.push(`${findings.length} finding(s): ${[...new Set(findings.map(f => f.rule))].join(', ')}`);
  else parts.push('no pattern matched');
  if (opaque.length) parts.push(`${opaque.length} opaque file(s)`);
  if (buildTime.length) parts.push('touches build-time code');
  const d = parts.join('; ');
  return d.length > 140 ? d.slice(0, 139) + '…' : d;
}

async function run({ github, context, core }) {
  const pr = context.payload.pull_request;
  const files = await github.paginate(github.rest.pulls.listFiles, {
    owner: context.repo.owner, repo: context.repo.repo, pull_number: pr.number, per_page: 100,
  });
  const result = scan(files);
  await core.summary.addRaw(report(result)).write();
  await github.rest.repos.createCommitStatus({
    owner: context.repo.owner, repo: context.repo.repo, sha: pr.head.sha,
    state: result.findings.length ? 'failure' : 'success',
    context: 'tripwire',
    target_url: `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
    description: describe(result),
  });
}

module.exports = { scan, report, describe, run };
