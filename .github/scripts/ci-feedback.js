// Feedback of a `/ci` command on its PR, shared by ci-command.yml (which posts the
// command comment) and the workflows it dispatches (which edit it and set the
// `/ci <target>` commit status). Loaded with `require` from a checkout of the running
// workflow's own commit (main for a `/ci` command), never from a PR head.
//
// A command comment is: the marker, a header line, a state line, then named sections,
// each between `<!-- section:NAME -->` and `<!-- /section:NAME -->`. Edits rewrite the
// state line and the sections they name and keep everything else.

const MARKER = '<!-- /ci -->';

// A heading, so a finished command stands out from a running one in the PR thread.
const STATES = {
  queued: '### ⏳ Queued',
  running: '### 🟡 Running',
  success: '### ✅ Done',
  failure: '### ❌ Failed',
  cancelled: '### ⚪ Cancelled',
};

function runUrl(context) {
  return `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
}

function stateLine(state, url) {
  const label = STATES[state] || state;
  return url ? `${label} · [run](${url})` : label;
}

function render({ header, state, sections }) {
  const lines = [MARKER, header, state];
  for (const [name, text] of sections) {
    lines.push('', `<!-- section:${name} -->`, text, `<!-- /section:${name} -->`);
  }
  return lines.join('\n');
}

function parse(body) {
  const lines = body.split('\n');
  const sections = new Map();
  const re = /<!-- section:([\w-]+) -->\n([\s\S]*?)\n<!-- \/section:\1 -->/g;
  for (const m of body.matchAll(re)) {
    sections.set(m[1], m[2]);
  }
  return { header: lines[1] || '', state: lines[2] || '', sections };
}

// Posts a command comment on PR `issue_number` and returns its id. `sections` is a
// list of [name, markdown] pairs; `url` links the state line to a run.
async function create(github, context, issue_number, { header, state = 'queued', url, sections = [] }) {
  const body = render({ header, state: stateLine(state, url), sections: new Map(sections) });
  const { data } = await github.rest.issues.createComment({
    owner: context.repo.owner, repo: context.repo.repo, issue_number, body,
  });
  return data.id;
}

// Sets the state line (linked to this run) and/or replaces the named sections of
// command comment `commentId`. Without a comment id (a run dispatched by hand rather
// than by `/ci`), posts a fresh comment headed `fallbackHeader` on `issueNumber`
// instead, or does nothing when that is unset too.
async function update(github, context, { commentId, issueNumber, fallbackHeader, state, sections = {} }) {
  const url = runUrl(context);
  if (!commentId) {
    if (!issueNumber) {
      return;
    }
    await create(github, context, Number(issueNumber), {
      header: fallbackHeader,
      state: state || 'running',
      url,
      sections: Object.entries(sections),
    });
    return;
  }
  const comment_id = Number(commentId);
  const { data } = await github.rest.issues.getComment({
    owner: context.repo.owner, repo: context.repo.repo, comment_id,
  });
  const current = parse(data.body || '');
  for (const [name, text] of Object.entries(sections)) {
    current.sections.set(name, text);
  }
  const body = render({
    header: current.header,
    state: state ? stateLine(state, url) : current.state,
    sections: current.sections,
  });
  await github.rest.issues.updateComment({
    owner: context.repo.owner, repo: context.repo.repo, comment_id, body,
  });
}

// Sets the `/ci <target>` commit status on `sha`, linked to this run. `state` is a
// commit status state, or `cancelled`, which is reported as `error`.
async function status(github, context, { sha, target, state, description }) {
  await github.rest.repos.createCommitStatus({
    owner: context.repo.owner, repo: context.repo.repo, sha,
    state: state === 'cancelled' ? 'error' : state,
    context: `/ci ${target}`, target_url: runUrl(context), description,
  });
}

// Overall result of a workflow's `needs` (JSON from `toJSON(needs)`): `failure` when a
// job failed, else `cancelled` when one was cancelled, else `success`, with one line per
// job. A successful `prepare` is left out of the lines.
function jobs(needsJson) {
  const icon = r => ({ success: '✅', skipped: '⏭️', cancelled: '⚪' }[r] || '❌');
  const entries = Object.entries(JSON.parse(needsJson))
    .filter(([k, v]) => !(k === 'prepare' && v.result === 'success'));
  const failed = entries.filter(([, v]) => !['success', 'skipped', 'cancelled'].includes(v.result));
  const cancelled = entries.some(([, v]) => v.result === 'cancelled');
  return {
    state: failed.length ? 'failure' : cancelled ? 'cancelled' : 'success',
    failed: failed.map(([k]) => k),
    text: entries.map(([k, v]) => `- ${icon(v.result)} \`${k}\`: ${v.result}`).join('\n'),
  };
}

// Overall result of this run's completed jobs, read from the API rather than from
// `needs`, so a job can summarize the run without depending on every other job: like
// `jobs`, with the failed jobs named as the run shows them (matrix legs included).
async function runJobs(github, context) {
  const all = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner: context.repo.owner, repo: context.repo.repo, run_id: context.runId,
    filter: 'latest', per_page: 100,
  });
  const done = all.filter(j => j.status === 'completed');
  const failed = done.filter(j => !['success', 'skipped', 'neutral', 'cancelled'].includes(j.conclusion));
  const cancelled = done.some(j => j.conclusion === 'cancelled');
  return {
    state: failed.length ? 'failure' : cancelled ? 'cancelled' : 'success',
    failed: failed.map(j => j.name),
  };
}

module.exports = { create, update, status, jobs, runJobs, runUrl };
