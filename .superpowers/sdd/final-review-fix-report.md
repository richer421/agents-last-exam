# Final Review Remediation Report

Date: 2026-07-29

Worktrees:

- ALE: `/Users/richer/.config/superpowers/worktrees/ale/atomic-solve-evaluate`
- anno-kit: `/Users/richer/.config/superpowers/worktrees/anno-kit/atomic-ale-sdk`

## Outcome

All seven final-review findings were remediated with focused RED/GREEN tests. The
atomic ALE suite, affected legacy task-data/sandbox tests, anno-runner SDK/context
tests, targeted lint/format checks, and the anno-runner Dockerfile static check
pass. No paid VM was provisioned and no real OSS object was successfully written.

The anno-runner checkpoint is:

- `8769b06 fix(runner): bound atomic ALE deadlines`

The ALE changes and this report are committed together in the ALE checkpoint.

## Remediations

### 1. Runner deadline and cleanup

- The outer deadline is configurable and defaults to 18,300 seconds: the
  18,000-second ALE budget plus a 300-second cleanup grace.
- Deadline expiry sends `SIGINT` first so ALE can unwind
  `AtomicRuntime.finally`, waits the bounded grace, and hard-kills only if the
  process remains alive.
- Temporary request cleanup preserves the primary error and keeps diagnostics
  bounded.
- Tests cover long-running success, configurable deadlines, graceful interrupt,
  hard-kill fallback, cleanup errors, and one-command solve/evaluate APIs without
  real sleeping.

### 2. Solve VM has no OSS identity

- Atomic solve deep-copies the environment and clears the Aliyun RAM role and
  bucket-output capability before provider construction.
- Declared `inputFiles` and software are downloaded by trusted host tooling,
  audited with file/count/size/path bounds, archived, transferred once, and
  revalidated inside the VM.
- Baked task data remains supported. Unsupported remote backends fail closed.
- The Agent-side command path contains no `ossutil`, metadata endpoint, bucket
  identity, or submission identity.
- Declared outputs are pulled into a private host snapshot before hashing and
  uploading. Remote size/SHA metadata is verified and `manifest.json` is
  committed last.

### 3. Evaluator trust gate

- `EvaluateRequest` requires an external registry record path and its SHA-256.
- The frozen v1 `EvaluatorRegistryRecord` includes ready status, task/evaluator
  identities, rubric/reference hashes, SDK/Harbor/RewardKit versions, image,
  PR/CI provenance, and ready time.
- Before provider acquisition, evaluate validates the record bytes/hash/schema
  and identities, rejects records inside the task checkout, requires a clean
  repository, and checks that `evaluator_version` is an ancestor of local
  `main`.
- Evaluator code is materialized with bounded `git archive` extraction from the
  exact authorized commit. Runtime/evaluator paths use that link-free archive,
  not mutable working-tree bytes.

### 4. Strict input/reference availability

- Task-card `inputFiles` and `referenceFiles` are parsed with bounded JSON and
  canonical path checks.
- Reference manifest bytes are downloaded and SHA-256 validated on the trusted
  host. Each referenced object is size/hash validated before archive creation
  and revalidated inside the VM.
- Empty/skipped/partial staging, missing files, unsafe paths/symlinks, archive
  mismatch, and unverifiable manifests fail before evaluator invocation.
- Host preparation errors are normalized to stable `input` or `reference`
  categories while preserving bounded causes.

### 5. Solve idempotency

- Agent selection and stable config digest resolution happen before runtime
  acquisition.
- Solve reads the canonical manifest before provisioning. A matching committed
  identity returns `submitted` immediately; an occupied conflicting key fails
  closed.
- The lost-response retry test proves there is no second provider acquisition
  or Agent run.

### 6. Artifact snapshot integrity

- Publication no longer hashes a live Agent path and later uploads that mutable
  path.
- Hashing and uploading use the same private host-staged bytes.
- The mutation/TOCTOU regression proves manifest digest, uploaded bytes, and
  verified metadata remain identical even if the VM output changes after pull.

### 7. Complete v1 envelopes

- `ArtifactEntry` requires `media_type`; `SubmissionManifest` requires
  `status="submitted"`.
- Scored `EvaluationResult` requires complete submission/task/image/evaluator
  identity, rubric hash, and Harbor reward/details provenance.
- `infra_failed` requires complete identity plus bounded `error_category`,
  `error_detail`, and `attempt_id`; serialized JSON omits outcome, score, rubric,
  and Harbor fields.
- Invalid output requires exactly `0.0`.
- Existing canonical results and publication calls validate full identity,
  including rubric hash, before reuse/write.
- CLI operation failures now emit complete v1 infrastructure envelopes.

## Verification Evidence

ALE:

```text
uv run pytest $(rg --files tests/ale_run | rg '/test_atomic.*\.py$') -q
123 passed in 13.06s

uv run pytest tests/ale_run/test_oss_task_data.py \
  tests/ale_run/test_baked_in_sandbox.py \
  tests/ale_run/test_sandbox_gather.py \
  tests/ale_run/test_sandbox_evaluator.py -q
37 passed in 3.01s

uv run ruff check <22 changed Python files>
All checks passed!

uv run ruff format --check <22 changed Python files>
22 files already formatted

git diff --check
clean
```

anno-runner:

```text
PYTHONPATH=. <verified-python> -m pytest \
  tests/test_ale_atomic.py tests/test_context.py -q
28 passed in 0.38s

ruff check runner_sdk/ale_atomic.py tests/test_ale_atomic.py
All checks passed!

ruff format --check runner_sdk/ale_atomic.py tests/test_ale_atomic.py
2 files already formatted

docker --context=default buildx build --builder default --check --pull=false \
  --build-arg ALE_REF=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  -f anno-runner/Dockerfile .
Check complete, no warnings found.
```

Focused no-cost integration evidence includes:

- solve lost-response retry and identity conflict;
- solve provider configuration with no RAM role;
- host-only input staging and immutable output snapshot;
- evaluator registry rejection before provider acquisition;
- exact-commit evaluator materialization;
- missing/partial/hash-mismatched reference rejection before evaluator;
- canonical result full-identity rejection;
- graceful SDK interrupt and hard-kill fallback.

## Safety Notes

- No real paid VM or real OSS E2E was run.
- During legacy fixture migration, incomplete mocks allowed bounded `ossutil`
  attempts against fake bucket URLs. Stat calls and one diagnostic upload
  attempt returned `403`; no write succeeded. The fixtures were corrected to
  patch both `ale_run.atomic.host_oss.run_host_ossutil` and module-local imported
  aliases, and the final verification uses only local fake stores.
- The first Docker check used an isolated builder and failed while resolving
  Docker Hub metadata (`EOF`). Re-running the same static check with the default
  builder's local `python:3.12-slim` cache completed with no warnings.

## Round 2

### Status

All seven remaining final-review findings were remediated with task-scoped
RED/GREEN tests and independent spec/code-quality reviews. Every task review
finished approved with no open finding.

### 1. Authoritative Evaluator Registry Root

- Removed request-controlled registry paths and digests from `EvaluateRequest`.
- Resolve one canonical identity-digest record below absolute
  `ALE_EVALUATOR_REGISTRY_ROOT`.
- Walk every root component descriptor-relative with no symlink following, open
  the record nonblocking/no-follow, and require a bounded regular JSON record.
- Preserve schema, exact identity, clean repository, and main-ancestor checks.
- Outside self-signed records fail before runtime or provider acquisition.

Commits: `c2ac8d5`, `b0bc7b1`.

### 2. Exact Clean Solve Checkout

- Keep canonical manifest reuse before repository validation and provider entry.
- Reject tracked, ordinary untracked, and ignored fresh-Solve repository state.
- Validate canonical task paths, disable Git replacement objects, verify the
  exact requested commit, and materialize bounded link-free Git bytes.
- Use the archive-backed copied request for runtime spec, TaskLoader, setup,
  prompt, runtime, and publication paths while retaining canonical identity.
- Preserve Evaluate's accepted ignored-cache behavior.

Commits: `98d0b62`, `a525ec9`, `5a9796f`.

### 3. Bounded Artifact Downloads

- Preflight the complete inspector snapshot before the first transfer.
- Stream VM files directly to private host files through bounded ranged reads;
  atomic publication never calls `download_to_local`.
- Enforce one monotonic whole-operation deadline around every range await,
  detect short-read stalls, overruns, remote-size drift, and late responses, and
  remove partial files on failure or cancellation.
- Recompute final size and SHA-256 before immutable host upload.

Commits: `0782ff0`, `54273a2`.

### 4. Real Harbor Evidence Publication

- Bounded-range retrieve `/logs/verifier/reward.json` and
  `/logs/verifier/reward-details.json` into private host staging.
- Enforce inclusive 32 KiB reward, 8 MiB details, and 8 MiB combined limits.
- Require strict finite JSON objects, exact identity types, and consistent
  score/outcome/hard-gate state in both objects.
- Publish reward, details, then canonical result through trusted host OSS.
  Successful and conflicting evidence writes are verified from downloaded
  remote bytes.
- Canonical replay revalidates both exact evidence objects before returning a
  score. The compact result contains evidence sizes, paths, digests, and bounded
  reward, never the full raw evaluator result.

Commits: `07ea780`, `5af7537`, `1d8a328`.

### 5. Runner Process Groups And Bounded Logs

- Drain stdout and stderr concurrently while retaining bounded diagnostics plus
  an explicit truncation sentinel.
- Signal the complete new-session process group with `SIGINT`, then `SIGKILL`
  after the existing 300-second grace, with only bounded waits and joins.
- Real subprocess tests cover high-volume dual-stream output, leader exit with a
  surviving descendant, full group termination, and leak-proof fixture cleanup.

anno-runner commits: `d279e70`, `53b3f29`.

### 6. Clean Input And Software Replacement

- Validate and fully extract trusted archives into a private base-local tree
  before touching live roots.
- Replace `input/` and `software/` as a tracked two-root transaction; omitted
  software removes stale software, while missing input fails before mutation.
- Rollback tracks moved and installed roots independently, attempts every
  recovery action, and preserves/reports backups if restoration is incomplete.
- Only an exact trusted success response authorizes host deletion of the nonce
  recovery path. Checked platform-specific cleanup observes nonzero results and
  never masks a primary error.

Commits: `0539b7f`, `dc16121`, `df0c0b0`.

### 7. Metadata-First OSS Staging

- Verified the installed `ossutil v1.7.18` pagination and long-listing contract.
- List all input and software metadata in bounded 16-object pages before any
  transfer; reject malformed output, pagination regress, duplicates, prefix
  mismatch, unsafe paths, namespace collisions, overlong components, and exact
  count/per-file/combined limit violations.
- Download only exact requester-pays object URLs. No production prefix `sync`
  remains.
- Recheck directory identity, symlink state, local size, and cumulative actual
  bytes after every copy. All runner and host filesystem failures normalize to
  the stable `input` category.
- Temporary cleanup runs exactly once; cleanup-only errors are categorized, and
  cleanup errors cannot mask staging or consumer failures.

Commits: `7f3e007`, `822beac`, `7e01a8f`.

### Round 2 Verification Evidence

ALE:

```text
uv run pytest $(rg --files tests/ale_run | rg '/test_atomic.*\.py$') -q
349 passed in 25.83s

uv run pytest tests/ale_run/test_oss_task_data.py \
  tests/ale_run/test_baked_in_sandbox.py \
  tests/ale_run/test_sandbox_gather.py \
  tests/ale_run/test_sandbox_evaluator.py -q
37 passed in 2.91s

uv run ruff check <18 round-2 Python files excluding the legacy sandbox contract>
All checks passed!

uv run ruff format --check <same 18 files>
18 files already formatted

uv run ruff check ale_run/base_interface/sandbox.py \
  --ignore I001,UP035,SIM117,BLE001
All checks passed!

git diff --check
clean
```

anno-runner:

```text
.venv/bin/python -m pytest tests/test_ale_atomic.py tests/test_context.py -q
32 passed in 2.94s

ruff check anno-runner/runner_sdk/ale_atomic.py \
  anno-runner/tests/test_ale_atomic.py
All checks passed!

ruff format --check anno-runner/runner_sdk/ale_atomic.py \
  anno-runner/tests/test_ale_atomic.py
2 files already formatted

git diff --check
clean

docker --context=default buildx build --builder default --check --pull=false \
  --build-arg ALE_REF=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  -f anno-runner/Dockerfile .
Check complete, no warnings found.
```

### Round 2 Safety Notes

- No paid VM or live OSS E2E was run.
- One intermediate pre-migration Evaluate fixture reached the locally configured
  host `ossutil`; it returned HTTP 403, wrote no object, and provisioned no VM.
  Final verification is fully fake-backed.
- The installed `ossutil` help and matching v1.7.18 source were inspected
  locally. No listing or download contacted an OSS endpoint.
- `ale_run/base_interface/sandbox.py` retains unrelated full-file legacy Ruff
  and format debt. The Task 3 contract-only change passes the documented scoped
  check, and all other round-2 Python files pass strict Ruff and format checks.
