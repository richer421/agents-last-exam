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
