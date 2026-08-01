# Clothing Pattern Replacement: Blind E2E Data Sheet

## Acceptance result

The final acceptance command was run once with three concurrent, evaluator-blind
Codex agents and exited successfully:

```bash
TRUE_SOTA_API_KEY="$(security find-generic-password \
  -a richer -s codex-custom-provider-api-key -w)" \
  .venv/bin/python -m ale_run run \
  experiments/clothing_pattern_blind_e2e.yaml --verbose
```

Execution date: 2026-07-26 Asia/Shanghai (2026-07-25 UTC). All three units
completed the complete path without operator intervention:

```text
Aliyun Windows VM -> OSS input -> Codex gpt-5.6-sol -> Photoshop
-> VM-to-OSS output -> near-data evaluation -> VM deletion
```

| Agent | Run suffix | Status | Score | Duration | Eval duration | Origin logs |
|---|---|---:|---:|---:|---:|---|
| blind-1 | `214631-2eddf1a5` | completed | 0.813367534480 | 1255.42 s | 165.8625 s | complete, 12 files, 40,711,923 B |
| blind-2 | `214631-b62b7757` | completed | 0.795714386231 | 727.23 s | 184.0825 s | complete, 12 files, 28,539,228 B |
| blind-3 | `214631-6de2c7d7` | completed | 0.795714413053 | 1048.96 s | 191.4367 s | complete, 14 files, 43,048,493 B |

Mean score: `0.801598777921`; minimum: `0.795714386231`; maximum:
`0.813367534480`. Score variation is model/task stochasticity and must not be
interpreted as a framework regression delta without repeated paired trials.

## Evaluator measurements

| Metric | blind-1 | blind-2 | blind-3 |
|---|---:|---:|---:|
| alpha similarity | 0.967678823 | 0.969960555 | 0.969960555 |
| foreground SSIM | 0.950096488 | 0.892846584 | 0.892846584 |
| full-image SSIM | 0.983405590 | 0.983192205 | 0.983192205 |
| PSD render SSIM | 0.426058382 | 0.425795704 | 0.425795704 |
| PSD structure | 0.196078431 | 0.196078431 | 0.196078431 |
| clothing SSIM | 0.908383906 | 0.902829349 | 0.902829409 |
| non-clothing SSIM | 0.987059951 | 0.884000480 | 0.884000480 |
| clothing change ratio | 1.000000000 | 0.984175372 | 0.984175554 |
| failure reason | null | null | null |

The score is not a single-image similarity proxy. It combines foreground and
full-image agreement, clothing-region fidelity/change, preservation outside the
clothing region, rendered PSD agreement, alpha agreement, and editable PSD
structure. Hard failures remain explicit in `failure_reason`.

Evaluator confidence comes from three layers:

1. Unit tests cover valid results and deliberately damaged/missing artifacts.
2. Blind candidates do not receive evaluator source or grading criteria.
3. The evaluator reports its component metrics, so a high aggregate cannot hide
   a failed required dimension.

This is evidence of useful discrimination, not a mathematical proof that every
future high score is good. Ongoing calibration should retain positive/negative
golden samples, adversarial samples, human pairwise labels, rank correlation,
false-positive/false-negative review, and evaluator versioning. A score should
be accepted only with `eval_status=success`, `failure_reason=null`, and the
component report, not as an unqualified scalar.

## OSS artifacts

Base prefix:

```text
oss://ale-artifacts/平面设计/RsZzbytPFajraGsMkezc0R4sn0x/recvpkQlYsiGTh/runs/<run_id>/output/
```

| Agent | Artifact | Size | SHA-256 |
|---|---|---:|---|
| blind-1 | `final_result.png` | 3,188,551 B | `da8ac1704b7d0af4331b5e48f5917e91bd365f8c3e9291b944b21e8ffebb412c` |
| blind-1 | `final_result.psd` | 51,543,513 B | `090b719b82e665bc1ddb986e5650c163eea99bdfcaa2ed5af504a76e3ba84a78` |
| blind-2 | `final_result.png` | 3,072,245 B | `f693042a84e939d1cecd6172204e1830ec7b53d0262da5e3881d145c817ebba1` |
| blind-2 | `final_result.psd` | 50,560,233 B | `43bd0f80f98fe12ba7e2bd5f70623587a482d42e8ca3d782f3d11c313dbecfb0` |
| blind-3 | `final_result.png` | 3,072,295 B | `2cd703c023408dee28ed292bc46fac783ce6336a1b8d5dbcefa39f6e1c3bfc62` |
| blind-3 | `final_result.psd` | 50,560,229 B | `ba5dfab5a95431a62caa454d0b38e93c19d60e60f8d3f418a246f6aaadf6cb69` |

Each prefix also contains a 555-byte `artifact_manifest.json`. The manifest
sizes match the OSS listings. Host run directories contain no final PSD or PNG.
The host receives only sanitized run/evaluation JSON, screenshots, bounded logs,
and artifact metadata; source/reference and full outputs stay in the data
environment or OSS.

## Configuration

- Model: `gpt-5.6-sol`
- API base URL: `https://true-sota.com/v1`
- Agent preset: Codex, low reasoning effort, full sandbox access
- Provider: Aliyun ECS, `ap-southeast-1`
- VM family: `ecs.u1-c1m4.2xlarge`
- Display: 1024 x 768
- Task data: `oss://ale-artifacts`
- Output: `oss://ale-artifacts`
- Concurrency: 3
- Wall budget: 5,400 seconds per unit
- Cleanup: delete
- Evaluation: near-data sandbox worker

The API key is read from macOS Keychain and is redacted from `run.json`, events,
logs, errors, and provider output.

## Generic ALE capabilities exercised

- PSD/PNG/reference remain near data; evaluator returns bounded JSON/log data.
- VM output uploads directly to OSS with size/SHA manifest generation.
- Origin gather permits code/text/log data but rejects media and reports honest
  complete/partial state.
- Incremental tailing uses absolute deadlines, transport-level time budgets,
  JSONL commit boundaries, and stable-but-behind detection.
- Sandbox launch has atomic replay locking and scalar PID acknowledgement.
- Aliyun transient control-plane failures retry and redact credentials.
- Windows `ossutil` bootstraps automatically from a pinned version when absent:
  versioned cache, archive and executable SHA-256 checks, bounded extraction,
  and atomic concurrent publication. No operator environment variable is needed.
- Evaluator archive inputs are allowlisted and size limited; media, reference,
  output, secrets, recursive globs, and suspicious magic bytes fail closed.

These are framework capabilities. The only task-specific implementation is the
clothing-pattern evaluator and its metric policy.

## Failure chronology and resolution

| Observed failure | Root cause | Generic resolution |
|---|---|---|
| Provider EOF/TLS failures | transient Aliyun control-plane transport | idempotent retry and secret-safe diagnostics |
| Evaluator module launch failed | invalid `python -m __main__` packaging | fixed evaluator worker module protocol |
| Large PSD/media reached host | gather boundary did not enforce data locality | artifact policy and direct VM-to-OSS upload |
| Large OTel reconcile delayed completion | unbounded/incomplete incremental reconciliation | absolute deadline, bounded range calls, honest partial status |
| Launcher reported failure while Codex ran | four-byte PID file used slow file-download path | stdout PID acknowledgement and short command probe |
| First final canary failed before task setup | Windows image lacked `ossutil` and required a manual env var | verified automatic Windows bootstrap and versioned cache |

After the final bootstrap fix, a fresh three-unit `ale run` completed with exit
code 0. No task-specific workaround was added to ALE.

## Verification

Fresh local verification after the final changes:

```text
ALE related suite:       67 passed
Task evaluator suite:     7 passed
Reconcile focused review: 24 passed
ruff:                     passed
git diff --check:         passed
Python 3.12 compile:      passed
Independent reviews:      no remaining P0/P1
```

Operational verification:

- Final three-run `ale run`: exit 0, 3/3 completed.
- OSS listing: three objects per run (manifest, PNG, PSD).
- Host final media search: zero files.
- Aliyun `ale-blind*` instance query after cleanup: `TotalCount=0`.

## Reproduction inputs

- `experiments/clothing_pattern_blind_e2e.yaml`
- `configs/agents/codex_true_sota_blind_1.yaml`
- `configs/agents/codex_true_sota_blind_2.yaml`
- `configs/agents/codex_true_sota_blind_3.yaml`
- `configs/environments/environment_aliyun_adobe_oss_e2e.yaml`

The complete host evidence for each run is under:

```text
/Users/richer/richer/wisdom-knowledge/.logs/ale/clothing_pattern_blind_e2e/
```
