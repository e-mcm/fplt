# fplt data bucket (Cloudflare R2): ops runbook and code plan

**Status:** proposal · **Repo:** `e-mcm/fplt` (public) · **Date:** 2026-11-24 (revised for `fplt-data-model.md`)

> **Scope after the data-model revision.** This doc owns the **bucket infrastructure** and the **raw-ingest move** from the `snapshots` branch to R2: the GitHub Actions capture jobs, `fetch_raw`, parity and cut-over. Everything about curated builds, runs, the ledger, official lines and experiments is defined in **`fplt-data-model.md`** and built through its migration steps (§8 there). Earlier drafts of this doc had a run-partitioned ledger (PR 4a), output pushes (PR 4b), an optional staged cache (PR 7) and a 90-day scratch bucket. These are **superseded** and now point to the data model.

## 0. What and why

Today, raw captures land on the orphan `snapshots` branch. `fetch_snapshots` copies them into each machine's `$FPLT_DATA_DIR/raw`, and everything else (`staged/`, `artifacts/`, `predictions/`, `plans/`, `ledger/`) lives only on the Mac. The bucket gives you:

1. **A durable home for `raw/`.** It is the only data that can't be rebuilt, because missed snapshots are unrecoverable.
2. **One place every machine reads from:** the Mac, the remote experiment VMs, cloud agents, Modal, and later the website build.
3. **Everything else as well,** kept forever: curated builds, runs (predictions, sub-model outputs, plans), official pointers, experiments, specs. The layout is defined in `fplt-data-model.md`.

**The principle: every object is written once.** Object storage has no append and no locks, so a file that is modified in place loses rows when two machines write it at the same time. Every table is therefore laid out as immutable partitions: one file per run, capture or build. Readers scan the whole prefix. Under this rule, anything can safely go into the bucket.

**Unchanged:** library code still reads a **local** `$FPLT_DATA_DIR` through `DataStore`/`PITView`. The bucket is a durable mirror that you pull from and push to at the edges. No model or feature code touches the network (AGENTS.md §2).

**Design conflict to resolve first:** `docs/design.md` decision 2 says object storage is "only worth revisiting if that branch grows unwieldy". Moving to R2 changes that decision, so PR 0 updates the design doc before any code lands (AGENTS.md §1).

## 1. Layout

Three buckets, because R2 tokens are scoped per bucket, not per prefix:

| Bucket | Prefixes | Who writes | Locked (indefinite) |
|---|---|---|---|
| `fplt-data` | `raw/`, `curated/`, `specs/`, `runs/` (official runs only), `official/`, `catalog/`, `status/` | GHA capture jobs (`raw/`); GHA ops jobs (official runs, pointers, `status/`); owner token | `raw/`, `curated/`, `specs/`, `runs/`, `official/` |
| `fplt-lab` | `runs/` (adhoc, experiment, benchmark, legacy), `experiments/`, `batches/`, `specs/`, `catalog/` | owner, cloud agents, Modal | `runs/`, `experiments/`, `batches/`, `specs/` |
| `fplt-public` (later) | whitelisted JSON for the website | publish step only | none (regenerable from official runs) |

- **No lifecycle or expiry rules anywhere.** Everything is kept forever (data model §5).
- `catalog/` and `status/` are the only rewritten prefixes. Both are derived and can be rebuilt (data model §3.5).
- `fplt-lab` is separate only so agent tokens can never write official data.
- Key paths copy the local layout (`fplt-data/raw/fpl_api/season=…/captured_at=…/bootstrap_static.json.gz`), so a pull is a plain `rclone copy` into `$FPLT_DATA_DIR`.
- Model and feature code keeps reading the **local** mirror through `PITView`. Polars can scan `s3://` directly, but that would put network failures inside backtests and make offline tests harder, so the pull and push stay at the edges.

## 2. Ops steps (one-off, about 30 minutes)

### 2.1 Account and buckets
1. Cloudflare dashboard → **R2 Object Storage** → enable R2. A payment card is required even if you stay inside the free tier, and you can't create API tokens until R2 is enabled.
2. Create bucket **`fplt-data`**. Location: Automatic. Storage class: Standard (the free tier covers Standard only). **Leave public access off.**
3. Create bucket **`fplt-lab`** the same way. Create `fplt-public` later, when the website starts.

### 2.2 Write-once enforcement
Bucket locks stop objects from being deleted or overwritten, and they also apply to objects already in the bucket. Install wrangler (`npm i -g wrangler`, then `wrangler login`). Add each lock **when its prefix first gets real data**, not before:

```bash
# now (raw is backfilled in §2.6)
wrangler r2 bucket lock add fplt-data --name raw-forever      --prefix raw/      --retention-indefinite
# when the data-model run writer lands (data model step 3)
wrangler r2 bucket lock add fplt-data --name runs-forever     --prefix runs/     --retention-indefinite
wrangler r2 bucket lock add fplt-lab  --name runs-forever     --prefix runs/     --retention-indefinite
wrangler r2 bucket lock add fplt-data --name specs-forever    --prefix specs/    --retention-indefinite
wrangler r2 bucket lock add fplt-lab  --name specs-forever    --prefix specs/    --retention-indefinite
# when curated builds land (data model step 5)
wrangler r2 bucket lock add fplt-data --name curated-forever  --prefix curated/  --retention-indefinite
# when official lines land (data model step 6)
wrangler r2 bucket lock add fplt-data --name official-forever --prefix official/ --retention-indefinite
# when experiments land (data model step 7)
wrangler r2 bucket lock add fplt-lab  --name exp-forever      --prefix experiments/ --retention-indefinite
wrangler r2 bucket lock add fplt-lab  --name batches-forever  --prefix batches/  --retention-indefinite
```

Consequence: a bad object under a locked prefix stays there forever. That is consistent with "every object is written once". If a capture is bad, quarantine it (ops-06): never delete it.

### 2.3 API tokens (R2 → Manage API tokens → Create)
Create three tokens, all with **Object** permissions (not Admin), each limited to specific buckets:

| Token | Permission | Buckets | Lives in |
|---|---|---|---|
| `fplt-gha-writer` | Object Read & Write | `fplt-data` | GitHub Actions secrets: capture jobs and the ops job (official runs, pointers, status) |
| `fplt-owner` | Object Read & Write | `fplt-data`, `fplt-lab` | `~/.config/fplt/env` on the Mac / your VM |
| `fplt-agent` | Object Read only on `fplt-data`; Read & Write on `fplt-lab` | both | Modal Secret, cloud-agent env |

If the dashboard won't let one token carry different permissions per bucket, split `fplt-agent` into two tokens. Copy each Access Key ID and Secret straight away; Cloudflare shows the secret only once. Also note your **Account ID**. The S3 endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.

### 2.4 Local config (Mac and VMs)
`fplt` already loads `~/.config/fplt/env`, and `deploy/launchd/run-with-env.sh` already sources it. Add these lines:

```bash
FPLT_R2_ACCOUNT_ID=<account id>
FPLT_R2_ACCESS_KEY_ID=<fplt-owner key id>
FPLT_R2_SECRET_ACCESS_KEY=<fplt-owner secret>
FPLT_DATA_BUCKET=fplt-data
FPLT_LAB_BUCKET=fplt-lab
```

Install rclone with `brew install rclone` on the Mac or `sudo apt-get install -y rclone` on Ubuntu. To use rclone by hand, define the remote once with these env vars (put them in your shell rc):

```bash
export RCLONE_CONFIG_FPLT_TYPE=s3 RCLONE_CONFIG_FPLT_PROVIDER=Cloudflare \
  RCLONE_CONFIG_FPLT_ENDPOINT="https://$FPLT_R2_ACCOUNT_ID.r2.cloudflarestorage.com" \
  RCLONE_CONFIG_FPLT_ACCESS_KEY_ID="$FPLT_R2_ACCESS_KEY_ID" \
  RCLONE_CONFIG_FPLT_SECRET_ACCESS_KEY="$FPLT_R2_SECRET_ACCESS_KEY" \
  RCLONE_CONFIG_FPLT_NO_CHECK_BUCKET=true      # object-scoped tokens can't create buckets
rclone lsd fplt:fplt-data                       # smoke test
```

### 2.5 GitHub
Repo → Settings → Secrets and variables → Actions:
- **Secrets:** `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (the `fplt-gha-writer` token).
- **Variable:** `DATA_BUCKET=fplt-data`.
- The same secrets serve the ingestion workflows now and the ops workflows later (official runs). Rotate them together (ops-11).
- Optional hardening: put the secrets in an Environment called `r2`, restricted to the `main` branch, and add `environment: r2` to the capture jobs. Scheduled workflows run on `main`, and fork PRs never receive secrets.

### 2.6 Backfill raw (once, from the Mac)
The Mac holds the most complete `raw/`, because the branch tip only has recent captures (about 6 MB and 180 files today).

```bash
rclone copy --immutable --fast-list -P "$FPLT_DATA_DIR/raw" fplt:fplt-data/raw
git clone --depth 1 --branch snapshots --single-branch https://github.com/e-mcm/fplt /tmp/snap
rclone copy --immutable --fast-list /tmp/snap/data/raw fplt:fplt-data/raw
rclone check --one-way "$FPLT_DATA_DIR/raw" fplt:fplt-data/raw     # expect 0 differences
```

`--immutable` makes rclone stop with an error if a file at the destination differs from the source, and it skips files that are identical. That gives write-once semantics from the client side, and the bucket lock enforces them on the server. **Never use `rclone sync`** against either bucket, because it deletes files at the destination.

Then test the lock: try to overwrite one raw object. It should be refused.

**Existing outputs** (`artifacts/`, `predictions/`, `plans/`, `ledger/`, `benchmark_projections/`, eval output dirs) are **not** copied raw. They are imported once as `kind=legacy` runs into `fplt-lab` by the data model's legacy-import step (step 9), which writes proper manifests. Until then they stay on the Mac. Back up `$FPLT_DATA_DIR` to an external disk or a private bucket prefix first if the Mac is their only copy.

### 2.7 Cost
Expect $0 a month. The free tier gives 10 GB-months of storage, 1M Class A and 10M Class B operations a month, and egress is free.

- **Writes:** captures are a handful of PUTs every 15 minutes, so roughly 10–20k a month.
- **Listings:** listing counts as Class A. A full-tree pull lists 1,000 keys per request, so even about 3,000 pulls a month of a 20k-object `raw/` comes to about 60k.

Above the free tier, Standard storage is $0.015 per GB-month. With runs and experiments kept forever, expect to pass 10 GB within a season or two of heavy experimenting: ~$1–2 a month (data model §5).

## 3. Code changes, as small PRs

Each PR stays under about 200 lines of non-test code, runs `make check`, and ships tests alongside the code.

### PR 0: design (docs only)
- `docs/design.md` decision 2: snapshot storage moves to R2 bucket `fplt-data`, and the `snapshots` branch is kept as a read-only archive after cut-over.
- §4.3: add the bucket layout from §1 above. The full storage contract lands via the data model's design PR (its step 0), which can be the same PR.
- Lines about the cadence and sensors table, and §4.x on the snapshots branch: describe dual-write, then the bucket as source of truth.
- AGENTS.md workflow section: cloud agents pull from `fplt-data` with the read-only token and write only to `fplt-lab`.
- Record the "every object is written once" principle. The ledger-as-view change is in the data model's design PR.

### PR 1: settings and the `fplt.data.remote` module
New settings fields (`FPLT_` prefix, loaded from `~/.config/fplt/env`):

```python
r2_account_id: str | None = None
r2_access_key_id: str | None = None
r2_secret_access_key: SecretStr | None = None
data_bucket: str | None = None       # unset ⇒ old snapshots-branch behaviour
lab_bucket: str | None = None
```

`src/fplt/data/remote.py` follows the existing `snapshots.py` pattern: it shells out to one well-known binary, and the tests monkeypatch a single `_rclone` seam. No new Python dependency.

```python
class RemoteError(RuntimeError):
    """An rclone transfer against the bucket failed."""

@dataclass(frozen=True)
class Remote:
    """An rclone target plus the env that configures it (credentials never in argv)."""
    target: str                 # "fplt:fplt-data", or a local dir in tests
    kind: Literal["data", "lab"]
    env: dict[str, str]

    @classmethod
    def from_settings(cls, s: Settings, bucket: str | None = None) -> "Remote | None":
        """Build the R2 remote from settings; ``None`` when no bucket is configured."""

# Which prefixes each bucket accepts; push() refuses anything else.
PUSHABLE = {
    "data": frozenset({"raw", "curated", "specs", "runs", "official", "catalog", "status"}),
    "lab":  frozenset({"runs", "experiments", "batches", "specs", "catalog"}),
}

def pull(remote: Remote, prefix: str, dest: Path) -> list[Path]:
    """Copy new objects under ``prefix`` into ``dest``; never overwrite local files.

    Returns:
        Newly written local paths (diff of the file set before and after).

    Raises:
        RemoteError: rclone failed, including a local/remote content mismatch.
    """
    # rclone copy {target}/{prefix} dest --immutable --fast-list --transfers 16

def push(remote: Remote, src: Path, prefix: str) -> None:
    """Upload new files under ``src`` to ``prefix``; refuse non-whitelisted prefixes."""
    # ValueError if prefix.split("/")[0] not in PUSHABLE[remote.kind]
    # rclone copy src {target}/{prefix} --immutable --fast-list
```

`_rclone_env(settings)` maps the settings onto `RCLONE_CONFIG_FPLT_{TYPE,PROVIDER,ENDPOINT,ACCESS_KEY_ID,SECRET_ACCESS_KEY,NO_CHECK_BUCKET}`. A missing `rclone` binary raises `RemoteError` with an install hint.

Tests:
- Unit tests mock `_rclone`: argv shape, env mapping, no secret in argv, prefix whitelist `ValueError`, non-zero exit → `RemoteError`, missing binary.
- An integration test runs **real rclone against local directories** (`Remote(target=str(tmp/"bucket"), env={})`). It covers pull → only new files; second pull → `[]`; changed local file → `RemoteError` (the immutable guard). It never touches the network and is skipped if rclone is absent. Add `sudo apt-get install -y rclone` to `ci.yml` so it runs in CI.

### PR 2: read raw from the bucket
- Add `fetch_raw(dest, settings) -> list[Path]` in `fplt.data.snapshots`. It calls `remote.pull(..., "raw", dest)` when `data_bucket` is set, and the existing `fetch_snapshots` (git branch) otherwise. The return type stays the same, so callers don't change behaviour.
- `ops_catchup.run_ops_catchup` and `orchestration/sensors.py` call `fetch_raw`.
- CLI `ops-catchup`: add `--source {auto,bucket,branch}` (default `auto`), and keep `--remote`/`--branch` for the branch path.
- Tests: both branches of `fetch_raw`, sensor tests updated, and a CLI flag test.

### PR 3: dual-write in the capture workflows
Add a composite action, `.github/actions/push-raw/action.yml`, so the four workflows don't repeat the same 15 lines:

```yaml
name: push-raw
description: Upload new raw captures to R2 (write-once)
inputs:
  src: { required: true }
runs:
  using: composite
  steps:
    - shell: bash
      run: sudo apt-get update -qq && sudo apt-get install -y -qq rclone
    - shell: bash
      env:
        RCLONE_CONFIG_FPLT_TYPE: s3
        RCLONE_CONFIG_FPLT_PROVIDER: Cloudflare
        RCLONE_CONFIG_FPLT_NO_CHECK_BUCKET: "true"
      run: |
        set -euo pipefail
        export RCLONE_CONFIG_FPLT_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        export RCLONE_CONFIG_FPLT_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
        export RCLONE_CONFIG_FPLT_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
        rclone copy "${{ inputs.src }}" "fplt:${DATA_BUCKET}/raw" \
          --immutable --fast-list --stats-one-line -v
```

In `snapshot.yml`, `understat.yml`, `api_football.yml` and `football_data.yml`, add this step **before** the existing branch publish:

```yaml
      - name: Upload to R2
        uses: ./.github/actions/push-raw
        with: { src: "${{ runner.temp }}/raw" }
        env:
          R2_ACCOUNT_ID: ${{ secrets.R2_ACCOUNT_ID }}
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          DATA_BUCKET: ${{ vars.DATA_BUCKET }}
```

Add `if: always()` to the branch-publish step. Each sink then runs even if the other fails, so a single R2 outage never loses a capture. Check each workflow's real output directory when you implement this; understat and API-Football may not use `$RUNNER_TEMP/raw`. Logs show only rclone's one-line stats; secrets are masked, and there are no object listings.

### PR 4: `fplt data pull/push` CLI
- Thin wrappers over `remote.pull`/`push`, bucket chosen by `--bucket data|lab`, prefixes checked against `PUSHABLE`.
- That's all. Pushing runs, curated builds, official pointers and experiments is done by the data model's writers (`RunWriter`, the build writer, the ops job, `fplt exp submit`), each built on this module. Earlier drafts had a run-partitioned ledger (PR 4a) and per-`as_of` output pushes (PR 4b); both are **superseded** by runs and the ledger view (data model §3.3, §3.6).

### PR 5: parity check (AGENTS "hard investigation → utility")
- `scripts/raw_parity.py` (or `fplt data parity`) clones the branch tip and runs `rclone check --one-way branch → bucket`. It reports missing or different keys and exits non-zero if there are any.
- Add a weekly GHA job that runs it during dual-write.

### PR 6: cut-over (after about 2 clean weeks of parity)
- Remove the branch-publish steps. The bucket becomes the only sink.
- `fetch_raw` default `auto` → bucket. The branch path stays available behind `--source branch` for one season, then goes.
- Keep the `snapshots` branch as a read-only archive. Don't delete it: it is raw data.
- Update design.md and `docs/dagster-cadence-and-setup.md`.

### ~~PR 7: build-keyed staged and feature caches~~ (superseded)
Replaced by the data model's **content-addressed curated builds** (§3.2, migration step 5), which are no longer optional: runs record the `build_id` they used.

## 4. Consumers after cut-over

| Where | Setup | Typical command |
|---|---|---|
| GHA ingestion jobs | `fplt-gha-writer` secrets | capture → `push-raw` composite action → `fplt-data/raw/` |
| GHA ops job (after ops-tick moves to GHA) | the same secrets | pull raw → build curated → official runs → pointers, all in `fplt-data` |
| Mac / your VM | `fplt-owner` token in `~/.config/fplt/env`, rclone via brew/apt | `fplt data pull raw`; ad hoc runs and experiments write to `fplt-lab` |
| Cloud agents | `fplt-agent` token in env, `apt-get install -y rclone`, `uv sync --frozen` | pull raw and curated builds from `fplt-data`; runs and experiments go to `fplt-lab` |
| Modal | `modal.Secret` with `fplt-agent`, image `.apt_install("rclone")` | pull once per container; write runs to `fplt-lab` |
| Website (later) | `fplt-public` + publish step | never reads `fplt-data` or `fplt-lab` directly |

## 5. Checklist
- [ ] R2 enabled; `fplt-data` and `fplt-lab` created, public access off, **no lifecycle rules**
- [ ] `raw/` lock added; other locks added as each prefix starts receiving data (§2.2)
- [ ] Three tokens created and stored (GHA secrets, `~/.config/fplt/env`, Modal Secret)
- [ ] Raw backfill done; `rclone check` clean; overwrite attempt refused
- [ ] Mac `$FPLT_DATA_DIR` backed up before the legacy import
- [ ] PR 0 → PR 6 in order for raw ingest; everything after follows `fplt-data-model.md` §8
