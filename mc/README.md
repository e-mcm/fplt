# fplt-mc

Native C++ Monte Carlo kernel for `fplt` (nanobind + scikit-build-core).

**Tree:** `mc/` (distribution name `fplt-mc`). Cloud agents and experiment VMs
**must never compile** this package — they install a prebuilt wheel via
`uv sync --frozen` with `no-build-package = ["fplt-mc"]`.

## Who compiles?

| Who | Command | Compiles? |
|---|---|---|
| Cloud agent / board VM | `uv sync --frozen` then `import fplt_mc` | **No** |
| Mac laptop, Python-only | `uv sync --frozen` | No (arm64 wheel) |
| C++ editor (laptop or this agent) | `make mc-dev` | Yes, locally only |

## Local editable build

Needs cmake ≥ 3.26, Ninja, a C++20 compiler, and the repo venv:

```bash
make mc-dev      # uv sync --frozen && uv pip install --no-deps -e mc
make mc-test     # ctest L0 + Python smoke
make mc-format   # clang-format -i on mc/**/*.{hpp,cpp}
make mc-bench    # §1.1 per-step timing harness (stub until kernels land)
```

## Port priority (after PR0)

PR0 vectorised NumPy hard starters: **~205 ms → ~5.6 ms**. Native K1 is
lower priority; **K2 allocate + K3 points** dominate remaining cost. Skeleton
headers for K1–K5 still live under `include/fplt_mc/`; implementation order
follows the re-ranked sequence in the build spec.

## Wheel targets (CI)

| Platform | Runner | Tag |
|---|---|---|
| Linux x86_64 | `ubuntu-24.04` + manylinux_2_28 | `manylinux_2_28_x86_64` |
| macOS arm64 | `macos-15` | `macosx_13_0_arm64` |

ABI: prefer `cp312-abi3` (nanobind stable ABI) when the build succeeds; else
`cp313` matching the repo pin.

## Private index (open)

Preferred host: private S3-compatible bucket (Cloudflare R2 or S3) serving a
static PEP 503 simple index. Until credentials/host are provisioned:

1. CI uploads wheels as GitHub Actions artifacts (`wheel-<plat>`).
2. `mc-release.yml` scaffolds the publish step but does not push to an index.
3. Do **not** pin `fplt-mc` in the root `uv.lock` until a wheel is reachable
   without compiling.

See the Project store doc `docs/native-mc-pr1.md` for arch confirmation and
open blockers.
