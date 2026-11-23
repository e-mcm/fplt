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

## Public wheel index (GitHub Pages)

Index URL: **https://e-mcm.github.io/fplt/simple/** (no credentials).

`mc-release.yml` builds wheels on version bumps to `main` and publishes them
to the `gh-pages` branch via a PEP 503 simple index (`scripts/mc_simple_index.py`).
Wheel filenames are immutable.

1. Enable **Settings → Pages → Deploy from branch `gh-pages` / root** once.
2. After the first publish, pin `fplt-mc` in the root `pyproject.toml` /
   `uv.lock` with `no-build-package = ["fplt-mc"]` (see comment there).
3. PR CI still uploads `wheel-<plat>` artifacts for the fresh build under test.

See Project store `docs/native-mc-pr1.md`.
