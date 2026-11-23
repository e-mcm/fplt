# CORE-MATH (exp stub)

PR1 exposes `fplt_mc::mathx::exp` as a thin wrapper around `std::exp`.

Vendoring plan (follow-up PR):
- Copy CORE-MATH `cr_exp` (MIT) sources into this directory.
- Switch `src/mathx.cpp` to call `cr_exp` so Poisson `exp(-λ)` is bit-identical
  on glibc and macOS libm.
- Add known-answer tests against CORE-MATH reference values.

Until then, `build_info()["math"] == "core-math-stub"`.
