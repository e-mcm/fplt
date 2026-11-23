# Random123 (Philox)

PR1 embeds a self-contained Philox4x64-10 implementation in `src/rng.cpp`
(algorithm from [Random123](https://github.com/DEShawResearch/random123),
BSD 3-clause, © D. E. Shaw Research).

A later PR may vendor the upstream headers here for KAT cross-checks. Do not
pull Random123 into the Python package surface.
