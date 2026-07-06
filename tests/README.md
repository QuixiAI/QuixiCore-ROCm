# Tests

Use `scripts/test` as the common entrypoint when possible.

Current ROCm tests include primitive unit tests under `tests/unit/` and
kernel-local Python tests. New contract tests should mirror the kernel taxonomy
under `tests/correctness/<family>/<operation>/`.
