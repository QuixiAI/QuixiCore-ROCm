# Security Policy

QuixiCore ROCm is a native GPU backend. Security issues may involve host-side
bindings, kernel launch validation, memory bounds, container/build tooling, or
packaged artifacts.

## Reporting

Do not open a public issue for a suspected vulnerability. Report security
issues through the QuixiAI GitHub security advisory flow or contact the
maintainers privately.

When reporting, include:

- Affected repository and commit.
- Affected AMD GPU, ROCm version, and container image if applicable.
- Minimal reproduction steps.
- Whether the issue affects public APIs, bindings, generated artifacts, or
  kernel execution.

## Scope

Issues in shared QuixiCore semantics should be reported against
QuixiAI/QuixiCore. Issues in ROCm implementation code should be reported
against this repository.
