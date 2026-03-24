# cambrian-artifacts

Generated artifacts for the [Cambrian](https://github.com/lispmeister/cambrian) self-reproducing code factory.

Each directory is a generation:

| Directory | Producer | Status |
|-----------|----------|--------|
| `gen-0/`  | hand-crafted | test artifact (Phase 0 validation) |
| `gen-1/`  | human + Claude Code | first Prime |
| `gen-N/`  | Gen-(N-1) Prime | autonomous reproduction |

Promoted generations are tagged (`gen-N`). Failed generations are tagged (`gen-N-failed`).

Generation history is tracked in `generations.json` (managed by the Supervisor).
