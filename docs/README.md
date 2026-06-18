# Documentation Index

This folder is written as takeover documentation for someone maintaining or scaling the firefly detection system.

Read in this order:

1. [repository-map.md](repository-map.md) - what each repo folder owns.
2. [environment.md](environment.md) - expected machine, Python packages, GPU notes, and path quoting.
3. [data-roots.md](data-roots.md) - the SSD data layout and catalog/model locations.
4. [day-pipeline.md](day-pipeline.md) - day-time application pipeline stages and outputs.
5. [night-pipeline.md](night-pipeline.md) - night-time application pipeline stages and outputs.
6. [gateway.md](gateway.md) - routing wrapper for running either pipeline.
7. [validation-and-outputs.md](validation-and-outputs.md) - ground truth formats, prediction schemas, overlays, and metrics.
8. [scaling-species.md](scaling-species.md) - ingestion, model zoo, baselines, YOLO tooling, and adding species.
9. [operations-runbook.md](operations-runbook.md) - common tasks and troubleshooting.

The repo is centered around two production-ish pipelines: day and night. Everything else is either a runner, evaluation wrapper, scaling tool, baseline comparison, or manual utility.

