# Original measured campaign

This directory publishes original benchmark implementations, plans, per-request synthetic generations/timings/token usage, quality responses, summaries, rejected configurations and restoration checks from the 2026-10-15 experiment. These are the original measurements, not a rerun of the portable release.

The qualifying runs are in `reasoning/` and the fresh-instance `confirmation/`. Earlier phases retain unsuccessful configurations and the evolution of the quality gate; they must not be combined into the final six-run claim. In particular, thinking-disabled FP16 was rejected for a newly failing observation, and prose identity was found unsuitable because the unchanged baseline itself varied across batch shapes.

`manifest.json` maps every published file to its original relative filename and SHA256. Larger JSON files use deterministic gzip; decompress with `gzip -dc FILE.json.gz`. The `-private` suffix is the original local filename, not a statement that these now-public synthetic benchmark data contain user conversations. All request text is generated benchmark material.

The portable launcher and benchmark at the repository root are the supported reproduction entrypoints. Historical scripts here retain `/work` and experiment-specific references and document the measured implementation. Operational scripts that stop/restore Ara's unrelated GLM service, internal service inventories, local handoff notes, cache contents, container logs and downloaded model weights are intentionally not distributed. This release includes the original benchmark data and sources without publishing credentials or third-party weights/runtime sources. Model pins and the full weight hash manifest are in the parent evidence directory.

Final restoration proofs show the pre-existing service was healthy, unchanged, and independently checked, with the GPU experiment lock released. No serving configuration was promoted.
