# Conditions for the figures

These are reference facts for captions outside the images. PNG/SVG exports deliberately contain only metric labels and results.

## 01 — Aggregate output

- Six timed runs across two independently started servers on one DGX Station GB300. Peak 11,840; range 11,677–11,840 aggregate output tok/s.
- 256 concurrent requests; exactly 512 input / 512 generated output tokens each; streaming HTTP on loopback; reasoning enabled. Output includes reasoning and final-answer tokens.
- Native NVFP4 weights, FP8 KV, FP16 recurrent state with stochastic rounding, MTP1; 16,384 configured context; prefix caching off. Warmup excluded.
- Aggregate rate includes prefill and client overhead: total completion tokens divided by batch wall time. A server means a fresh serving process, not another physical machine.
- Public comparison uses a different precision/context and an undisclosed workload; this figure makes no matched comparison with Actual.

## 02 — Matched throughput improvement

- At 256 concurrent requests, pooled rates are 9,711.96 tok/s for FP32 recurrent state and 11,815.99 for FP16 state plus stochastic rounding: +21.664%, rounded to +21.7%.
- Each bar pools total tokens / total elapsed time over three initial-instance runs. Both arms have the same model, reasoning setting, workload, runtime, FlashInfer, MTP1 and asynchronous scheduling.
- The intervention combines state precision and stochastic rounding; individual contributions were not isolated. Both passed 23 fixed quality cases repeated three times, plus six smoke checks.
- Same 512-input / 512-output, NVFP4-weight, FP8-KV, 16K-context scope as figure 01. Bars start at zero; no uncertainty or confidence interval is implied.

## 03 — Single-stream decode

- Earlier Triton/MTP3 probe: 540.65 decode tok/s, rounded to 541. Latest MTP1 probes: 390.29 and 397.65 tok/s, rounded to the displayed range 390–398.
- Each probe: one request, 8,192 input / 128 generated output tokens. Common-decode metric excludes the first token and starts at first streamed text.
- The latest bar length uses pooled common-decode time across the two equal-length probes; its label preserves the observed range. The earlier result is one observation.
- Different configurations and quality scopes: earlier 20-check suite; latest 23 distinct cases repeated three times. These are not one configuration's simultaneous single-stream and aggregate records.

## Exports

- Main figures: dark canvas and blue emphasis, inspired by Cline's published benchmark graphics.
- `light/`: warm canvas and muted terracotta alternative, inspired by Anthropic's editorial presentation.
- Every PNG is 1920 × 1080. SVGs remain editable. No logos or third-party artwork are reproduced.

Design references: [Cline's IMO post](https://cline.bot/blog/deepseek-wins-imo-gold-on-12-cents) and [Anthropic's evals article](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
