# Nemotron 3.5 Super VL on GB300

**11,677–11,840 aggregate output tok/s on one DGX Station GB300.** Pinned vLLM, native MTP1, async scheduling, FlashInfer, and FP16 recurrent state with stochastic rounding.

![Intelligence × speed](figures/intelligence-speed.png)

*Chart reference points are visually estimated, with 100K-input reference speeds versus local 8K. The highlighted intelligence score was not measured locally.*

## Run

Requires an **ARM64 DGX Station GB300**, Docker with NVIDIA Container Toolkit, a compatible driver and an idle GPU. The container has a 360 GB host-memory limit; weights are approximately 75 GB.

```bash
git clone https://github.com/arafatkatze/nemotron-3.5-gb300-recipes.git
cd nemotron-3.5-gb300-recipes
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/download_model.py
bash scripts/start.sh candidate
```

Startup takes several minutes. Follow it with `docker logs -f nemotron35-recipes`. The OpenAI-compatible API runs at `http://127.0.0.1:8015/v1`, model `nemotron-3.5-super-vl`.

The [launcher](scripts/start.sh) pins the image and settings: NVFP4 weights, FP8 KV, 16K context, 256 sequences, prefix caching off. Preview with `bash scripts/start.sh candidate --dry-run`.

## Results

Six runs across two server instances, 256 concurrent requests, 512 input / 512 output tokens. Both instances passed 69/69 fixed correctness observations. Output includes reasoning tokens. The published FP8 / 1M-context result uses different conditions.

[Benchmark instructions and full configuration](docs/BENCHMARK.md) · [Raw evidence](evidence/campaign/)

Stop with `docker stop nemotron35-recipes`; remove the stopped container with `docker rm nemotron35-recipes`. Weights and results remain on disk.

[MIT](LICENSE) for recipe code; NVIDIA’s terms apply to model weights.
