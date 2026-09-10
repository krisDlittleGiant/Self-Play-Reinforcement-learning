# SGLang patch stack

This directory is the source of truth for the SGLang code used by this repository. The
runtime checkout under `.runtime/sglang-miles` and its node-local `/tmp` mirror are generated
artifacts and are intentionally not tracked.

`env/setup_uv_env.sh` checks out SGLang at
`cb05a44f35a7c9e27e46d74112cc841ca674ef43` and applies these files in order:

1. `sglang-miles-cb05a44-gaudi.patch` — Miles' Gaudi attention, graph, allocator, scheduler,
   sampling, and weight-update implementation.
2. `sglang-cb05a44-hpu-container-compat.patch` — compatibility with the Synapse 1.22.2,
   Torch 2.7.1 container used by VERL.
3. `sglang-cb05a44-verl-hpu-runtime.patch` — GRPO integration fixes found in this repo,
   including HPU-safe host metadata for position construction/native SDPA indexing and
   synchronous same-stream decode-result D2H copies.

The external `miles-gaudi` checkout is reference provenance only. It is not imported by the
launcher and is not required once these patches are present.
