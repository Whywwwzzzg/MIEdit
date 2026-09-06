# Experiment configuration

This folder reproduces the original SD3.5 Medium PIE-Bench result for:

`maxk_11_maxv_12_cfgdiff_8_9_10_11_12_13_layers_7_10_13_16_19_sgs_1p6_gs_3p6_steps_27_skip_7`

Generation is followed automatically by full PIE-Bench evaluation:

```bash
python run_pie_bench.py
```

Server launcher:

```bash
nohup env GPU=7 ./run_pie_bench_sequential.sh > run_pie_bench_sequential.log 2>&1 &
```

Default parameters:

- SD3.5 Medium local model
- PIE-Bench source: `../../PIE-Bench_v1`
- image size: `512 x 512`
- seed: `2`
- deterministic algorithms: enabled
- `num_inference_steps`: `27`
- `skip_steps`: `7`
- `source_guidance_scale`: `1.6`
- `guidance_scale`: `3.6`
- feature layers: `7,10,13,16,19`
- `cfg_diff_steps`: `8,9,10,11,12,13`
- `max_single_k_mask_steps`: `11`
- `max_single_v_mask_steps`: `12`
- mask aggregation: `cosine_dissimilarity`
- KV replacement: enabled, starts at transformer block `11`
- latent blending: enabled, steps `3` through `27`
- SASolver `use_h_zero_cache`: enabled

## Exact reproduction requirement

Run all 700 samples sequentially in one process. `scheduler.step` preserves the
original behavior of consuming the process-level CUDA RNG. Splitting index
ranges across workers resets that RNG at each worker boundary and changes the
later images. Do not use multi-process index sharding or `--skip_existing` when
reproducing the published result from scratch.

`run_pie_bench.py` evaluates automatically unless `--no_eval` is explicitly
provided.
