# MIEdit

Official code for **Multi-History-Step SDE Inversion for Image Editing with Superior Regional Awareness**.

MIEdit is a training-free image editing framework based on SDE inversion. It combines a predictor-corrector multi-history-step inversion scheme with Inversion-Time Automatic Semantic Angle Masking (IASM) for efficient editing and improved non-edited region preservation.

## Files

- `pipeline_stablediffusion3.py`: Stable Diffusion 3 editing pipeline with MIEdit editing utilities.
- `custom_attention_processor.py`: SD3 attention processors, feature hooks, and KV replacement utilities.
- `scheduling_sasolver.py`: SA-Solver scheduler used by the editing pipeline.
- `run_pie_bench.py`: PIE-Bench evaluation entry point.
- `matric_calculator.py`: Metric computation utilities.
- `docs/`: Project page for GitHub Pages.

## Environment

Install the main dependencies:

```bash
pip install -r requirements.txt
```

The code expects a CUDA-enabled PyTorch environment and local access to the required diffusion and CLIP model checkpoints.

## Example

```bash
python run_pie_bench.py --help
```

## Project Page

The project page is in `docs/index.html`. After enabling GitHub Pages for this repository, it can be served from the `docs/` directory.

