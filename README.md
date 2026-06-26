# MIEdit

Official code for **Multi-History-Step SDE Inversion for Image Editing with Superior Regional Awareness**.

In recent years, diffusion stochastic differential equation (SDE) inversion and inversion-free methods have become prevalent for training-free image editing, as they can achieve faithful reconstruction without tuning. However, existing approaches remain inefficient, exhibit limited plasticity, and struggle to accurately preserve unedited regions. To address these issues, we propose MIEdit, a training-free editing framework based on SDE inversion. MIEdit introduces a predictor-corrector multi-history-step scheme to achieve superior editing quality with fewer steps. We further mitigate heterogeneity and conflict between the multi-conditioned noise residuals and gradient terms during sampling, improving stability and editing plasticity under large edits. MIEdit also includes Inversion-Time Automatic Semantic Angle Masking (IASM); it leverages classifier-free guidance to automatically generate semantic angle masks during inversion and applies them throughout the sampling process for regional constraints, without extra user inputs. We additionally construct EditEval++ (30 fine-grained tasks, 1,000+ image-text-mask triplets) for comprehensive evaluation; experiments show that MIEdit outperforms state-of-the-art techniques. Project page: <https://whywwwzzzg.github.io/MIEdit/>.

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

Use `--model_path` to specify your local Stable Diffusion 3.5 checkpoint directory. For example:

```bash
python run_pie_bench.py \
  --model_path /path/to/stable-diffusion-3.5-medium \
  --source_path /path/to/PIE-Bench_v1 \
  --target_path outputs/MIEdit_SD3.5 \
  --clip_model_path /path/to/clip-vit-large-patch14
```

## Benchmark

EditEval++ is available at <https://drive.google.com/file/d/1d1ekSATh2LWEOftfB0A3-UvpiX5wRVsX/view>.

## Project Page

The project page is available at <https://whywwwzzzg.github.io/MIEdit/>.

The source files are in `docs/` and are deployed with GitHub Pages.
