import os
import torch
import argparse
import csv
import json
import math
import random
import re
import gc
import numpy as np
from PIL import Image

from pipeline_stablediffusion3 import SD3EditingPipeline
from scheduling_sasolver import SASolverScheduler


METHOD_NAME = "layers_8_11_14_17_20_23_sgs_1p6_gs_3p6_steps_27_skip_7"
DEFAULT_CLIP_MODEL_PATH = "/data/disk2/haiyan/huggingface/hub/models--openai--clip-vit-large-patch14"
DEFAULT_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]


def parse_mask_feature_layers(raw_layers: str | None):
    if not raw_layers:
        return None

    layers = []
    for item in raw_layers.replace(",", " ").split():
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"transformer_blocks.{item}"
        layers.append(item)

    if not layers:
        raise ValueError("--mask_feature_layers must contain at least one layer")
    return layers


def load_metrics_calculator_class():
    from matric_calculator import MetricsCalculator
    return MetricsCalculator


def mask_decode(encoded_mask, image_shape=(512, 512)):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def normalize_value(value):
    if isinstance(value, str):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return float(np.asarray(value).mean())
    if isinstance(value, np.generic):
        return value.item()
    return value


def value_to_float(value):
    value = normalize_value(value)
    if isinstance(value, str):
        if value.lower() == "nan":
            return None
        return float(value)
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    raise ValueError(f"Unknown metric: {metric}")


def parse_method_scales(name):
    match = re.search(r"sgs_([0-9mp.-]+)_gs_([0-9mp.-]+)", name)
    if not match:
        return "", ""
    return (
        match.group(2).replace("m", "-").replace("p", "."),
        match.group(1).replace("m", "-").replace("p", "."),
    )


def write_summary(summary_path, method, metrics, all_results, missing_target_count):
    guidance_scale, source_guidance_scale = parse_method_scales(method)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "guidance_scale",
            "source_guidance_scale",
            "metric",
            "mean",
            "valid_count",
            "nan_count",
            "missing_target_count",
        ])
        for metric in metrics:
            values = all_results[metric]
            valid_values = [value for value in (value_to_float(v) for v in values) if value is not None]
            writer.writerow([
                method,
                guidance_scale,
                source_guidance_scale,
                metric,
                sum(valid_values) / len(valid_values) if valid_values else "nan",
                len(valid_values),
                len(values) - len(valid_values),
                missing_target_count,
            ])


def evaluate_results(args):
    source_path = os.path.abspath(args.source_path)
    target_path = os.path.abspath(args.target_path)
    annotation_mapping_file = os.path.join(source_path, "mapping_file.json")
    src_image_folder = os.path.join(source_path, "annotation_images")
    tgt_image_folder = os.path.join(target_path, "annotation_images")
    method = os.path.basename(os.path.normpath(target_path))

    result_path = args.eval_result_path or os.path.join(os.path.dirname(target_path), "eval_camera_ready.csv")
    summary_path = args.eval_summary_path or os.path.join(os.path.dirname(target_path), "eval_camera_ready_summary.csv")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if not os.path.exists(annotation_mapping_file):
        raise FileNotFoundError(f"Annotation mapping file not found: {annotation_mapping_file}")
    if not os.path.exists(src_image_folder):
        raise FileNotFoundError(f"Source image folder not found: {src_image_folder}")
    if not os.path.exists(tgt_image_folder):
        raise FileNotFoundError(f"Target image folder not found: {tgt_image_folder}")

    print("\n" + "=" * 60)
    print("Starting evaluation")
    print(f"method: {method}")
    print(f"target images: {tgt_image_folder}")
    print(f"result csv: {result_path}")
    print(f"summary csv: {summary_path}")

    MetricsCalculator = load_metrics_calculator_class()
    metrics_calculator = MetricsCalculator(args.eval_device, clip_model_path=args.clip_model_path)

    with open(annotation_mapping_file, "r", encoding="utf-8") as f:
        annotation_file = json.load(f)

    annotation_keys = list(annotation_file.keys())
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(annotation_keys)
    end_idx = min(end_idx, len(annotation_keys))
    selected_keys = annotation_keys[start_idx:end_idx]
    metrics = args.metrics
    all_results = {metric: [] for metric in metrics}
    missing_target_count = 0

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id"] + [f"{method}|{metric}" for metric in metrics])

        for row_index, key in enumerate(selected_keys, start=1):
            item = annotation_file[key]
            if str(item["editing_type_id"]) not in args.edit_category_list:
                continue

            print(f"[{row_index}/{len(selected_keys)}] evaluating image {key}")
            base_image_path = item["image_path"]
            src_image_path = os.path.join(src_image_folder, base_image_path)
            tgt_image_path = os.path.join(tgt_image_folder, base_image_path)
            if not os.path.exists(src_image_path):
                raise FileNotFoundError(f"Source image not found: {src_image_path}")

            src_image = Image.open(src_image_path).convert("RGB")
            mask = mask_decode(item["mask"], image_shape=(args.mask_height, args.mask_width))
            mask = mask[:, :, np.newaxis].repeat([3], axis=2)
            original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

            evaluation_row = [key]
            if not os.path.exists(tgt_image_path):
                message = f"Target image not found: {tgt_image_path}"
                if args.missing_target == "error":
                    raise FileNotFoundError(message)
                print(f"  [missing] {message}")
                missing_target_count += 1
                for metric in metrics:
                    evaluation_row.append("nan")
                    all_results[metric].append("nan")
                writer.writerow(evaluation_row)
                f.flush()
                continue

            tgt_image = Image.open(tgt_image_path).convert("RGB")
            if tgt_image.size[0] != tgt_image.size[1]:
                tgt_image = tgt_image.crop((
                    tgt_image.size[0] - 512,
                    tgt_image.size[1] - 512,
                    tgt_image.size[0],
                    tgt_image.size[1],
                ))

            for metric in metrics:
                print(f"  metric: {metric}")
                try:
                    result = calculate_metric(
                        metrics_calculator,
                        metric,
                        src_image,
                        tgt_image,
                        mask,
                        mask,
                        original_prompt,
                        editing_prompt,
                    )
                    result = normalize_value(result)
                except Exception:
                    if args.metric_error == "error":
                        raise
                    result = "nan"
                evaluation_row.append(result)
                all_results[metric].append(result)

            writer.writerow(evaluation_row)
            f.flush()

        avg_row = ["Average"]
        for metric in metrics:
            valid_values = [value for value in (value_to_float(v) for v in all_results[metric]) if value is not None]
            avg_row.append(sum(valid_values) / len(valid_values) if valid_values else "nan")
        writer.writerow([])
        writer.writerow(avg_row)

    write_summary(summary_path, method, metrics, all_results, missing_target_count)
    print(f"Saved detail CSV: {result_path}")
    print(f"Saved summary CSV: {summary_path}")


# ============ 确定性设置 ============
def set_deterministic(seed=42, enable=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if enable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


# ============ 模型加载 ============
def load_pipe(
    device: torch.device,
    torch_dtype: torch.dtype = torch.float16,
    model_path: str = "/root/autodl-tmp/models/sd3.5-large",
    use_h_zero_cache: bool = True,
):
    print(f"Loading SD3 pipeline from {model_path} ...")
    pipe = SD3EditingPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )

    print(f"Loading SASolver scheduler... use_h_zero_cache={use_h_zero_cache}")
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config["use_h_zero_cache"] = use_h_zero_cache
    pipe.scheduler = SASolverScheduler.from_config(scheduler_config)

    pipe.to(device)
    print(f"Pipeline loaded successfully on {device}")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Run PIE Bench evaluation with Stable Diffusion 3.5 Medium")
    parser.add_argument('--source_path', type=str, default='../../PIE-Bench_v1',
                        help='Path to the source directory containing mapping_file.json and annotation_images/')
    parser.add_argument(
        '--target_path',
        type=str,
        default='camera_ready_results/layers_8_11_14_17_20_23_sgs_1p6_gs_3p6_steps_27_skip_7',
                        help='Path to the target directory where results will be saved')
    parser.add_argument('--width', type=int, default=512,
                        help='Output image width (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                        help='Output image height (default: 512)')
    parser.add_argument('--num_inference_steps', type=int, default=27,
                        help='Number of inference steps (default: 29)')
    parser.add_argument('--guidance_scale', type=float, default=3.6,
                        help='Guidance scale for generation/editing (default: 3.75)')
    parser.add_argument('--source_guidance_scale', type=float, default=1.6,
                        help='Guidance scale for inversion (default: 1.5)')
    parser.add_argument('--skip_steps', type=int, default=7,
                        help='Number of initial steps to use inversion latents (default: 9)')
    parser.add_argument('--generator_seed', type=int, default=2,
                        help='Random seed for generator (default: 42)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if output file already exists')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start processing from this annotation index (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End processing at this annotation index (exclusive, None=process all)')
    parser.add_argument('--disable_deterministic', action='store_true',
                        help='Disable deterministic seed setup (still sets seeds)')
    parser.add_argument('--cleanup_interval', type=int, default=0,
                        help='Run gc/empty_cache every N successful samples (0 disables periodic cleanup)')
    parser.add_argument('--mask_feature_layers', type=str, default='8,11,14,17,20,23',
                        help='CFG-difference feature layers, e.g. "6,10,14,18,22"')
    parser.add_argument('--disable_h_zero_cache', action='store_true',
                        help='Set SASolverScheduler use_h_zero_cache=False')
    parser.add_argument('--no_eval', action='store_true',
                        help='Only run generation; skip automatic evaluation')
    parser.add_argument('--eval_device', type=str, default='cuda',
                        help='Device used for automatic evaluation')
    parser.add_argument('--clip_model_path', type=str, default=DEFAULT_CLIP_MODEL_PATH,
                        help='Local CLIP model path used for evaluation')
    parser.add_argument('--eval_result_path', type=str, default=None,
                        help='CSV path for per-image evaluation results')
    parser.add_argument('--eval_summary_path', type=str, default=None,
                        help='CSV path for evaluation summary')
    parser.add_argument('--metrics', nargs='+', type=str, default=DEFAULT_METRICS)
    parser.add_argument('--edit_category_list', nargs='+', type=str, default=[str(i) for i in range(10)])
    parser.add_argument('--mask_height', type=int, default=512)
    parser.add_argument('--mask_width', type=int, default=512)
    parser.add_argument('--missing_target', choices=['nan', 'error'], default='nan')
    parser.add_argument('--metric_error', choices=['nan', 'error'], default='error')

    parser.add_argument(
        '--model_path',
        type=str,
        default='/data/disk2/haiyan/hf_models/sd3_5_medium/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80',
        help='Local path to the SD3 model directory'
    )

    args = parser.parse_args()
    mask_feature_layers = parse_mask_feature_layers(args.mask_feature_layers)
    if mask_feature_layers is not None:
        print(f"Using mask feature layers: {mask_feature_layers}")

    # 设置确定性
    set_deterministic(seed=args.generator_seed, enable=not args.disable_deterministic)

    # device & dtype
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    # 加载 pipeline
    pipe = load_pipe(
        device=device,
        torch_dtype=weight_dtype,
        model_path=args.model_path,
        use_h_zero_cache=not args.disable_h_zero_cache,
    )

    root = args.source_path
    target = args.target_path

    annotation_file_name = os.path.join(root, "mapping_file.json")
    if not os.path.exists(annotation_file_name):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file_name}")

    with open(annotation_file_name, "r", encoding="utf-8") as f:
        annotation_file = json.load(f)

    print(f"Found {len(annotation_file)} annotations to process")

    annotation_keys = list(annotation_file.keys())
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(annotation_keys)
    end_idx = min(end_idx, len(annotation_keys))

    print(f"Processing annotations from index {start_idx} to {end_idx}")
    success_count = 0

    # 输出 mask 的目录（我帮你放到 target 下，避免散落在工作目录）
    mask_root = os.path.join(target, "masks")

    for idx in range(start_idx, end_idx):
        annotation_idx = annotation_keys[idx]
        annotation = annotation_file[annotation_idx]

        print(f"\n{'='*60}")
        print(f"Processing [{idx+1}/{len(annotation_keys)}] annotation {annotation_idx}")

        img_path = os.path.join(root, "annotation_images", annotation["image_path"])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        annotation_dir = os.path.dirname(annotation["image_path"])
        full_dir_path = os.path.join(target, "annotation_images", annotation_dir)
        os.makedirs(full_dir_path, exist_ok=True)

        out_path = os.path.join(full_dir_path, os.path.basename(annotation["image_path"]))

        # mask 路径（跟随同样的子目录结构）
        mask_dir = os.path.join(mask_root, annotation_dir)
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, os.path.basename(annotation["image_path"]))

        if args.skip_existing and os.path.exists(out_path):
            print(f"Skipping - output already exists: {out_path}")
            continue

        try:
            imagein = Image.open(img_path).convert("RGB")
            imagein.thumbnail((args.width, args.height), Image.Resampling.LANCZOS)

            import re
            source_prompt = re.sub(r'[\[\]{}()]', '', annotation["original_prompt"])
            target_prompt = re.sub(r'[\[\]{}()]', '', annotation["editing_prompt"])

            print(f"  Source prompt: {source_prompt}")
            print(f"  Target prompt: {target_prompt}")

            blended_word_str = annotation.get("blended_word", "")
            if blended_word_str:
                print(f"  Blended word: {blended_word_str}")

            print("  Running inference...")

            # 每次推理都新建 generator，保证可复现
            generator = torch.Generator(device=device).manual_seed(args.generator_seed)

            result = pipe.generate_with_gt_mask_editing(
                image=imagein,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                num_inference_steps=args.num_inference_steps,
                source_guidance_scale=args.source_guidance_scale,
                target_guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
                save_image=False,           
                save_dir=None,
                aggregation_method="cosine_dissimilarity",
                mask_feature_layers=mask_feature_layers,
                use_kv_replacement=True,
                skip_steps=args.skip_steps,
                use_latent_mixing=True,
                mixing_start_step=3,
                mixing_end_step=args.num_inference_steps,
            )

            output_image = result.get('generated_image', None)
            editing_mask = result.get('editing_mask', None)

            if output_image is None:
                raise RuntimeError("result 中没有 generated_image，检查 generate_with_gt_mask_editing 的返回格式")

            output_image.save(out_path)
            print(f"  ✓ Saved image: {out_path}")

            if editing_mask is not None:
                mask_np = (editing_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
         
                mask_img = mask_img.resize((mask_img.width * 8, mask_img.height * 8), Image.NEAREST)
                mask_img.save(mask_path)
                print(f"  ✓ Saved mask:  {mask_path}")

            del output_image
            del imagein
            success_count += 1
            if args.cleanup_interval > 0 and success_count % args.cleanup_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  ✓ Periodic memory cleanup at sample {success_count}")

        except Exception as e:
            print(f"❌ Error processing annotation {annotation_idx}: {str(e)}")
            import traceback
            traceback.print_exc()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {target}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.no_eval:
        evaluate_results(args)


if __name__ == "__main__":
    main()
