import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from matric_calculator import MetricsCalculator


DEFAULT_MANIFEST_PATH = "../MERGED_DATASET_LATEST2/MERGED_DATASET_LATEST2/merged_manifest.json"
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


def clean_prompt(text):
    text = "" if text is None else str(text)
    return re.sub(r"[\[\]{}()]", "", text).strip()


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "data", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported manifest format: {path}")


def normalize_relative_path(path_value):
    if not path_value:
        return None
    path_value = str(path_value).replace("\\", "/")
    while path_value.startswith("./"):
        path_value = path_value[2:]
    return path_value.lstrip("/")


def is_absolute_path_value(path_value):
    if not path_value:
        return False
    path_text = str(path_value)
    return Path(path_text).is_absolute() or path_text.startswith(("/", "\\")) or (
        len(path_text) > 1 and path_text[1] == ":"
    )


def get_manifest_image_path(item):
    for key in ("image_path", "image_relative_path", "copied_to"):
        value = item.get(key)
        if value and not is_absolute_path_value(value):
            return value
    return item.get("image_path") or item.get("image_relative_path") or item.get("copied_to")


def resolve_image_path(item, dataset_root):
    candidates = []

    image_path = normalize_relative_path(get_manifest_image_path(item))
    if image_path:
        candidates.append(dataset_root / image_path)

    original_copied_to = normalize_relative_path(item.get("original_copied_to"))
    if original_copied_to:
        candidates.append(dataset_root / original_copied_to)

    for key in ("image_path", "original_image_path"):
        raw_path = item.get(key)
        if not raw_path:
            continue
        raw_path = Path(str(raw_path))
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.append(dataset_root / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot resolve source image for {item.get('unified_id', '<unknown>')}.\nTried:\n  {tried}")


def get_output_relative_path(item, image_path):
    edit_type = item.get("edit_type") or "unknown"
    filename = item.get("new_filename") or f"{item.get('unified_id', image_path.stem)}{image_path.suffix}"
    return f"{edit_type}/{filename}"


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


def write_summary(summary_path, method, metrics, all_results, missing_target_count):
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
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
                metric,
                sum(valid_values) / len(valid_values) if valid_values else "nan",
                len(valid_values),
                len(values) - len(valid_values),
                missing_target_count,
            ])


def resolve_mask_path(item, dataset_root):
    candidates = []

    mask_path = item.get("mask_path")
    if mask_path:
        mask_path = Path(str(mask_path))
        if not mask_path.is_absolute():
            candidates.append(dataset_root / mask_path)
        else:
            candidates.append(mask_path)

    image_path = normalize_relative_path(get_manifest_image_path(item))
    if image_path:
        candidates.append((dataset_root / "_masks" / image_path).with_suffix(".png"))

    edit_type = item.get("edit_type") or "unknown"
    new_filename = item.get("new_filename")
    if new_filename:
        candidates.append(dataset_root / "_masks" / edit_type / Path(str(new_filename)).with_suffix(".png").name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot resolve mask for {item.get('unified_id', '<unknown>')}.\nTried:\n  {tried}")


def find_generated_image(item, source_image_path, target_roots):
    rel_path = Path(get_output_relative_path(item, source_image_path))
    candidates = []
    for root in target_roots:
        candidates.append(root / "annotation_images" / rel_path)
        candidates.append(root / rel_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate, rel_path

    return None, rel_path


def resize_image_to(image, size, resample=Image.Resampling.LANCZOS):
    if image.size == size:
        return image
    return image.resize(size, resample)


def load_binary_mask(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    mask = resize_image_to(mask, size, Image.Resampling.NEAREST)
    mask_np = (np.array(mask) > 127).astype(np.float32)
    return mask_np[:, :, None].repeat(3, axis=2)


def write_detail_header(writer, method, metrics):
    writer.writerow([
        "index",
        "unified_id",
        "edit_type",
        "source_image",
        "generated_image",
        "mask",
        *[f"{method}|{metric}" for metric in metrics],
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIEdit outputs on EditEval++")
    parser.add_argument("--manifest_path", type=str, default=DEFAULT_MANIFEST_PATH,
                        help="Path to merged_manifest.json")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Dataset root. Defaults to the directory containing merged_manifest.json")
    parser.add_argument("--target_path", nargs="+", required=True,
                        help="One or more run_editevalpp output directories, e.g. editevalpp_results/gpu4 gpu5 ...")
    parser.add_argument("--result_path", type=str, default=None,
                        help="CSV path for per-image results")
    parser.add_argument("--summary_path", type=str, default=None,
                        help="CSV path for metric summary")
    parser.add_argument("--metrics", nargs="+", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--edit_type_list", nargs="+", type=str, default=None)
    parser.add_argument("--source_prompt_field", type=str, default="source_description")
    parser.add_argument("--target_prompt_field", type=str, default="target_description")
    parser.add_argument("--missing_target", choices=["nan", "error"], default="nan")
    parser.add_argument("--metric_error", choices=["nan", "error"], default="error")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else manifest_path.parent
    target_roots = [Path(path).expanduser().resolve() for path in args.target_path]
    method = "+".join(root.name for root in target_roots)
    result_path = Path(args.result_path).expanduser().resolve() if args.result_path else target_roots[0].parent / "editevalpp_eval_detail.csv"
    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else target_roots[0].parent / "editevalpp_eval_summary.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    if args.edit_type_list:
        keep = set(args.edit_type_list)
        manifest = [item for item in manifest if item.get("edit_type") in keep]

    start_idx = max(args.start_idx, 0)
    end_idx = args.end_idx if args.end_idx is not None else len(manifest)
    end_idx = min(end_idx, len(manifest))
    selected_items = manifest[start_idx:end_idx]
    metrics = args.metrics
    all_results = {metric: [] for metric in metrics}
    missing_target_count = 0

    print("=" * 60)
    print("Starting EditEval++ evaluation")
    print(f"Manifest: {manifest_path}")
    print(f"Dataset root: {dataset_root}")
    print("Target roots:")
    for root in target_roots:
        print(f"  {root}")
    print(f"Result CSV: {result_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Samples: {len(selected_items)} (index {start_idx} to {end_idx})")

    metrics_calculator = MetricsCalculator(args.eval_device, clip_model_path=args.clip_model_path)

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_detail_header(writer, method, metrics)

        for offset, item in enumerate(selected_items, start=start_idx):
            unified_id = item.get("unified_id") or f"sample_{offset:06d}"
            edit_type = item.get("edit_type")
            print(f"[{offset + 1}/{len(manifest)}] evaluating {unified_id}")

            try:
                source_image_path = resolve_image_path(item, dataset_root)
                generated_image_path, rel_path = find_generated_image(item, source_image_path, target_roots)
                mask_path = resolve_mask_path(item, dataset_root)
                source_prompt = clean_prompt(item.get(args.source_prompt_field))
                target_prompt = clean_prompt(item.get(args.target_prompt_field))
            except Exception as exc:
                if args.missing_target == "error":
                    raise
                print(f"  [missing metadata] {exc}")
                missing_target_count += 1
                row = [offset, unified_id, edit_type, "", "", "", *["nan" for _ in metrics]]
                writer.writerow(row)
                for metric in metrics:
                    all_results[metric].append("nan")
                continue

            row = [offset, unified_id, edit_type, str(source_image_path), str(generated_image_path or ""), str(mask_path)]
            if generated_image_path is None:
                message = f"Generated image not found for {unified_id}: {rel_path}"
                if args.missing_target == "error":
                    raise FileNotFoundError(message)
                print(f"  [missing] {message}")
                missing_target_count += 1
                row.extend(["nan" for _ in metrics])
                writer.writerow(row)
                for metric in metrics:
                    all_results[metric].append("nan")
                continue

            src_image = Image.open(source_image_path).convert("RGB")
            gen_image = Image.open(generated_image_path).convert("RGB")
            src_image = resize_image_to(src_image, gen_image.size, Image.Resampling.LANCZOS)
            mask = load_binary_mask(mask_path, gen_image.size)

            for metric in metrics:
                print(f"  metric: {metric}")
                try:
                    result = calculate_metric(
                        metrics_calculator,
                        metric,
                        src_image,
                        gen_image,
                        mask,
                        mask,
                        source_prompt,
                        target_prompt,
                    )
                    result = normalize_value(result)
                except Exception:
                    if args.metric_error == "error":
                        raise
                    result = "nan"
                row.append(result)
                all_results[metric].append(result)

            writer.writerow(row)
            f.flush()

        avg_row = ["Average", "", "", "", "", ""]
        for metric in metrics:
            valid_values = [value for value in (value_to_float(v) for v in all_results[metric]) if value is not None]
            avg_row.append(sum(valid_values) / len(valid_values) if valid_values else "nan")
        writer.writerow([])
        writer.writerow(avg_row)

    write_summary(str(summary_path), method, metrics, all_results, missing_target_count)
    print(f"Saved detail CSV: {result_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
