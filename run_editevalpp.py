import argparse
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from run_pie_bench import load_pipe, parse_mask_feature_layers, set_deterministic


DEFAULT_MANIFEST_PATH = "../MERGED_DATASET_LATEST2/MERGED_DATASET_LATEST2/merged_manifest.json"
DEFAULT_TARGET_PATH = "editevalpp_results/layers_8_11_14_17_20_23_sgs_1p6_gs_3p6_steps_27_skip_7"
DEFAULT_MODEL_PATH = (
    "/data/disk2/haiyan/hf_models/sd3_5_medium/"
    "models--stabilityai--stable-diffusion-3.5-medium/"
    "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
)


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


def save_mask(editing_mask, mask_path):
    if editing_mask is None:
        return False
    mask_np = (editing_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize((mask_img.width * 8, mask_img.height * 8), Image.NEAREST)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(mask_path)
    return True


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run MIEdit on EditEval++ with PIE-Bench default hyperparameters")
    parser.add_argument("--manifest_path", type=str, default=DEFAULT_MANIFEST_PATH,
                        help="Path to merged_manifest.json")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Dataset root. Defaults to the directory containing merged_manifest.json")
    parser.add_argument("--target_path", type=str, default=DEFAULT_TARGET_PATH,
                        help="Output directory where edited images and masks will be saved")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Local path to the Stable Diffusion 3.5 model directory")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=27)
    parser.add_argument("--guidance_scale", type=float, default=3.6)
    parser.add_argument("--source_guidance_scale", type=float, default=1.6)
    parser.add_argument("--skip_steps", type=int, default=7)
    parser.add_argument("--generator_seed", type=int, default=2)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--edit_type_list", nargs="+", type=str, default=None,
                        help="Optional subset of EditEval++ edit_type values to process")
    parser.add_argument("--source_prompt_field", type=str, default="source_description")
    parser.add_argument("--target_prompt_field", type=str, default="target_description")
    parser.add_argument("--mask_feature_layers", type=str, default="8,11,14,17,20,23")
    parser.add_argument("--disable_h_zero_cache", action="store_true")
    parser.add_argument("--disable_deterministic", action="store_true")
    parser.add_argument("--cleanup_interval", type=int, default=0)
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="JSONL path for processed sample metadata. Defaults to target_path/run_manifest.jsonl")
    parser.add_argument("--log_prefix", type=str, default=None,
                        help="Prefix for each log line. Defaults to the target directory name")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else manifest_path.parent
    target_root = Path(args.target_path).expanduser().resolve()
    image_root = target_root / "annotation_images"
    mask_root = target_root / "masks"
    metadata_path = Path(args.metadata_path).expanduser().resolve() if args.metadata_path else target_root / "run_manifest.jsonl"
    log_prefix = args.log_prefix if args.log_prefix is not None else f"[{target_root.name}] "

    def log(message=""):
        print(f"{log_prefix}{message}", flush=True)

    manifest = load_manifest(manifest_path)
    if args.edit_type_list:
        keep = set(args.edit_type_list)
        manifest = [item for item in manifest if item.get("edit_type") in keep]

    start_idx = max(args.start_idx, 0)
    end_idx = args.end_idx if args.end_idx is not None else len(manifest)
    end_idx = min(end_idx, len(manifest))
    items = manifest[start_idx:end_idx]

    mask_feature_layers = parse_mask_feature_layers(args.mask_feature_layers)
    log(f"Manifest: {manifest_path}")
    log(f"Dataset root: {dataset_root}")
    log(f"Output root: {target_root}")
    log(f"Samples: {len(items)} (index {start_idx} to {end_idx})")
    log(f"Using mask feature layers: {mask_feature_layers}")

    set_deterministic(seed=args.generator_seed, enable=not args.disable_deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(
        device=device,
        torch_dtype=torch.float16,
        model_path=args.model_path,
        use_h_zero_cache=not args.disable_h_zero_cache,
    )

    rows = []
    success_count = 0

    for local_idx, item in enumerate(items, start=start_idx):
        unified_id = item.get("unified_id") or f"sample_{local_idx:06d}"
        log("")
        log("=" * 60)
        log(f"Processing [{local_idx + 1}/{len(manifest)}] {unified_id}")

        try:
            image_path = resolve_image_path(item, dataset_root)
            rel_path = get_output_relative_path(item, image_path)
            out_path = image_root / rel_path
            mask_path = (mask_root / rel_path).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                log(f"Skipping existing output: {out_path}")
                continue

            source_prompt = clean_prompt(item.get(args.source_prompt_field))
            target_prompt = clean_prompt(item.get(args.target_prompt_field))
            if not source_prompt or not target_prompt:
                raise ValueError(
                    f"Missing prompts for {unified_id}: "
                    f"{args.source_prompt_field}={source_prompt!r}, "
                    f"{args.target_prompt_field}={target_prompt!r}"
                )

            log(f"  Image: {image_path}")
            log(f"  Output image: {out_path}")
            log(f"  Output mask:  {mask_path}")
            log(f"  Source prompt: {source_prompt}")
            log(f"  Target prompt: {target_prompt}")

            imagein = Image.open(image_path).convert("RGB")
            imagein.thumbnail((args.width, args.height), Image.Resampling.LANCZOS)
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

            output_image = result.get("generated_image")
            if output_image is None:
                raise RuntimeError("generate_with_gt_mask_editing did not return generated_image")

            output_image.save(out_path)
            has_mask = save_mask(result.get("editing_mask"), mask_path)
            log(f"  Saved image: {out_path}")
            if has_mask:
                log(f"  Saved mask:  {mask_path}")

            rows.append({
                "index": local_idx,
                "unified_id": unified_id,
                "edit_type": item.get("edit_type"),
                "source_image": str(image_path),
                "output_image": str(out_path),
                "output_mask": str(mask_path) if has_mask else None,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "edit_instruction": item.get("edit_instruction"),
            })

            del output_image
            del imagein
            success_count += 1
            if args.cleanup_interval > 0 and success_count % args.cleanup_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write_jsonl(metadata_path, rows)
                log(f"  Periodic cleanup and metadata save at sample {success_count}")

        except Exception as exc:
            log(f"Error processing {unified_id}: {exc}")
            import traceback
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    write_jsonl(metadata_path, rows)
    log("")
    log("=" * 60)
    log(f"Processing complete. Successful samples: {success_count}")
    log(f"Images saved to: {image_root}")
    log(f"Masks saved to: {mask_root}")
    log(f"Metadata saved to: {metadata_path}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
