"""
SD3 Editing Pipeline — external code built on top of diffusers.

Subclasses the stock StableDiffusion3Pipeline and adds only the
`generate_with_gt_mask_editing` method (plus its helpers).
All standard pipeline functionality (from_pretrained, __call__, encode_prompt, …)
is inherited from diffusers.

Optimised for GPU-rich environments (no unnecessary CPU↔GPU copies).
"""

import inspect
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline

from custom_attention_processor import (
    replace_sd3_attention_with_kv_replacement,
    update_sd3_kv_processors,
    restore_sd3_original_attention_processors,
    SD3FeatureMapStore,
    register_sd3_feature_hooks,
    cleanup_sd3_feature_hooks,
    clear_sd3_stored_kv,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        if "timesteps" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError("Scheduler does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError("Scheduler does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Mask-generation helper (CRF + Otsu) — factored out to avoid code duplication
# ---------------------------------------------------------------------------

def _sigmoid_crf_otsu(abs_diff: np.ndarray):
    """Sigmoid normalise → CRF → Otsu.  Returns refined_prob and otsu_norm, or None on failure."""
    H, W = abs_diff.shape
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian
        import cv2

        if abs_diff.max() - abs_diff.min() < 1e-6:
            pseudo_prob = np.ones_like(abs_diff) * 0.5
        else:
            k = 5.0
            diff_mean = np.mean(abs_diff)
            diff_std = np.std(abs_diff)
            diff_std = min(diff_std, np.percentile(abs_diff, 90) - np.percentile(abs_diff, 10) + 1e-8)
            if diff_std < 1e-7:
                normalized = np.zeros_like(abs_diff)
            else:
                diff_std = max(diff_std, 1e-6)
                normalized = (abs_diff - diff_mean) / (3 * diff_std)
            pseudo_prob = 1.0 / (1.0 + np.exp(-k * normalized))
        pseudo_prob = np.clip(pseudo_prob, 0, 1).astype(np.float32)

        unary_probs = np.stack([1 - pseudo_prob, pseudo_prob], axis=0)
        unary = unary_from_softmax(unary_probs)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary.astype(np.float32))
        spatial_std = max(1.0, min(5.0, max(H, W) / 32.0))
        feats = create_pairwise_gaussian(sdims=(spatial_std, spatial_std), shape=(H, W))
        d.addPairwiseEnergy(feats, compat=3.0, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        fi = np.zeros((H * W, 1), dtype=np.float32)
        fi[:, 0] = pseudo_prob.flatten()
        d.addPairwiseEnergy(fi.T, compat=1.0, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        refined_prob = np.array(Q).reshape((2, H, W))[1]

        rp_u8 = (refined_prob * 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(rp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return refined_prob, otsu_thresh / 255.0

    except Exception as e:
        print(f"[Mask] CRF failed ({e}), falling back to raw Otsu")
        import cv2
        norm = (abs_diff - abs_diff.min()) / (abs_diff.max() - abs_diff.min() + 1e-8)
        u8 = (norm * 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return norm, otsu_thresh / 255.0


# ---------------------------------------------------------------------------
# SD3 Editing Pipeline
# ---------------------------------------------------------------------------

class SD3EditingPipeline(StableDiffusion3Pipeline):
    """
    Extends ``StableDiffusion3Pipeline`` with ``generate_with_gt_mask_editing``.
    """

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def prepare_attention_mask_for_latent(
        self, attention_mask, target_size=(64, 64), blur_radius=2.0,
    ):
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).float()

        if attention_mask.shape[-2:] != target_size:
            attention_mask = F.interpolate(
                attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask.dim() == 2 else attention_mask,
                size=target_size, mode="bilinear", align_corners=False,
            ).squeeze()

        if blur_radius > 0:
            ks = int(2 * blur_radius) + 1
            sigma = blur_radius / 2
            coords = torch.arange(ks, dtype=attention_mask.dtype, device=attention_mask.device) - ks // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask.dim() == 2 else attention_mask
            attention_mask = F.conv2d(attention_mask, g.view(1, 1, 1, -1), padding="same")
            attention_mask = F.conv2d(attention_mask, g.view(1, 1, -1, 1), padding="same")
            attention_mask = attention_mask.squeeze()

        return attention_mask.unsqueeze(0).unsqueeze(0)

    def mix_latents_with_mask(self, generation_latent, inversion_latent, mask):
        if mask.shape[1] == 1:
            mask = mask.expand(-1, generation_latent.shape[1], -1, -1)
        return mask * generation_latent + (1 - mask) * inversion_latent

    def _morphological_close_mask(self, binary_mask, kernel_size=5, dilation_iter=2):
        try:
            import cv2
        except ImportError:
            return binary_mask
        mu8 = (binary_mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, kernel, iterations=1)
        if dilation_iter > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.dilate(closed, dk, iterations=dilation_iter)
        return (closed / 255.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Main entry-point
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate_with_gt_mask_editing(
        self,
        image,
        source_prompt: str,
        target_prompt: str,
        num_inference_steps: int = 30,
        source_guidance_scale: float = 3.5,
        target_guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        generator: Optional[torch.Generator] = None,
        save_image: bool = False,
        save_dir: str = "./sd3_edited_images",
        use_kv_replacement: bool = True,
        skip_steps: int = 5,
        use_latent_mixing: bool = True,
        mixing_start_step: int = 3,
        mixing_end_step: int = 30,
        max_sequence_length: int = 256,
        image_name: str = "image",
        aggregation_method: str = "cosine_dissimilarity",
        cfg_diff_steps: Optional[list] = None,
        mask_feature_layers: Optional[List[str]] = None,
        kv_start_layer: int = 11,
        kv_save_steps_offset: int = 0,
    ):
        """Image editing via inversion -> mask generation -> guided re-generation."""
        device = self.transformer.device
        batch_size = 1

        # ===================== 1. Encode prompts =====================

        source_embeds, negative_source_embeds, source_pooled, neg_source_pooled = self.encode_prompt(
            prompt=source_prompt, prompt_2=source_prompt, prompt_3=source_prompt,
            negative_prompt=target_prompt, negative_prompt_2=target_prompt, negative_prompt_3=target_prompt,
            do_classifier_free_guidance=True, device=device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )
        target_embeds, negative_target_embeds, target_pooled, neg_target_pooled = self.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device,
            num_images_per_prompt=1, max_sequence_length=max_sequence_length,
        )

        # ===================== 2. Scheduler =====================
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            lh = int(height) // self.vae_scale_factor
            lw = int(width) // self.vae_scale_factor
            ps = self.transformer.config.patch_size
            image_seq_len = (lh // ps) * (lw // ps)
            scheduler_kwargs["mu"] = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, **scheduler_kwargs,
        )
        num_steps = len(timesteps)  # pre-compute once

        # ===================== 3. Inversion phase =====================

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.preprocess(image, height=height, width=width)
        image = image.to(device=device, dtype=self.transformer.dtype)

        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        xts = self.scheduler.sample_xts_from_x0(init_latents, num_inference_steps=num_steps)

        # Install KV processors (first call — creates objects)
        stored_kv: Dict = {}
        replace_sd3_attention_with_kv_replacement(
            self.transformer, save_kv=True, stored_kv=stored_kv, kv_start_layer=kv_start_layer,
        )
        # Build a flat cache of processor references for fast updates later
        _proc_cache: list = []
        update_sd3_kv_processors(
            self.transformer, save_kv=True, stored_kv=stored_kv,
            current_step=0, kv_start_layer=kv_start_layer, _cache=_proc_cache,
        )

        # Feature hooks
        inv_feat_store = SD3FeatureMapStore(selected_layers=mask_feature_layers)
        inv_hooks = register_sd3_feature_hooks(self.transformer, inv_feat_store)
        if mask_feature_layers is not None and len(inv_hooks) != len(mask_feature_layers):
            raise ValueError(
                "Some mask_feature_layers were not found in the transformer: "
                f"requested={mask_feature_layers}, registered_hooks={len(inv_hooks)}"
            )

        # CFG embeddings for source
        do_cfg_src = source_guidance_scale > 1.0
        if do_cfg_src:
            src_emb_cfg = torch.cat([negative_source_embeds, source_embeds], dim=0)
            src_pool_cfg = torch.cat([neg_source_pooled, source_pooled], dim=0)
        else:
            src_emb_cfg = source_embeds
            src_pool_cfg = source_pooled

        # Bookkeeping
        noise_list1 = [None] * num_steps
        noise_list2 = [None] * num_steps
        noise_pred_list = []

        if cfg_diff_steps is None:
            cfg_diff_steps = [7, 8, 9, 10, 11, 12]
        cfg_diff_set = set(cfg_diff_steps)        # O(1) lookup
        max_cfg_step = max(cfg_diff_steps)
        # cfg_diff_steps logged at the start

        accumulated_cfg_diffs = []
        final_mask_kv = None
        final_mask_cfg = None

        max_save_step = num_steps - skip_steps - 1   # pre-compute

        # How many inversion transformer calls we actually need:
        # 1) scheduler.invert needs model_output_list entries (predictor order P, corrector order C)
        # 2) model_outputs warmup needs entries for the last `warmup_cap` skip steps
        max_invert_step = num_steps - skip_steps  # scheduler.invert runs steps 0..this-1
        pred_order = self.scheduler.config.get("predictor_order", 2)
        corr_order = self.scheduler.config.get("corrector_order", 2)
        extra_for_invert = max(pred_order - 1, corr_order - 2)  # 2 for pred=3, corr=4
        warmup_cap = max(pred_order, corr_order - 1)             # 3 for pred=3, corr=4
        max_inv_transformer_steps = min(num_steps, max_invert_step + extra_for_invert)
        # Inversion step count logged

        # Materialise reversed timesteps as a list
        inv_timesteps = list(reversed(timesteps))

        for i, t in enumerate(inv_timesteps):
            if i >= max_inv_transformer_steps:
                break  # remaining steps not needed (editing skip steps use xts directly)

            should_save = kv_save_steps_offset <= i <= max_save_step
            # Fast in-place update (no tree walk)
            update_sd3_kv_processors(
                self.transformer, save_kv=should_save, stored_kv=stored_kv,
                current_step=i, kv_start_layer=kv_start_layer, _cache=_proc_cache,
            )

            xtm1 = xts[i + 1][None]
            lat_in = torch.cat([xtm1, xtm1], dim=0) if do_cfg_src else xtm1

            ts = t if torch.is_tensor(t) and t.dim() > 0 else torch.tensor([t], device=device)
            ts = ts.expand(lat_in.shape[0])

            noise_pred = self.transformer(
                hidden_states=lat_in, timestep=ts,
                encoder_hidden_states=src_emb_cfg, pooled_projections=src_pool_cfg,
                return_dict=False,
            )[0]

            # ---------- CFG-difference mask ----------
            if i in cfg_diff_set and do_cfg_src:
                raw_diff = inv_feat_store.compute_raw_cfg_difference(aggregation_method=aggregation_method)
                if raw_diff is not None:
                    accumulated_cfg_diffs.append(raw_diff.copy())
                    del raw_diff

            # ---------- CFG ----------
            if do_cfg_src:
                np_uncond, np_text = noise_pred.chunk(2)

                if i == max_cfg_step and accumulated_cfg_diffs:
                    avg_diff = np.mean(accumulated_cfg_diffs, axis=0)
                    abs_d = np.abs(avg_diff)
                    refined, otsu_n = _sigmoid_crf_otsu(abs_d)

                    mask_cfg_np = (refined > otsu_n).astype(np.float32)
                    mask_kv_np = self._morphological_close_mask(
                        (refined > otsu_n).astype(np.float32), kernel_size=3,
                    )

                    cleanup_sd3_feature_hooks(inv_hooks)
                    restore_sd3_original_attention_processors(self.transformer)
                    inv_feat_store.clear()
                    del inv_hooks, inv_feat_store, accumulated_cfg_diffs

                    # Re-install processors (they were just restored) and rebuild cache
                    replace_sd3_attention_with_kv_replacement(
                        self.transformer, save_kv=True, stored_kv=stored_kv, kv_start_layer=kv_start_layer,
                    )
                    _proc_cache.clear()
                    update_sd3_kv_processors(
                        self.transformer, save_kv=True, stored_kv=stored_kv,
                        current_step=i, kv_start_layer=kv_start_layer, _cache=_proc_cache,
                    )

                    final_mask_kv = torch.from_numpy(mask_kv_np).to(device=device, dtype=noise_pred.dtype)
                    final_mask_cfg = torch.from_numpy(mask_cfg_np).to(device=device, dtype=noise_pred.dtype)

                noise_pred = np_uncond + source_guidance_scale * (np_text - np_uncond)

            model_out = self.scheduler.convert_model_output(
                noise_pred, sample=xtm1, timestep=ts, k=num_steps - i - 1,
            )
            noise_pred_list.append(model_out)
            del noise_pred, xtm1, lat_in

        # ---------- scheduler.invert ----------
        # 20 steps (0..19): produces noise_list1[0..19] and noise_list2[0..19].
        # After reversal, editing step 7 uses noise_list1[19] ✓, n2=0 (first non-skip).
        # scheduler.invert phase
        ts_with_zero = torch.cat([
            torch.tensor([0], device=device, dtype=timesteps.dtype),
            torch.tensor(inv_timesteps, device=device),
        ])
        for i, t in enumerate(ts_with_zero[:max_invert_step]):
            k = len(ts_with_zero) - 2 - i
            _, latents_inv, n1, n2, xt2 = self.scheduler.invert(
                rela=xts[i + 1][None], model_output_list=noise_pred_list,
                sample=xts[i][None], k=k, zhengxu=i + 1,
                timestep=t, xTtimestep=ts_with_zero[i + 1], x_0=init_latents,
            )
            noise_list1[i] = n1
            noise_list2[i] = n2
            xts[i] = xt2

        restore_sd3_original_attention_processors(self.transformer)

        # Reverse noise lists (index 0 = first generation step)
        noise_list1 = noise_list1[::-1]
        noise_list2 = noise_list2[::-1]

        # Inversion complete

        # ===================== 4. Editing phase =====================

        # Reset scheduler state and simulate warmup from skip steps.
        # Pre-fill model_outputs with inversion's converted predictions (same sigma,
        # sample differs by one xts index — negligible at high noise levels).
        # Only the last 2 slots are needed: slot0 gets shifted out at the first step.
        n_history = min(skip_steps, warmup_cap - 1)  # 2 history entries needed (slot1, slot2)
        self.scheduler.lower_order_nums = min(skip_steps, warmup_cap)
        self.scheduler.last_sample = None
        self.scheduler._step_index = None

        self.scheduler.timestep_list = [None] * warmup_cap
        self.scheduler.model_outputs = [None] * warmup_cap
        for j in range(n_history):
            # Fill the last n_history slots (newest at [-1])
            edit_step = skip_steps - n_history + j       # e.g. 5, 6 for n_history=2
            inv_idx = num_steps - 1 - edit_step          # e.g. 21, 20
            slot = warmup_cap - n_history + j            # e.g. 1, 2
            self.scheduler.model_outputs[slot] = noise_pred_list[inv_idx]
            self.scheduler.timestep_list[slot] = timesteps[edit_step]

        # Start from the noisiest xts (editing skip steps will directly follow xts trajectory)
        current_latents = xts[num_steps][None].to(device=device, dtype=self.transformer.dtype)

        do_cfg_tgt = target_guidance_scale > 1.0
        if do_cfg_tgt:
            tgt_emb_cfg = torch.cat([negative_target_embeds, target_embeds], dim=0)
            tgt_pool_cfg = torch.cat([neg_target_pooled, target_pooled], dim=0)
        else:
            tgt_emb_cfg, tgt_pool_cfg = target_embeds, target_pooled

        # Install KV-replacement processors for editing + cache
        replace_sd3_attention_with_kv_replacement(
            self.transformer, use_kv_replacement=False, stored_kv=stored_kv, kv_start_layer=kv_start_layer,
        )
        _proc_cache_edit: list = []
        update_sd3_kv_processors(
            self.transformer, use_kv_replacement=False, stored_kv=stored_kv,
            kv_start_layer=kv_start_layer, _cache=_proc_cache_edit,
        )

        min_kv_saved = kv_save_steps_offset
        max_kv_saved = num_steps - skip_steps - 1

        for step_idx, t in enumerate(timesteps):
            inv_step = num_steps - 1 - step_idx

            # --- Skip steps: directly use inversion trajectory (no transformer, no scheduler.step) ---
            if step_idx < skip_steps:
                # The inversion latent at this position IS the correct trajectory.
                # No transformer call, no noise1/noise2, no scheduler.step needed.
                xt_i = xts[inv_step]
                # xts entries may be 3-D (original from sample_xts_from_x0) or
                # 4-D (updated by scheduler.invert with batch dim). Normalise to 4-D.
                if xt_i.dim() == 3:
                    xt_i = xt_i.unsqueeze(0)
                current_latents = xt_i.to(device=device, dtype=current_latents.dtype)
                continue

            # --- Non-skip steps: full editing with KV replacement ---
            n1 = noise_list1[step_idx]
            # n2 comes from the "previous corrector". For the first non-skip step,
            # there was no previous scheduler.step (skip steps use xts directly), so n2=0.
            if step_idx <= skip_steps:
                n2 = 0
            else:
                n2 = noise_list2[step_idx - 1]

            ts = t if torch.is_tensor(t) and t.dim() > 0 else torch.tensor([t], device=device)
            ts_batch = ts.expand(batch_size)

            # Check if this is a zero-step (h≈0): gradient=0, model output unused
            skip_transformer = (
                getattr(self.scheduler, "use_h_zero_cache", True)
                and
                self.scheduler.step_index is not None
                and self.scheduler.check_h_zero(self.scheduler.step_index)
            )

            if skip_transformer:
                noise_pred = torch.zeros_like(current_latents)
            else:
                # KV replacement decision
                should_use_kv = (
                    use_kv_replacement
                    and min_kv_saved <= inv_step <= max_kv_saved
                )
                kv_mask = None
                if should_use_kv:
                    if final_mask_kv is not None:
                        kv_mask = final_mask_kv.unsqueeze(0) if final_mask_kv.dim() == 2 else final_mask_kv
                    else:
                        sh = height // (self.vae_scale_factor * self.transformer.config.patch_size)
                        sw = width // (self.vae_scale_factor * self.transformer.config.patch_size)
                        kv_mask = torch.ones(1, 1, sh, sw, dtype=torch.float32, device=device)

                # Fast in-place update
                update_sd3_kv_processors(
                    self.transformer, use_kv_replacement=should_use_kv,
                    kv_replacement_mask=kv_mask, stored_kv=stored_kv,
                    current_step=inv_step, kv_start_layer=kv_start_layer,
                    _cache=_proc_cache_edit,
                )

                # Forward pass (transformer)
                if do_cfg_tgt:
                    lat_in = torch.cat([current_latents, current_latents], dim=0).to(tgt_emb_cfg.dtype)
                    ts_cfg = ts.expand(lat_in.shape[0])
                    noise_pred = self.transformer(
                        hidden_states=lat_in, timestep=ts_cfg,
                        encoder_hidden_states=tgt_emb_cfg, pooled_projections=tgt_pool_cfg,
                        return_dict=False,
                    )[0]
                    np_u, np_t = noise_pred.chunk(2)
                    noise_pred = np_u + target_guidance_scale * (np_t - np_u)
                    del lat_in, np_u, np_t
                else:
                    noise_pred = self.transformer(
                        hidden_states=current_latents, timestep=ts_batch,
                        encoder_hidden_states=target_embeds, pooled_projections=target_pooled,
                        return_dict=False,
                    )[0]

            # Scheduler step
            current_latents = self.scheduler.step(
                noise_pred, t, current_latents, return_dict=False, noise1=n1, noise2=n2,
            )[0]

            # Latent mixing
            if use_latent_mixing and final_mask_cfg is not None:
                if mixing_start_step <= step_idx <= mixing_end_step:
                    inv_mix = inv_step
                    if inv_mix < len(xts):
                        inv_lat = xts[inv_mix].to(device)
                        lh, lw = current_latents.shape[-2:]
                        lat_mask = self.prepare_attention_mask_for_latent(
                            final_mask_cfg, target_size=(lh, lw), blur_radius=2.0,
                        ).to(device)
                        current_latents = self.mix_latents_with_mask(current_latents, inv_lat, lat_mask)

            del noise_pred

        restore_sd3_original_attention_processors(self.transformer)

        # ===================== 5. Decode =====================
        lat_dec = (current_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        lat_dec = lat_dec.to(self.vae.dtype)
        gen_img = self.vae.decode(lat_dec, return_dict=False)[0]
        gen_img = self.image_processor.postprocess(gen_img, output_type="pil")[0]

        if save_image:
            os.makedirs(save_dir, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in target_prompt)[:60].strip()
            gen_img.save(os.path.join(save_dir, f"sd3_edited_{safe}.png"))

        # ===================== 6. Cleanup =====================
        clear_sd3_stored_kv(stored_kv)
        del stored_kv

        return {
            "generated_image": gen_img,
            "editing_mask": final_mask_kv.cpu() if final_mask_kv is not None else None,
        }
