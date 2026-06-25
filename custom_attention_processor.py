"""
SD3 Joint Attention KV Replacement Processor and utilities.

Optimised for GPU-rich environments:
- Stored K/V kept on GPU (no CPU round-trip)
- Feature maps kept on GPU
- Processors updated in-place (no recursive tree walk every step)
- Layer number parsed once and cached
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# =====================================================================
# Attention Processor
# =====================================================================

class JointKVReplacementProcessor:
    """
    Joint Attention Processor with K,V replacement support.

    During the *inversion* pass  (save_kv=True)  it stores per-layer, per-step K and V.
    During the *editing* pass   (use_kv_replacement=True) it mixes the stored
    (inversion) K,V with the current (generation) K,V according to a spatial mask.
    """

    def __init__(
        self,
        save_kv: bool = False,
        use_kv_replacement: bool = False,
        kv_replacement_mask: Optional[torch.Tensor] = None,
        stored_kv: Optional[Dict] = None,
        current_step: int = 0,
        kv_start_layer: int = 11,
        layer_num: int = -1,          # pre-parsed, avoids string splitting
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")

        self.save_kv = save_kv
        self.use_kv_replacement = use_kv_replacement
        self.kv_replacement_mask = kv_replacement_mask
        self.stored_kv = stored_kv if stored_kv is not None else {}
        self.current_step = current_step
        self.kv_start_layer = kv_start_layer
        self.layer_num = layer_num     # cached — set once at install time
        self._should_apply: bool = (layer_num >= kv_start_layer) if layer_num >= 0 else False

    # ------------------------------------------------------------------ #

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # 1. Image Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 2. KV save / replacement (fast path: cached _should_apply)
        layer_name = getattr(attn, "_layer_name", None)

        if self._should_apply and layer_name is not None:
            # 2a. Save K, V (inversion phase) — stay on GPU
            if self.save_kv:
                key_flat = key.transpose(1, 2).reshape(batch_size, -1, inner_dim)
                value_flat = value.transpose(1, 2).reshape(batch_size, -1, inner_dim)

                if batch_size > 1:
                    half = batch_size // 2
                    kv_entry = {
                        "key_positive": key_flat[half:].detach().clone(),
                        "value_positive": value_flat[half:].detach().clone(),
                    }
                else:
                    kv_entry = {
                        "key": key_flat.detach().clone(),
                        "value": value_flat.detach().clone(),
                    }

                if layer_name not in self.stored_kv:
                    self.stored_kv[layer_name] = []
                store = self.stored_kv[layer_name]
                while len(store) <= self.current_step:
                    store.append(None)
                store[self.current_step] = kv_entry

            # 2b. Use KV replacement (editing phase)
            if (
                self.use_kv_replacement
                and layer_name in self.stored_kv
                and self.kv_replacement_mask is not None
                and self.current_step < len(self.stored_kv[layer_name])
            ):
                kv_data = self.stored_kv[layer_name][self.current_step]
                if kv_data is not None:
                    key_img = key.transpose(1, 2).reshape(batch_size, -1, inner_dim)
                    value_img = value.transpose(1, 2).reshape(batch_size, -1, inner_dim)

                    if batch_size > 1:
                        half = batch_size // 2
                        stored_key_pos = kv_data.get("key_positive", kv_data.get("key"))
                        stored_val_pos = kv_data.get("value_positive", kv_data.get("value"))

                        key_neg, key_pos = key_img[:half], key_img[half:]
                        val_neg, val_pos = value_img[:half], value_img[half:]

                        key_neg, val_neg = self._mix_kv(
                            key_neg, val_neg, stored_key_pos, stored_val_pos, self.kv_replacement_mask,
                        )
                        key_pos, val_pos = self._mix_kv(
                            key_pos, val_pos, stored_key_pos, stored_val_pos, self.kv_replacement_mask,
                        )
                        key_img = torch.cat([key_neg, key_pos], dim=0)
                        value_img = torch.cat([val_neg, val_pos], dim=0)
                    else:
                        stored_key = kv_data.get("key")
                        stored_val = kv_data.get("value")
                        if stored_key is not None and stored_val is not None:
                            key_img, value_img = self._mix_kv(
                                key_img, value_img, stored_key, stored_val, self.kv_replacement_mask,
                            )

                    key = key_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    value = value_img.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # Free consumed KV data
                    self.stored_kv[layer_name][self.current_step] = None

        # 3. Text Q, K, V  +  concatenation
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)

            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)

            query = torch.cat([query, eq], dim=2)
            key = torch.cat([key, ek], dim=2)
            value = torch.cat([value, ev], dim=2)

        # 4. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # 5. Split outputs
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # 6. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states

    # ------------------------------------------------------------------ #

    @staticmethod
    def _mix_kv(key_gen, value_gen, key_inv, value_inv, mask):
        """Mix generation and inversion K,V according to a spatial mask.

        mask=1 -> use generation (foreground),  mask=0 -> use inversion (background).
        All tensors are expected on the same device (GPU).
        """
        dtype = key_gen.dtype
        # Stored KV is already on GPU; just cast dtype if needed
        if key_inv.dtype != dtype:
            key_inv = key_inv.to(dtype=dtype)
            value_inv = value_inv.to(dtype=dtype)

        B, N, _C = key_gen.shape
        m = mask
        if m.dtype != dtype:
            m = m.to(dtype=dtype)

        mask_flat = m.view(B, -1, 1)
        if mask_flat.shape[1] != N:
            target_H = target_W = int(N ** 0.5)
            m_spatial = m.squeeze(1) if m.dim() > 2 else m
            m_spatial = F.interpolate(
                m_spatial.unsqueeze(0).unsqueeze(0) if m_spatial.dim() == 2 else m_spatial.unsqueeze(1),
                size=(target_H, target_W),
                mode="bilinear",
                align_corners=False,
            )
            mask_flat = m_spatial.view(B, -1, 1)

        key_mixed = mask_flat * key_gen + (1.0 - mask_flat) * key_inv
        value_mixed = mask_flat * value_gen + (1.0 - mask_flat) * value_inv
        return key_mixed, value_mixed


# =====================================================================
# Processor installation helpers  (optimised: in-place update)
# =====================================================================

def _parse_layer_num(name: str) -> int:
    """Extract the integer block index from a name like 'transformer_blocks.14.attn'."""
    if "transformer_blocks." not in name:
        return -1
    try:
        return int(name.split("transformer_blocks.")[1].split(".")[0])
    except (ValueError, IndexError):
        return -1


def replace_sd3_attention_with_kv_replacement(
    model,
    save_kv: bool = False,
    use_kv_replacement: bool = False,
    kv_replacement_mask: Optional[torch.Tensor] = None,
    stored_kv: Optional[Dict] = None,
    current_step: int = 0,
    kv_start_layer: int = 11,
):
    """Install or update KV-replacement processors on *model*.

    On the first call new ``JointKVReplacementProcessor`` objects are created.
    On subsequent calls (when the processor is already the right type) we only
    **update the mutable attributes** — avoiding a full recursive tree walk and
    object re-creation.
    """
    if stored_kv is None:
        stored_kv = {}

    def _install_or_update(module, name=""):
        if hasattr(module, "processor"):
            proc = module.processor
            if isinstance(proc, JointKVReplacementProcessor):
                # Fast path — update in place
                proc.save_kv = save_kv
                proc.use_kv_replacement = use_kv_replacement
                proc.kv_replacement_mask = kv_replacement_mask
                proc.stored_kv = stored_kv
                proc.current_step = current_step
                proc.kv_start_layer = kv_start_layer
                proc._should_apply = (proc.layer_num >= kv_start_layer) if proc.layer_num >= 0 else False
            else:
                # First call — create
                layer_num = _parse_layer_num(name)
                module._layer_name = name
                module.processor = JointKVReplacementProcessor(
                    save_kv=save_kv,
                    use_kv_replacement=use_kv_replacement,
                    kv_replacement_mask=kv_replacement_mask,
                    stored_kv=stored_kv,
                    current_step=current_step,
                    kv_start_layer=kv_start_layer,
                    layer_num=layer_num,
                )
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            _install_or_update(child, full_name)

    _install_or_update(model)
    return stored_kv


def update_sd3_kv_processors(
    model,
    *,
    save_kv: bool = False,
    use_kv_replacement: bool = False,
    kv_replacement_mask: Optional[torch.Tensor] = None,
    stored_kv: Optional[Dict] = None,
    current_step: int = 0,
    kv_start_layer: int = 11,
    _cache: Optional[List] = None,
):
    """Ultra-fast path: update *already-installed* processors via a cached list.

    On the first call, pass ``_cache=[]``; the function populates it with
    references to every ``JointKVReplacementProcessor``.  On subsequent calls
    pass the same list — the recursive tree walk is skipped entirely.
    """
    if _cache is not None and len(_cache) > 0:
        # ~0 overhead: just iterate the flat list
        for proc in _cache:
            proc.save_kv = save_kv
            proc.use_kv_replacement = use_kv_replacement
            proc.kv_replacement_mask = kv_replacement_mask
            if stored_kv is not None:
                proc.stored_kv = stored_kv
            proc.current_step = current_step
            proc.kv_start_layer = kv_start_layer
            proc._should_apply = (proc.layer_num >= kv_start_layer) if proc.layer_num >= 0 else False
        return

    # First call — collect processors into cache
    if _cache is None:
        _cache = []

    for module in model.modules():
        if hasattr(module, "processor") and isinstance(module.processor, JointKVReplacementProcessor):
            proc = module.processor
            proc.save_kv = save_kv
            proc.use_kv_replacement = use_kv_replacement
            proc.kv_replacement_mask = kv_replacement_mask
            if stored_kv is not None:
                proc.stored_kv = stored_kv
            proc.current_step = current_step
            proc.kv_start_layer = kv_start_layer
            proc._should_apply = (proc.layer_num >= kv_start_layer) if proc.layer_num >= 0 else False
            _cache.append(proc)


def restore_sd3_original_attention_processors(model):
    """Restore the default JointAttnProcessor2_0 on every attention module."""
    from diffusers.models.attention_processor import JointAttnProcessor2_0

    for module in model.modules():
        if hasattr(module, "processor") and isinstance(module.processor, JointKVReplacementProcessor):
            module.processor = JointAttnProcessor2_0()
            if hasattr(module, "_layer_name"):
                delattr(module, "_layer_name")


def clear_sd3_stored_kv(stored_kv: Dict):
    """Release all tensors held in *stored_kv* and free GPU memory."""
    for layer_name in list(stored_kv.keys()):
        steps = stored_kv[layer_name]
        if steps:
            for i, kv_dict in enumerate(steps):
                if kv_dict:
                    for v in kv_dict.values():
                        del v
                    steps[i] = None
            stored_kv[layer_name] = None
    stored_kv.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# Feature-map store  (for CFG-difference mask generation)
# =====================================================================

SD3_KEY_LAYERS = [
    "transformer_blocks.6",
    "transformer_blocks.10",
    "transformer_blocks.14",
    "transformer_blocks.18",
    "transformer_blocks.22",
]


class SD3FeatureMapStore:
    """Collect and process intermediate SD3 transformer feature maps.

    Feature maps are kept **on GPU** to avoid CPU↔GPU copies.
    """

    def __init__(self, selected_layers=None):
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.layer_names: list = []
        self.selected_layers = selected_layers or SD3_KEY_LAYERS

    def clear(self):
        self.feature_maps.clear()
        self.layer_names.clear()

    def store_feature_map(
        self,
        layer_name: str,
        feature_data: torch.Tensor,
        num_image_tokens: Optional[int] = None,
    ):
        """Store the image-token portion of a feature map (stays on GPU)."""
        if self.selected_layers is not None and layer_name not in self.selected_layers:
            return

        if num_image_tokens is not None:
            feature_data_image = feature_data[:, :num_image_tokens, :]
        else:
            seq_len = feature_data.shape[1]
            spatial_size = int(seq_len ** 0.5)
            if spatial_size * spatial_size == seq_len:
                feature_data_image = feature_data
            else:
                possible_sizes = [64, 32, 16, 8]
                feature_data_image = feature_data
                for img_size in possible_sizes:
                    img_tokens = img_size * img_size
                    if seq_len >= img_tokens and (seq_len - img_tokens) <= 512:
                        feature_data_image = feature_data[:, :img_tokens, :]
                        break

        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
        # Keep on GPU — detach only (no .cpu())
        self.feature_maps[layer_name] = feature_data_image.detach()

    def compute_raw_cfg_difference(self, aggregation_method: str = "l2_norm"):
        """Compute a spatial heatmap of the CFG (neg vs pos) difference.

        All computation stays on GPU; only the final small (H,W) result is
        moved to CPU/numpy.
        """
        import numpy as np

        if not self.feature_maps:
            return None

        # Collect all diffs as GPU tensors first, only move to CPU at the end
        gpu_diffs: list = []

        for layer_name in self.layer_names:
            if layer_name not in self.feature_maps:
                continue

            feat_map = self.feature_maps[layer_name]  # already on GPU
            if feat_map.shape[0] < 2:
                continue

            half = feat_map.shape[0] // 2
            feat_neg = feat_map[:half].mean(dim=0)
            feat_pos = feat_map[half:].mean(dim=0)

            if aggregation_method == "cosine_dissimilarity":
                f_pos_n = F.normalize(feat_pos, p=2, dim=-1)
                f_neg_n = F.normalize(feat_neg, p=2, dim=-1)
                diff_spatial = 1.0 - (f_pos_n * f_neg_n).sum(dim=-1)
            else:
                diff = feat_pos - feat_neg
                if aggregation_method == "l2_norm":
                    diff_spatial = torch.norm(diff, dim=-1)
                elif aggregation_method == "mean_abs":
                    diff_spatial = diff.abs().mean(dim=-1)
                elif aggregation_method == "max_abs":
                    diff_spatial = diff.abs().max(dim=-1)[0]
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            seq_len = diff_spatial.shape[0]
            spatial_size = int(seq_len ** 0.5)

            if spatial_size * spatial_size == seq_len:
                # Perfect square — reshape directly on GPU
                gpu_diffs.append(diff_spatial.view(1, 1, spatial_size, spatial_size))
            else:
                # Non-square — pad and reshape on GPU, then resize with F.interpolate
                target_sz = 32
                ah = int(seq_len ** 0.5)
                aw = int((seq_len + ah - 1) // ah)  # ceil division
                pad_len = ah * aw - seq_len
                if pad_len > 0:
                    diff_spatial = torch.cat([diff_spatial, diff_spatial[-1:].expand(pad_len)])
                d2 = diff_spatial[: ah * aw].view(1, 1, ah, aw)
                d2 = F.interpolate(d2, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
                gpu_diffs.append(d2)

            del diff_spatial

        if not gpu_diffs:
            return None

        # Resize all to the largest spatial size on GPU, then average
        target_sz = max(d.shape[-1] for d in gpu_diffs)
        resized = []
        for d in gpu_diffs:
            if d.shape[-1] != target_sz:
                d = F.interpolate(d, size=(target_sz, target_sz), mode="bilinear", align_corners=False)
            resized.append(d)

        # Single GPU→CPU transfer at the very end
        combined = torch.cat(resized, dim=0).mean(dim=0).squeeze()
        return combined.cpu().numpy()


# =====================================================================
# Feature-hook helpers
# =====================================================================

def register_sd3_feature_hooks(model, feature_store: SD3FeatureMapStore):
    """Register forward hooks on selected transformer blocks to capture features."""
    hooks = []

    def _make_hook(layer_name):
        def hook(module, input, output, kwargs=None):
            num_image_tokens = None
            if kwargs and "hidden_states" in kwargs:
                num_image_tokens = kwargs["hidden_states"].shape[1]
            elif input and torch.is_tensor(input[0]):
                num_image_tokens = input[0].shape[1]

            feat = None
            if isinstance(output, tuple) and len(output) >= 2:
                feat = output[1]
            elif isinstance(output, dict) and "hidden_states" in output:
                feat = output["hidden_states"]
            elif torch.is_tensor(output):
                feat = output

            if feat is not None:
                feature_store.store_feature_map(layer_name, feat, num_image_tokens)

        return hook

    for name, module in model.named_modules():
        if name in feature_store.selected_layers:
            try:
                h = module.register_forward_hook(_make_hook(name), with_kwargs=True)
            except TypeError:
                h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)
    return hooks


def cleanup_sd3_feature_hooks(hooks):
    """Remove all registered feature hooks."""
    for h in hooks:
        h.remove()
