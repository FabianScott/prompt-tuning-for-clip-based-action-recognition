import torch
from .CoOp import CoOpModel
from .ContextLearners import DualContextLearner


class DualModel(CoOpModel):
    def __init__(self, config, class_names, device='cuda', **kwargs):
        super().__init__(config, class_names, device, **kwargs)

    def init_context_learner(self):
        self.context_learner = DualContextLearner(
            tokenizer=self.tokenizer,
            class_names=self.class_names,
            embed_dim_text=self.clip.text_model.config.hidden_size,
            embed_dim_vision=self.clip.vision_model.config.hidden_size,
            config=self.config,
            device=self.device,
        ).to(self.device)

    def encode_video_batch(self, pixel_values, mask=None, normalize=False, temporally_pool=True, return_attentions=False):
        # pixel_values: [B, V, T, C, H, W]
        B, V, T, C, H, W = pixel_values.shape
        B, V, T, C, H, W = int(B), int(V), int(T), int(C), int(H), int(W)   # For flop code compatibility

        # reshape frames -> (B*V*T, C, H, W) without forcing a copy when possible
        reshaped_pixels = pixel_values.view(-1, C, H, W)   # equivalent to contiguous().view but avoids copy if contiguous

        # --- get visual prompts ONCE per video (B * V) and cheaply expand to per-frame ---
        # get_visual_context returns (batch_size, ctx_len, dim)
        visual_prompts_bv = self.context_learner.get_visual_context(B * V)            # shape: [B*V, ctx_len, dim]
        # expand to per-frame by repeat_interleave so we don't call get_visual_context B*V*T times
        visual_prompts = visual_prompts_bv.repeat_interleave(T, dim=0)               # shape: [B*V*T, ctx_len, dim]

        # --- CLIP forward (per-frame) ---
        x = self.clip.vision_model.embeddings(reshaped_pixels)                       # [B*V*T, N_tokens, embed_dim]
        x = torch.cat([x, visual_prompts], dim=1)                                    # [B*V*T, N_prompts+N_tokens, embed_dim]
        n_tokens_to_keep = x.size(1) if self.config["keep-vision-prompts-throughout"] else x.size(1) - visual_prompts.size(1)
        # apply pre-layernorm once on the concatenated sequence (same as before)
        x = self.clip.vision_model.pre_layrnorm(x)

        attentions = [] if return_attentions else None
        for vision_layer in self.clip.vision_model.encoder.layers:
            if return_attentions:
                x, attn = vision_layer(
                    x,
                    attention_mask=None,
                    causal_attention_mask=None,
                    output_attentions=True,
                )
                attentions.append(attn[..., :n_tokens_to_keep, :n_tokens_to_keep])  # (B*V*T, heads, N, N)
            else:
                x = vision_layer(
                    x,
                    attention_mask=None,
                    causal_attention_mask=None,
                )[0]

            x = x[:, :n_tokens_to_keep, :]
        # run encoder (still per-frame sequences, unchanged numerically)
        # vision_outputs = self.clip.vision_model.encoder(x).last_hidden_state   # [B*V*T, seq_len, embed_dim]

        # take CLS/last state token (same as original)
        vid_feats = x[:, 0, :]                                           # [B*V*T, embed_dim]
        vid_feats = self.clip.vision_model.post_layernorm(vid_feats)                  # [B*V*T, shared_embed_dim]
        vid_feats = self.clip.visual_projection(vid_feats)                            # [B*V*T, shared_embed_dim]

        if normalize:
            vid_feats = vid_feats / vid_feats.norm(dim=-1, keepdim=True)

        if temporally_pool:
            vid_feats = self._temporally_pool_with_mask(vid_feats, B, V, T, mask)

        if return_attentions:
            return vid_feats, attentions

        return vid_feats
