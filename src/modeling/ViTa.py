from typing import Optional
import torch
import torch.utils.checkpoint as cp
from .CoOp import CoOpModel
from .utils import check_tensor

class ViTaModel(CoOpModel):
    def __init__(self, config, class_names, device='cuda', **kwargs):
        super().__init__(config, class_names, device, **kwargs)

    def init_context_learner(self):
        from .ContextLearners import ViTaContextLearner
        # Ensure that the number of layers is set in the config 
        self.config["N-text-layers"] = self.clip.text_model.config.num_hidden_layers
        self.config["N-vision-layers"] = self.clip.vision_model.config.num_hidden_layers
        self.checkpoint_last_n = self.config.get("checkpoint-last-n", 6)
        self.context_learner = ViTaContextLearner(
            config=self.config,
            tokenizer=self.tokenizer,
            class_names=self.class_names,
            embed_dim_text=self.clip.text_model.config.hidden_size,
            embed_dim_vision=self.clip.vision_model.config.hidden_size,
            device=self.device,
        ).to(self.device)

    def forward(
            self, 
            videos: torch.Tensor, 
            video_masks: Optional[torch.BoolTensor] = None, 
            text_features: Optional[torch.Tensor] = None,
            class_names: Optional[list[str]] = None, 
            softmax: bool = False,
            return_attentions: bool = False,
            ):
        B, V, T, C, H, W = videos.shape
        B, V, T, C, H, W = int(B), int(V), int(T), int(C), int(H), int(W)
        videos = videos.to(self.device)
        video_masks = video_masks.to(self.device) if video_masks is not None else None
        # Text is simply encoded as per CoOp
        text_features = self.encode_texts(class_names=class_names).to(self.device) if text_features is None else text_features.to(self.device)

        # Get embed the entire batch of frames
        reshaped_pixels = videos.contiguous().view(B*V*T, C, H, W).to(self.device)
        videos_embedded = self.clip.vision_model.embeddings(reshaped_pixels)          # [B * V * T, N tokens, visual_embed_dim]
        videos_embedded = self.clip.vision_model.pre_layrnorm(videos_embedded)        # [B * V * T, N tokens, visual_embed_dim]
        cls_norm = None
        attentions = [] if return_attentions else None
        for i, vision_layer in enumerate(self.clip.vision_model.encoder.layers):
            print(i) if self.debug else None
            def layer_forward(frame_emb: torch.Tensor, B: int, V: int, T: int, prev_cls_norm: torch.Tensor = None):
                n_tokens, d = frame_emb.shape[1], frame_emb.shape[2]
                frame_emb = frame_emb.view(B, V, T, n_tokens, d)
                check_tensor(frame_emb, "frame_emb", layer_idx=i) if self.debug else None

                # ----- summary attention on cls -----
                cls_token = frame_emb[:, :, :, :1, :].reshape(B*V*T, 1, d)
                check_tensor(cls_token, "cls_token", layer_idx=i) if self.debug else None

                cls_proj = self.context_learner.summary_projection[i](cls_token)
                check_tensor(cls_proj, "cls_proj", layer_idx=i) if self.debug else None

                cls_norm = self.context_learner.layer_norms_pre_summary[i](cls_proj)
                check_tensor(cls_norm, "cls_norm", layer_idx=i) if self.debug else None

                summary_token, _ = self.context_learner.summary_attention_layers[i](cls_norm, cls_norm, cls_norm)
                check_tensor(summary_token, "summary_token_pre_add", layer_idx=i) if self.debug else None

                summary_token = summary_token + cls_norm
                summary_token = summary_token.view(B, V, T, 1, d)

                # ----- frame & video tokens -----
                frame_tokens = self.context_learner.frame_level_tokens.unsqueeze(0).unsqueeze(0).expand(B, V, T, -1, -1)
                # Add cls token to frame tokens as per the paper
                frame_tokens = frame_tokens + prev_cls_norm.reshape(B, V, T, 1, -1).repeat(1, 1, 1, frame_tokens.shape[3], 1) if self.config["has-discriminative-conditioning"] and prev_cls_norm is not None else frame_tokens
                check_tensor(frame_tokens, "frame_tokens", layer_idx=i) if self.debug else None
                video_tokens = self.context_learner.context_vectors_vision.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, V, T, -1, -1)
                check_tensor(video_tokens, "video_tokens", layer_idx=i) if self.debug else None
                
                catted = torch.cat([frame_emb, summary_token, frame_tokens, video_tokens], dim=3).view(B*V*T, -1, d)
                check_tensor(catted, "catted", layer_idx=i) if self.debug else None

                # ----- vision layer -----
                x = vision_layer.layer_norm1(catted)
                check_tensor(x, "x_pre_attn", layer_idx=i) if self.debug else None

                self_attended, attn = vision_layer.self_attn(x, attention_mask=None, causal_attention_mask=None, output_attentions=return_attentions)
                check_tensor(self_attended, "self_attended", layer_idx=i) if self.debug else None
                if return_attentions:
                    attentions.append(attn[..., :n_tokens, :n_tokens])  # (B*V*T, heads, N, N)
                self_attended = self_attended + catted
                vision_emb_part = self_attended[:, :n_tokens, :]
                check_tensor(vision_emb_part, "vision_emb_part", layer_idx=i) if self.debug else None

                out_norm = vision_layer.layer_norm2(vision_emb_part)
                check_tensor(out_norm, "out_norm", layer_idx=i) if self.debug else None

                vision_out = vision_layer.mlp(out_norm)
                check_tensor(vision_out, "vision_out_pre_add", layer_idx=i) if self.debug else None

                vision_out = vision_out + vision_emb_part
                check_tensor(vision_out, "vision_out_final", layer_idx=i) if self.debug else None

                return vision_out, cls_norm

            if self.use_checkpoint and i >= self.clip.vision_model.config.num_hidden_layers - self.checkpoint_last_n:
                videos_embedded, cls_norm = cp.checkpoint(
                    lambda x: layer_forward(x, B, V, T, cls_norm), videos_embedded, use_reentrant=False
                )
            else:
                videos_embedded, cls_norm = layer_forward(videos_embedded, B, V, T, cls_norm)
        # Append cls of frame after all layers
        frames_out = videos_embedded.view(B, V, T, -1, self.context_learner.embed_dim_vision)              # [B, V, T, N tokens, visual_embed_dim]
        cls_tokens = frames_out[:, :, :, 0, :].view(B*V*T, self.context_learner.embed_dim_vision)            # [B, T, visual_embed_dim]
        # Apply the layer norm and projection as per CLIP
        vision_embedded = self.clip.vision_model.post_layernorm(cls_tokens)                             # [B * V * T, visual_embed_dim], 
        vision_features = self.clip.visual_projection(vision_embedded)                                  # [B * V * T, shared_embed_dim]
        vision_features = self._temporally_pool_with_mask(vision_features, B=B, V=V, T=T, mask=video_masks)  # [B, V, shared_embed_dim]
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)  # Normalize
        logits = self._get_logits(text_features, vision_features, softmax=softmax)
        if return_attentions:
            return logits, text_features, vision_features, attentions
        return logits, text_features, vision_features
    
    def encode_video_batch(
            self, 
            pixel_values: torch.FloatTensor, 
            mask: Optional[torch.BoolTensor] = None, 
            normalize: bool = True, 
            temporally_pool: bool = True, 
            return_attentions: bool = False
            ) -> torch.FloatTensor:
        _, _, vid_feats, attentions = self.forward(
            videos=pixel_values, 
            video_masks=mask, 
            text_features=None, 
            class_names=None, 
            softmax=False,
            return_attentions=return_attentions
        )
        if normalize:
            vid_feats = vid_feats / vid_feats.norm(dim=-1, keepdim=True)
        if temporally_pool:
            B, V, T, C = pixel_values.shape[0], pixel_values.shape[1], pixel_values.shape[2], vid_feats.shape[-1]
            vid_feats = self._temporally_pool_with_mask(vid_feats, B, V, T, mask)
        if return_attentions:
            return vid_feats, attentions
        return vid_feats