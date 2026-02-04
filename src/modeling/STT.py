from typing import Optional
import torch
import torch.utils.checkpoint as cp
from .CoOp import CoOpModel
from .utils import check_tensor

class STTModel(CoOpModel):
    def __init__(self, config, class_names, device='cuda', **kwargs):
        super().__init__(config, class_names, device, **kwargs)

    def init_context_learner(self):
        from .ContextLearners import STTContextLearner
        # Ensure that the number of layers is set in the config 
        self.config["N-text-layers"] = self.clip.text_model.config.num_hidden_layers
        self.config["N-vision-layers"] = self.clip.vision_model.config.num_hidden_layers
        self.context_learner = STTContextLearner(
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
        B, V, T, C, H, W = int(B), int(V), int(T), int(C), int(H), int(W)        # ??

        videos = videos.to(self.device)
        video_masks = video_masks.to(self.device) if video_masks is not None else None

        # Get text context embeddings
        text_embedded, attention_mask_text = self.context_learner.get_text_context_embeds(
            self.text_embedding_layer, class_names=class_names
        )
        # Add batch and view dimension to text vectors
        text_embedded = text_embedded.unsqueeze(0).unsqueeze(0).repeat_interleave(B, dim=0).repeat_interleave(V, dim=1)  # [B, V, N_classes, N_tokens, dim]
        attention_mask_text = attention_mask_text.repeat_interleave(B, dim=0).repeat_interleave(V, dim=0)  # [B, V, N_classes, N_tokens]

        reshaped_pixels = videos.view(B * V * T, C, H, W)
        x = self.clip.vision_model.embeddings(reshaped_pixels)          # [B * V * T, N tokens, visual_embed_dim]
        vision_embedded = self.clip.vision_model.pre_layrnorm(x)
        n_classes = text_embedded.size(2)
        n_tokens_t = text_embedded.size(3)
        n_tokens_v = vision_embedded.size(1)
        cls_norm = None
        attentions = [] if return_attentions else None

        # Layer-wise checkpointed forward
        for i, (text_layer, vision_layer) in enumerate(
            zip(self.clip.text_model.encoder.layers, self.clip.vision_model.encoder.layers)
        ):
            def layer_forward_no_transfer(text_emb, frame_emb, prev_cls_norm: torch.Tensor):
                n_tokens, d = frame_emb.shape[1], frame_emb.shape[2]
                frame_emb = frame_emb.view(B, V, T, n_tokens, d)
                check_tensor(frame_emb, "frame_emb", layer_idx=i) if self.debug else None

                # ----- summary attention on cls -----
                cls_token = frame_emb[:, :, :, :1, :].contiguous().reshape(B*V*T, 1, d)
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
                frame_tokens = frame_tokens + prev_cls_norm.reshape(B, V, T, 1, -1).repeat(1, 1, 1, frame_tokens.shape[3], 1) if prev_cls_norm is not None else frame_tokens
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

                # # ----- text layer -----
                text_emb = text_emb.view(B * V * text_emb.shape[2], -1, text_emb.shape[-1])  # [B*V*N_classes, N_tokens, dim]
                check_tensor(text_emb, "text_emb_no_summary", layer_idx=i) if self.debug else None
                attn_mask = attention_mask_text[:, None, None, :]

                text_out, = text_layer(text_emb, attention_mask=attn_mask, causal_attention_mask=None)
                check_tensor(text_out, "text_out", layer_idx=i) if self.debug else None
                # Keep only the original tokens, discard prompts
                return text_out[:, :n_tokens_t, :].reshape(B, V, n_classes, n_tokens_t, text_out.shape[-1]), vision_out[:, :n_tokens_v, :], cls_norm
            
            def layer_forward_with_transfer(text_emb, frame_emb, i=i, prev_cls_norm=None):
                n_tokens, d = frame_emb.shape[1], frame_emb.shape[2]
                frame_emb = frame_emb.view(B, V, T, n_tokens, d)
                check_tensor(frame_emb, "frame_emb", layer_idx=i) if self.debug else None

                # ----- summary attention on cls -----
                cls_token = frame_emb[:, :, :, :1, :].contiguous().reshape(B*V*T, 1, d)
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
                frame_tokens = frame_tokens + prev_cls_norm.reshape(B, V, T, 1, -1).repeat(1, 1, 1, frame_tokens.shape[3], 1) if prev_cls_norm is not None else frame_tokens
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

                # # ----- text layer -----
                summary_token_text = self.context_learner.summary_to_text_projections[i](summary_token.contiguous().reshape(B, V, T * d))            # shape: [B, V, dim]
                check_tensor(summary_token_text, "summary_token_text", layer_idx=i) if self.debug else None
                reshaped_summary_token_text = summary_token_text.contiguous().unsqueeze(2).repeat_interleave(int(text_emb.shape[2]), dim=2).unsqueeze(3)  # shape: [B, V, N_classes, 1, dim]
                check_tensor(reshaped_summary_token_text, "reshaped_summary_token_text", layer_idx=i) if self.debug else None
                text_emb = torch.cat([text_emb, reshaped_summary_token_text], dim=3).contiguous().view(B * V * int(text_emb.shape[2]), -1, int(text_emb.shape[-1])) # [B*V*N_classes, N_tokens + 1, dim]
                check_tensor(text_emb, "text_emb_with_summary", layer_idx=i) if self.debug else None
                # Attention mask
                extra_attn = torch.ones((text_emb.shape[0], text_emb.shape[1] - attention_mask_text.shape[1]),
                                        device=text_emb.device)
                attn_mask = torch.cat((attention_mask_text, extra_attn), dim=1)[:, None, None, :]

                text_out, = text_layer(text_emb, attention_mask=attn_mask, causal_attention_mask=None)
                check_tensor(text_out, "text_out", layer_idx=i) if self.debug else None
                # Keep only the original tokens, discard prompts
                return text_out[:, :n_tokens_t, :].reshape(B, V, n_classes, n_tokens_t, int(text_out.shape[-1])), vision_out[:, :n_tokens_v, :], cls_norm

            if self.context_learner.summary_attention_layers[i] is None:
                text_embedded = text_embedded.clone()
                vision_embedded = vision_embedded.clone()

                text_embedded = text_layer(text_embedded, attention_mask=attention_mask_text[:, None, None, :], causal_attention_mask=None)[0]
                vision_embedded = vision_layer(vision_embedded, attention_mask=None, causal_attention_mask=None)[0]
            else:
                if self.context_learner.summary_to_text_projections[i] is None:
                    text_embedded, vision_embedded, cls_norm = cp.checkpoint(
                        layer_forward_no_transfer, 
                        text_embedded, 
                        vision_embedded,
                        prev_cls_norm=cls_norm,
                        use_reentrant=False) if self.config["use-checkpointing"] else \
                        layer_forward_no_transfer(
                            text_embedded, 
                            vision_embedded,
                            prev_cls_norm=cls_norm
                        )
                else:
                    text_embedded, vision_embedded, cls_norm = cp.checkpoint(
                        layer_forward_with_transfer, 
                        text_embedded, 
                        vision_embedded,
                        prev_cls_norm=cls_norm,
                        use_reentrant=False) if self.config["use-checkpointing"] else \
                        layer_forward_with_transfer(
                            text_embedded, 
                            vision_embedded,
                            i=i,
                            prev_cls_norm=cls_norm
                        )

        # Apply final layer norms and projection
        text_embedded = self.clip.text_model.final_layer_norm(text_embedded)
        text_last_hidden_state = self.clip.text_projection(text_embedded)
        text_features = self._get_pooled_text_output(text_last_hidden_state, attention_mask=attention_mask_text)

        frames_out = vision_embedded.view(B, V, T, -1, self.context_learner.embed_dim_vision)              # [B, V, T, N tokens, visual_embed_dim]
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
