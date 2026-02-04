from typing import Optional
import torch
import torch.utils.checkpoint as cp
from .CoOp import CoOpModel


class ViLTModel(CoOpModel):
    def __init__(self, config, class_names, device='cuda', **kwargs):
        super().__init__(config, class_names, device, **kwargs)

    def init_context_learner(self):
        from .ContextLearners import ViLTContextLearner
        # Ensure that the number of layers is set in the config 
        self.config["N-text-layers"] = self.clip.text_model.config.num_hidden_layers
        self.config["N-vision-layers"] = self.clip.vision_model.config.num_hidden_layers
        self.context_learner = ViLTContextLearner(
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
        text_features = text_features.to(self.device) if text_features is not None else None

        # Get text context embeddings
        text_embedded, attention_mask_text = self.context_learner.get_text_context_embeds(
            self.text_embedding_layer, class_names=class_names
        )
        # Visual prompts
        visual_prompts_bv = self.context_learner.get_visual_context(B * V)            # shape: [B*V, ctx_len, dim]
        # expand to per-frame by repeat_interleave so we don't call get_visual_context B*V*T times
        visual_prompts = visual_prompts_bv.repeat_interleave(T, dim=0)               # shape: [B*V*T, ctx_len, dim]

        reshaped_pixels = videos.view(B * V * T, C, H, W)
        x = self.clip.vision_model.embeddings(reshaped_pixels)          # [B * V * T, N tokens, visual_embed_dim]
        n_tokens_t = text_embedded.size(1)
        n_tokens_v = x.size(1)
        x = torch.cat([x, visual_prompts], dim=1)
        vision_embedded = self.clip.vision_model.pre_layrnorm(x)
        self.context_learner.compute_translation_features()
        attentions = [] if return_attentions else None
        # Layer-wise checkpointed forward
        for i, (text_layer, vision_layer) in enumerate(
            zip(self.clip.text_model.encoder.layers, self.clip.vision_model.encoder.layers)
        ):
            def layer_forward(text_emb, vision_emb, t_text, v_text, t_vision, v_vision):
                # Text
                if t_text:
                    text_context = [self.context_learner.text_attention_context[idx]
                                    .unsqueeze(0).expand(self.context_learner.num_classes, -1, -1)
                                    for idx in t_text]
                    text_emb = torch.cat((text_emb, *text_context), dim=1)
                if v_text and v_vision:    # Only translatable from vision to text if vision prompts were added
                    v2t = [self.context_learner.vision_to_text_translations_pre_computed[idx].unsqueeze(0).expand(self.context_learner.num_classes, -1, -1)
                        for idx in v_text]
                    text_emb = torch.cat((text_emb, *v2t), dim=1)

                # Vision
                if v_vision:
                    vision_context = [self.context_learner.vision_attention_context[idx]
                                    .unsqueeze(0).expand(vision_emb.size(0), -1, -1)
                                    for idx in v_vision]
                    vision_emb = torch.cat((vision_emb, *vision_context), dim=1)
                if t_vision and t_text: # Only translatatble from text to vision if text prompts were added
                    t2v = [self.context_learner.text_to_vision_translations_pre_computed[idx].unsqueeze(0).expand(vision_emb.size(0), -1, -1)
                        for idx in t_vision]
                    vision_emb = torch.cat((vision_emb, *t2v), dim=1)
                # Attention mask
                extra_attn = torch.ones((text_emb.shape[0], text_emb.shape[1] - attention_mask_text.shape[1]), device=text_emb.device)
                attn_mask = torch.cat((attention_mask_text, extra_attn), dim=1)[:, None, None, :]
                text_out, = text_layer(text_emb, attention_mask=attn_mask, causal_attention_mask=None)

                if return_attentions:
                    vision_out, attn = vision_layer(vision_emb, attention_mask=None, causal_attention_mask=None, output_attentions=True)
                    attentions.append(attn[..., :n_tokens_v, :n_tokens_v])
                else:
                    vision_out, = vision_layer(vision_emb, attention_mask=None, causal_attention_mask=None)

                # Keep only the original tokens, discard prompts
                return text_out[:, :n_tokens_t, :], vision_out[:, :n_tokens_v, :]
            
            t_text=self.config["modality-transfer-text"][i][0]
            v_text=self.config["modality-transfer-text"][i][1]
            v_vision=self.config["modality-transfer-vision"][i][0]
            t_vision=self.config["modality-transfer-vision"][i][1]

            # t_text, v_text, t_vision, v_vision = (False, False, False, False)
            if not any([t_text, v_text, t_vision, v_vision]):
                text_embedded = text_layer(text_embedded, attention_mask=attention_mask_text[:, None, None, :], causal_attention_mask=None)[0]
                vision_embedded = vision_layer(vision_embedded, attention_mask=None, causal_attention_mask=None)[0]
            else:
                text_embedded, vision_embedded = cp.checkpoint(
                    layer_forward, 
                    text_embedded, 
                    vision_embedded,
                    t_text=t_text,
                    v_text=v_text,
                    t_vision=t_vision,
                    v_vision=v_vision,
                    use_reentrant=False) if self.config["use-checkpointing"] else \
                    layer_forward(
                        text_embedded, 
                        vision_embedded,
                        t_text=t_text,
                        v_text=v_text,
                        t_vision=t_vision,
                        v_vision=v_vision,
                    )
        # Apply final layer norms and projection
        text_embedded = self.clip.text_model.final_layer_norm(text_embedded)
        text_last_hidden_state = self.clip.text_projection(text_embedded)
        text_features = self._get_pooled_text_output(text_last_hidden_state, attention_mask=attention_mask_text)

        vision_embedded = self.clip.vision_model.post_layernorm(vision_embedded[:, 0, :])
        vision_features = self.clip.visual_projection(vision_embedded)
        vision_features = self._temporally_pool_with_mask(vision_features, B, V, T, video_masks)

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
