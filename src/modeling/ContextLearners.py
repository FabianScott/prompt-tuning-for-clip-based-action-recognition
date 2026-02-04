# --------------------------- Prompt Learner (CoOp) ---------------------------
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPTokenizer


from typing import Optional


class TextContextLearner(nn.Module):
    """Learns a set of continuous context vectors and composes prompts for each class.

    Implementation notes:
    - We keep a shared learnable context of length K.
    - For each class, we append the tokenized class name tokens (converted to embeddings)
      after the learnable context and feed `inputs_embeds` to CLIP's text encoder.
    - If the sizes do not match when loading a checkpoint, we print a warning and skip 
      loading that parameter. This allows for adding a module to a simpler pre-trained class.
    """

    def __init__(
            self, 
            tokenizer: CLIPTokenizer, 
            embed_dim_text: int, 
            embed_dim_vision: int, 
            class_names: list[str], 
            config: dict, 
            device: str = "cpu", 
            pad_side: str = "right"
            ):
        super().__init__()
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.config = config
        self.num_classes = len(class_names)
        self.ctx_len = config["ctx-len"]
        self.std_init = config["std-init"]
        self.embed_dim_text = embed_dim_text
        self.embed_dim_vision = embed_dim_vision
        self.device = device
        self.init_text = config["regularisation-text"]
        self.class_token_ids: Optional[torch.LongTensor] = None
        self.class_attention_mask: Optional[torch.LongTensor] = None
        self.hand_crafted_attention_mask: Optional[torch.LongTensor] = None
        self.hand_crafted_token_ids: Optional[torch.LongTensor] = None
        self.hand_crafted_features: Optional[torch.Tensor] = None
        # Shared context vectors: (ctx_len, embed_dim)
        self.context_vectors_text = nn.Parameter(torch.randn(self.ctx_len, embed_dim_text) * self.std_init)
        self.temporal_pooling = self.get_temporal_pooling_method()
        # From ViTa clip:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Pre-tokenize class names (with padding) to use token ids for class suffix
        self.tokenizer.padding_side = pad_side

    def tokenize_classes(self, class_names: Optional[list[str]] = None) -> tuple[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        class_names = class_names if class_names is not None else self.class_names
        tok_out = self.tokenizer(class_names, padding=True, return_tensors="pt", add_special_tokens=True,)
        class_token_ids = tok_out["input_ids"].to(self.device)     # (num_classes, len(max_tokenization), )
        class_attention_mask = tok_out["attention_mask"].to(self.device, torch.int)   # (num_classes, len(max_tokenization), )

        return class_token_ids, class_attention_mask

    def tokenize_text(self, text: Optional[str] = None) -> tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        # Tokenize init text, not used yet
        text = text if text is not None else self.init_text
        if text is None:
            return None, None
        else:
            tok_out_hand_crafted = self.tokenizer([self.init_text + f" {classname}" for classname in self.class_names], padding=True, return_tensors="pt", add_special_tokens=True,)
            hand_crafted_token_ids = tok_out_hand_crafted["input_ids"]
            hand_crafted_attention_mask = tok_out_hand_crafted["attention_mask"]

        return hand_crafted_token_ids, hand_crafted_attention_mask

    def get_text_context_embeds(self, text_embedding_layer: nn.Embedding, class_names: Optional[None]) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Build inputs_embeds and attention_masks for each text class prompt.
        Allows for new class_names to be passed, these will be used over the saved ones.
        Returns:
            inputs_embeds: Tensor of shape (num_classes, prompt_len, embed_dim)
            attention_mask: LongTensor of shape (num_classes, prompt_len)
        """
        if class_names is not None:     # If new classes are provided, use these
            class_token_ids, class_attention_mask = self.tokenize_classes(class_names=class_names)
            class_token_ids, class_attention_mask = class_token_ids.to(self.device), class_attention_mask.to(self.device)
        elif self.class_token_ids is not None:  # Use saved classes if no new classes provided
            class_token_ids, class_attention_mask = self.class_token_ids, self.class_attention_mask
        else:   # If no classes provided and no saved ones, use initial class_names and save them
            class_token_ids, class_attention_mask = self.tokenize_classes(class_names=class_names)
            self.class_token_ids, self.class_attention_mask = class_token_ids, class_attention_mask

        num_classes = class_token_ids.shape[0]
        device = self.context_vectors_text.device

        class_embeddings = text_embedding_layer(class_token_ids)  # (num_classes, len(max_tokenisation), D)
        context_vectors_expanded = self.context_vectors_text.unsqueeze(0).expand(class_embeddings.shape[0], -1, -1)  # (num_classes, ctx_dim, D)

        # Get start token at the start. Has no impact on the attention mask
        combined_embeddings = torch.cat([class_embeddings[:, :1,], context_vectors_expanded, class_embeddings[:, 1:]], dim=1)
        combined_attention_mask = torch.cat([torch.ones((num_classes, self.ctx_len), device=device), class_attention_mask], dim=1)

        return combined_embeddings.to(self.device), combined_attention_mask.to(self.device)

    def get_visual_context(self, batch_size: int):
        """Returns visual prompts to prepend to the vision transformer patch embeddings.
        Shape: (B, ctx_len_vision, embed_dim_vision)
        """
        raise NotImplementedError("This method should be implemented in subclasses if visual prompts are used.")
    
    def get_hand_crafted_features(self, clip_model: CLIPModel):
        """Returns the embedded hand crafted prompts"""
        if self.hand_crafted_features is None:
            if self.hand_crafted_token_ids is None or self.hand_crafted_attention_mask is None:
                self.hand_crafted_token_ids, self.hand_crafted_attention_mask = self.tokenize_text()
                self.hand_crafted_token_ids = self.hand_crafted_token_ids.to(self.device)
                self.hand_crafted_attention_mask = self.hand_crafted_attention_mask.to(self.device)
            self.hand_crafted_features = clip_model.get_text_features(
                input_ids=self.hand_crafted_token_ids,
                attention_mask=self.hand_crafted_attention_mask # Ensure padding is ignored
            )
        return self.hand_crafted_features
    
    def get_temporal_pooling_method(self):
        name = self.config["temporal-pooling"]
        if name == "mean":
            from .utils import mean_pooling
            return mean_pooling
        elif name == "max":
            from .utils import max_pooling
            return max_pooling
        elif name == "attention":
            from .utils import AttentionPooling
            return AttentionPooling(embed_dim=self.embed_dim_text, num_heads=self.config["num-heads-attention-pooling"]).to(self.device)
        else:
            raise ValueError(f"Unknown temporal pooling method: {name}")

    def post_backward_computations(self):
        pass

    def clear_pre_computations(self):
        pass

    def save_context(self, filepath: str):
        """Saves the learned context vectors to a file, with attention pooling weights if present."""
        temp_pool = self.temporal_pooling if self.config["temporal-pooling"] == "attention" else None

        torch.save({
            'text_context': self.context_vectors_text.detach().cpu(),
            'temporal_pooling': temp_pool
        }, filepath)
       

    def load_context(self, filepath: str):
        """Loads learned context vectors from a file."""
        loaded_context = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(loaded_context, dict):
            if 'text_context' in loaded_context:
                loaded_text = loaded_context['text_context']
                if loaded_text.shape != self.context_vectors_text.shape:
                    print(f"Loaded context shape {loaded_text.shape} does not match expected shape {self.context_vectors_text.shape}")
                with torch.no_grad():
                    self.context_vectors_text.copy_(loaded_text)
            else:
                print("No text_context found in the loaded file.")
            if self.config["temporal-pooling"] == "attention" and 'temporal_pooling' in loaded_context:
                self.temporal_pooling = loaded_context['temporal_pooling']
        # Read old files:
        else:
            if loaded_context.shape != self.context_vectors_text.shape:
                print(f"Loaded context shape {loaded_context.shape} does not match expected shape {self.context_vectors_text.shape}")
            with torch.no_grad():
                self.context_vectors_text.copy_(loaded_context)
        return loaded_context

class DualContextLearner(TextContextLearner):
    """Extends TextPromptLearner with visual prompt learning for video frames."""

    def __init__(self, tokenizer, embed_dim_text, embed_dim_vision, class_names, config, device="cpu"):
        super().__init__(tokenizer, embed_dim_text, embed_dim_vision, class_names, config=config, device=device)
        self.ctx_len_vision = config["ctx-len-video"]
        # Visual prompt vectors (for each frame)
        self.context_vectors_vision = nn.Parameter(
            torch.randn(self.ctx_len_vision, self.embed_dim_vision) * self.std_init
        )

    def get_visual_context(self, batch_size: int):
        """
        Returns visual prompts to prepend to the vision transformer patch embeddings.
        Shape: (B, ctx_len_vision, embed_dim_vision)
        """
        return self.context_vectors_vision.unsqueeze(0).repeat(batch_size, 1, 1)

    def save_context(self, filepath):
        """Saves both text and visual context vectors to a file."""
        temp_pool = self.temporal_pooling if self.config["temporal-pooling"] == "attention" else None
        torch.save({
            'text_context': self.context_vectors_text.detach().cpu(),
            'vision_context': self.context_vectors_vision.detach().cpu(),
            'temporal_pooling': temp_pool
        }, filepath)
    
    def load_context(self, filepath):
        """Loads both text and visual context vectors from a file."""
        loaded = super().load_context(filepath)
        if 'text_context' in loaded:
            loaded_text = loaded['text_context']
            if loaded_text.shape != self.context_vectors_text.shape:
                print(f"Loaded text context shape {loaded_text.shape} does not match expected shape {self.context_vectors_text.shape}")
            with torch.no_grad():
                self.context_vectors_text.copy_(loaded_text)
        else:
            print("No text_context found in the loaded file.")
        if 'vision_context' in loaded:
            loaded_vision = loaded['vision_context']
            if loaded_vision.shape != self.context_vectors_vision.shape:
                print(f"Loaded vision context shape {loaded_vision.shape} does not match expected shape {self.context_vectors_vision.shape}")
            with torch.no_grad():
                self.context_vectors_vision.copy_(loaded_vision)
        else:
            print("No vision_context found in the loaded file.")
        if self.config["temporal-pooling"] == "attention" and 'temporal_pooling' in loaded:
            self.temporal_pooling = loaded['temporal_pooling']
        return loaded


class ViLTContextLearner(DualContextLearner):
    def __init__(self, tokenizer, embed_dim_text, embed_dim_vision, class_names, config, device="cpu"):
        super().__init__(tokenizer, embed_dim_text, embed_dim_vision=embed_dim_vision, class_names=class_names, config=config, device=device)
        self.init_ViLT_layers()

    def init_ViLT_layers(self):
        """
        Initilise the additional parameters needed for ViLT
        - Attention contexts for each layer and modality
        - Translation layers between modalities for each layer
        Implementation notes:
        - Attention contexts are learnable parameters of shape (N_layers, num_per_layer, embed_dim)
        - Translation layers are shallow linear layers mapping from one embed_dim to the other
        - Despite come of the connections not being used in the current config, we still create them all for easier loading/saving
        """
        N_text_layers = self.config["N-text-layers"]
        N_vision_layers = self.config["N-vision-layers"]
        if N_text_layers != N_vision_layers:
            print(f"Warning: Different number of text and vision layers ({N_text_layers} vs {N_vision_layers})")
        num_tokens_per_layer_text = self.config["num-tokens-per-layer-text"]
        num_tokens_per_layer_vision = self.config["num-tokens-per-layer-vision"]

        self.text_attention_context = nn.ParameterList([None for _ in range(N_text_layers)])
        self.vision_attention_context = nn.ParameterList([None for _ in range(N_vision_layers)])
        self.text_to_vision_translations = nn.ModuleList([None for _ in range(N_text_layers)])
        self.vision_to_text_translations = nn.ModuleList([None for _ in range(N_vision_layers)])

        for i in range(N_vision_layers):
            t_text=self.config["modality-transfer-text"][i][0]
            v_text=self.config["modality-transfer-text"][i][1]
            v_vision=self.config["modality-transfer-vision"][i][0]
            t_vision=self.config["modality-transfer-vision"][i][1]

            if t_text:
                self.text_attention_context[i] = nn.Parameter(torch.randn(num_tokens_per_layer_text, self.embed_dim_text, device=self.device))
            if v_text:
                self.vision_to_text_translations[i] = nn.Linear(self.embed_dim_vision, self.embed_dim_text).to(self.device)
            if v_vision:
                self.vision_attention_context[i] = nn.Parameter(torch.randn(num_tokens_per_layer_vision, self.embed_dim_vision, device=self.device))
            if t_vision:
                self.text_to_vision_translations[i] = nn.Linear(self.embed_dim_text, self.embed_dim_vision).to(self.device)

        self.text_to_vision_translations_pre_computed = [None for _ in range(N_text_layers)]
        self.vision_to_text_translations_pre_computed = [None for _ in range(N_vision_layers)]

    def post_backward_computations(self):
        super().post_backward_computations()
        self.compute_translation_features()

    def clear_pre_computations(self):
        super().clear_pre_computations()
        self.clear_precomputed_flags()

    def compute_translation_features(self):
        self.text_to_vision_translations_pre_computed = [
            self.text_to_vision_translations[i](self.text_attention_context[i])
            if (
                self.text_to_vision_translations[i] is not None
                and self.text_attention_context[i] is not None
            )
            else self.text_to_vision_translations_pre_computed[i]
            for i in range(len(self.text_to_vision_translations))
        ]

        self.vision_to_text_translations_pre_computed = [
            self.vision_to_text_translations[i](self.vision_attention_context[i])
            if (
                self.vision_to_text_translations[i] is not None
                and self.vision_attention_context[i] is not None
            )
            else self.vision_to_text_translations_pre_computed[i]
            for i in range(len(self.vision_to_text_translations))
        ]

    def clear_precomputed_flags(self):
        self.text_to_vision_translations_pre_computed = [None for _ in range(len(self.text_to_vision_translations))]
        self.vision_to_text_translations_pre_computed = [None for _ in range(len(self.vision_to_text_translations))]

    def save_context(self, filepath):
        temp_pool = self.temporal_pooling if self.config["temporal-pooling"] == "attention" else None
        torch.save({
            'text_context': self.context_vectors_text.detach().cpu(),
            'vision_context': self.context_vectors_vision.detach().cpu(),
            'text_attention_context': self.text_attention_context.detach().cpu(),
            'vision_attention_context': self.vision_attention_context.detach().cpu(),
            'text_to_vision_translations': [layer.state_dict() for layer in self.text_to_vision_translations],
            'vision_to_text_translations': [layer.state_dict() for layer in self.vision_to_text_translations],
            'temporal_pooling': temp_pool
        }, filepath)

    def load_context(self, filepath):
        loaded = super().load_context(filepath)

        loaded_text_attn = loaded['text_attention_context']
        if loaded_text_attn.shape != self.text_attention_context.shape:
            print(f"Loaded text attention context shape {loaded_text_attn.shape} does not match expected shape {self.text_attention_context.shape}")
        with torch.no_grad():
            self.text_attention_context.copy_(loaded_text_attn)
        
        loaded_vision_attn = loaded['vision_attention_context']
        if loaded_vision_attn.shape != self.vision_attention_context.shape:
            print(f"Loaded vision attention context shape {loaded_vision_attn.shape} does not match expected shape {self.vision_attention_context.shape}")
        with torch.no_grad():
            self.vision_attention_context.copy_(loaded_vision_attn)

        loaded_text_to_vision = loaded['text_to_vision_translations']
        if len(loaded_text_to_vision) != len(self.text_to_vision_translations):
            print(f"Loaded text to vision translations length {len(loaded_text_to_vision)} does not match expected length {len(self.text_to_vision_translations)}")
        for layer, state_dict in zip(self.text_to_vision_translations, loaded_text_to_vision):
            layer.load_state_dict(state_dict)

        loaded_vision_to_text = loaded['vision_to_text_translations']
        if len(loaded_vision_to_text) != len(self.vision_to_text_translations):
            print(f"Loaded vision to text translations length {len(loaded_vision_to_text)} does not match expected length {len(self.vision_to_text_translations)}")
        for layer, state_dict in zip(self.vision_to_text_translations, loaded_vision_to_text):
            layer.load_state_dict(state_dict)
        if self.config["temporal-pooling"] == "attention" and 'temporal_pooling' in loaded:
            self.temporal_pooling = loaded['temporal_pooling']
        return loaded

class ViTaContextLearner(DualContextLearner):
    def __init__(self, tokenizer, embed_dim_text, embed_dim_vision, class_names, config, device="cpu"):
        super().__init__(tokenizer, embed_dim_text, embed_dim_vision=embed_dim_vision, class_names=class_names, config=config, device=device)
        self.init_ViTa_layers()

    def init_ViTa_layers(self):
        N_text_layers = self.config["N-text-layers"]
        N_vision_layers = self.config["N-vision-layers"]
        
        # One projection per layer to project the CLS token to the summary token space
        self.summary_projection = nn.ModuleList([
            nn.Linear(self.embed_dim_vision, self.embed_dim_vision)
            for _ in range(N_vision_layers)
        ])
        # Layer norms before the summary attention
        self.layer_norms_pre_summary = nn.ModuleList([
            nn.LayerNorm(self.embed_dim_vision) for _ in range(N_vision_layers)
        ])
        # Create a separate attention layer for layer wise summary
        self.summary_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim_vision,
                num_heads=self.config["num-heads-summary-attention"],
                device=self.device,
                batch_first=True    # (B, Seq, D) instead of (Seq, B, D)
            ) for _ in range(N_vision_layers)
        ])
        # Tokens appended for each frame token and inserted at each layer
        self.frame_level_tokens = nn.Parameter(torch.randn(self.config["num-frames"], self.config["num-frame-tokens"], self.embed_dim_vision) * self.std_init)

    def save_context(self, filepath):
        temp_pool = self.temporal_pooling if self.config["temporal-pooling"] == "attention" else None
        torch.save({
            'text_context': self.context_vectors_text.detach().cpu(),
            'vision_context': self.context_vectors_vision.detach().cpu(),
            'summary_projection': [layer.state_dict() for layer in self.summary_projection],
            'layer_norms_pre_summary': [layer.state_dict() for layer in self.layer_norms_pre_summary],
            'summary_attention_layers': [layer.state_dict() for layer in self.summary_attention_layers],
            'clip_level_tokens': self.frame_level_tokens.detach().cpu(),
            'temporal_pooling': temp_pool
        }, filepath)

    def load_context(self, filepath):
        loaded = super().load_context(filepath)

        if 'clip_level_tokens' in loaded:
            loaded_clip_tokens = loaded['clip_level_tokens']
            if loaded_clip_tokens.shape != self.frame_level_tokens.shape:
                print(f"Loaded clip level tokens shape {loaded_clip_tokens.shape} does not match expected shape {self.frame_level_tokens.shape}")
            with torch.no_grad():
                self.frame_level_tokens.copy_(loaded_clip_tokens)
        else:
            print("No clip_level_tokens found in the loaded file.")

        loaded_summary_proj = loaded['summary_projection']
        if len(loaded_summary_proj) != len(self.summary_projection):
            print(f"Loaded summary projection length {len(loaded_summary_proj)} does not match expected length {len(self.summary_projection)}")
        for layer, state_dict in zip(self.summary_projection, loaded_summary_proj):
            layer.load_state_dict(state_dict)

        loaded_layer_norms = loaded['layer_norms_pre_summary']
        if len(loaded_layer_norms) != len(self.layer_norms_pre_summary):
            print(f"Loaded layer norms length {len(loaded_layer_norms)} does not match expected length {len(self.layer_norms_pre_summary)}")
        for layer, state_dict in zip(self.layer_norms_pre_summary, loaded_layer_norms):
            layer.load_state_dict(state_dict)

        loaded_summary_attn = loaded['summary_attention_layers']
        if len(loaded_summary_attn) != len(self.summary_attention_layers):
            print(f"Loaded summary attention layers length {len(loaded_summary_attn)} does not match expected length {len(self.summary_attention_layers)}")
        for layer, state_dict in zip(self.summary_attention_layers, loaded_summary_attn):
            layer.load_state_dict(state_dict)
        if self.config["temporal-pooling"] == "attention" and 'temporal_pooling' in loaded:
            self.temporal_pooling.load_state_dict(loaded['temporal_pooling'])
        
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print("NaN in", name)
            print(name, param.dtype)

        return loaded


class STTContextLearner(ViTaContextLearner):
    def __init__(self, tokenizer, embed_dim_text, embed_dim_vision, class_names, config, device="cpu"):
        super().__init__(tokenizer, embed_dim_text, embed_dim_vision=embed_dim_vision, class_names=class_names, config=config, device=device)
        self.init_STT_layers()

    def init_STT_layers(self):
        num_frames = self.config["num-frames"]
        # One projection per layer to project the summary to the text token space
        self.summary_to_text_projections = nn.ModuleList([
            nn.Linear(self.embed_dim_vision * num_frames, self.embed_dim_text) if i in self.config["layers-with-summary-to-text"] else None
            for i in range(self.config["N-vision-layers"])
        ])

class ActionCLIPContextLearner(DualContextLearner):
    def __init__(self, tokenizer, embed_dim_text, embed_dim_vision, class_names, config, device="cpu"):
        super().__init__(tokenizer, embed_dim_text, embed_dim_vision=embed_dim_vision, class_names=class_names, config=config, device=device)
        self.init_ActionCLIP_layers()

    def init_ActionCLIP_layers(self):
        self.frame_projection = nn.Linear(self.embed_dim_vision, self.embed_dim_vision)

    def save_context(self, filepath):
        temp_pool = self.temporal_pooling if self.config["temporal-pooling"] == "attention" else None
        torch.save({
            'text_context': self.context_vectors_text.detach().cpu(),
            'vision_context': self.context_vectors_vision.detach().cpu(),
            'frame_projection': self.frame_projection.state_dict(),
            'temporal_pooling': temp_pool
        }, filepath)

    def load_context(self, filepath):
        loaded = super().load_context(filepath)

        if 'frame_projection' in loaded:
            self.frame_projection.load_state_dict(loaded['frame_projection'])
        else:
            print("No frame_projection found in the loaded file.")
        return loaded