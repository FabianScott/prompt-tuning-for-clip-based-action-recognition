import os
from typing import Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from .CoOp import CoOpModel
from ..plots import plot_numpy_video
import matplotlib.pyplot as plt

class VideoExplainer:
    def __init__(self, model: CoOpModel, target_layer: torch.nn.Module):
        self.model = model
        self.layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd(m, i, o): self.activations = o
        def bwd(m, grad_in, grad_out): self.gradients = grad_out[0]
        self.hooks.append(self.layer.register_forward_hook(fwd))
        self.hooks.append(self.layer.register_full_backward_hook(bwd))

    def clear_hooks(self):
        for h in self.hooks: h.remove()
        self.activations = None
        self.gradients = None

    def explain(self, videos, text_features: Optional[torch.Tensor] = None, class_idx=None, method="gradcam", fps=1, log_to_wandb=True, plot_path: str = ""):
        """
        videos: (B, V, T, C, H, W)
        text_features: (n_classes, D)
        method: "gradcam" or "attention_rollout"
        """
        B, V, T, C, H, W = videos.shape
        all_exist = True
        if plot_path:
            for b in range(B):
                for v in range(V):
                    if os.path.exists(plot_path.replace("tmp_view", f"video{b}_view{v}")):
                        print(f"Plot path {plot_path.replace('tmp_view', f'video{b}_view{v}')} exists, skipping plot.")
                        continue
                    else:
                        all_exist = False
        if all_exist and plot_path:
            print("All plot paths exist, skipping explanation.")
            return

        if method == "gradcam":
            cam, importance = self._gradcam(videos, text_features, class_idx)
        elif method == "attention-rollout":
            cam, importance = self._attention_rollout(videos, rollout_type="full")
        elif method == "attention-rollout-cls":
            cam, importance = self._attention_rollout(videos, rollout_type="cls")
        elif method == "attention-rollout-cls-weighted":
            cam, importance = self._attention_rollout(videos, rollout_type="cls_weighted")
        else:
            raise ValueError(f"Unknown method {method}")
        if plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            cam_videos = self._overlay_heatmaps(videos, cam)
            B, V, T, C, H, W = cam_videos.shape
            for b in range(B):
                for v in range(V):
                    if os.path.exists(plot_path.replace("tmp_view", f"video{b}_view{v}")):
                        print(f"Plot path {plot_path.replace('tmp_view', f'video{b}_view{v}')} exists, skipping plot.")
                        continue
                    plot_numpy_video(
                        videos=cam_videos[b,v],  # (T,C,H,W)
                        frame_names=[f"Frame {i}" for i in range(T)],
                        nrow=4,
                        save_path=plot_path.replace("tmp_view", f"video{b}_view{v}"),
                        title=f"Explainer CAM Videos temporal view {v}",
                    )
            plt.close('all')
        if log_to_wandb:
            cam_videos = self._overlay_heatmaps(videos, cam)
            self._log_wandb(cam_videos, cam, importance, fps)


    # ---------------- Internal Methods ----------------
    def _gradcam(self, videos, text_features, class_idx=None):
        for p in self.model.clip.parameters():
            p.requires_grad_(True)
        videos = videos.to(self.model.device)
        B, V, T, C, H, W = videos.shape

        # 1) forward without temporal pooling
        reshaped_pixels = videos.view(B*V*T, C, H, W)
        vision_out = self.model.clip.vision_model(
            pixel_values=reshaped_pixels,
            output_hidden_states=True
        )

        hidden = vision_out.last_hidden_state  # (B*T, N, D)
        hidden = self.model.clip.vision_model.post_layernorm(hidden)

        vid_feats = self.model.clip.visual_projection(hidden[:, 0])        
        vid_feats = vid_feats.view(B, V*T, -1)  # (B, V*T, D)
        logits = self.model._get_logits(text_features, vid_feats, view_pooling=False)

        if class_idx is None:
            class_idx = logits.argmax(dim=-1).mean().long().item()

        score = logits[..., class_idx].mean() * 1000  # scale to have larger gradients
        self.model.clip.zero_grad()
        score.backward(retain_graph=True)

        # compute CAM
        A, G = self.activations, self.gradients
        weights = G.abs().mean(dim=-1)
        cam = (weights[:,1:].unsqueeze(-1) * A[:,1:]).sum(dim=-1)
        cam = self._process_cam(cam, B, V, T)
        importance = cam.mean(dim=(3,4)).detach()  # (B,V,T)
        return cam, importance

    def _attention_rollout(self, videos, text_features=None, discard_ratio=0.0, head_fusion="mean", rollout_type="full"):
        """
        Compute attention rollout maps for video frames.
        
        Args:
            videos: (B, V, T, C, H, W)
            text_features: ignored, kept for compatibility
            discard_ratio: fraction of lowest attention to discard per layer
            head_fusion: "mean" or "max" over attention heads
            rollout_type: "full" (full token-to-token with residuals),
                         "cls" (CLS-to-patch only, log-space accumulation),
                         "cls_weighted" (full rollout weighted by CLS attention)
        Returns:
            cam: (B, V, T, H_patch, W_patch)
            importance: (B, V, T)
        """
        videos = videos.to(self.model.device)
        for p in self.model.clip.parameters():
            p.requires_grad_(True)
        B, V, T, C, H, W = videos.shape
        vid_feats, attns = self.model.encode_video_batch(
            pixel_values=videos, 
            return_attentions=True,
            temporally_pool=False
            )
            
        # Collect attentions from all layers
        # attns = [a.detach() for a in out.attentions]  # each (B*T, n_heads, N, N)
        n_layers = len(attns)
        
        if rollout_type == "full":
            cam = self._rollout_full(attns, discard_ratio, head_fusion, B, V, T)
        elif rollout_type == "cls":
            cam = self._rollout_cls(attns, discard_ratio, head_fusion, B, V, T)
        elif rollout_type == "cls_weighted":
            cam = self._rollout_cls_weighted(attns, discard_ratio, head_fusion, B, V, T)
        else:
            raise ValueError(f"Unknown rollout_type: {rollout_type}")
        
        importance = cam.mean(dim=(3,4))  # (B,V,T)
        return cam, importance

    def _rollout_full(self, attns, discard_ratio, head_fusion, B, V, T):
        """Full token-to-token attention rollout with residual connections."""
        rollout = None
        for layer_attn in tqdm(attns, desc="Computing Full Attention Rollout"):
            # fuse heads
            if head_fusion == "mean":
                attn_heads = layer_attn.mean(dim=1)  # (B*T, N, N)
            elif head_fusion == "max":
                attn_heads, _ = layer_attn.max(dim=1)
            else:
                raise ValueError("head_fusion must be 'mean' or 'max'")
            
            # optionally discard low attentions
            if discard_ratio > 0:
                flat = attn_heads.flatten(1)
                threshold = flat.topk(int(flat.size(1)*(1-discard_ratio)), dim=1)[0][:,-1].unsqueeze(-1).unsqueeze(-1)
                attn_heads = torch.where(attn_heads >= threshold, attn_heads, torch.zeros_like(attn_heads))
            
            # add identity (residual connection)
            attn_heads = attn_heads + torch.eye(attn_heads.size(-1), device=attn_heads.device)
            attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
            
            # rollout via matrix multiplication
            if rollout is None:
                rollout = attn_heads
            else:
                rollout = rollout @ attn_heads
        
        # Extract CLS token attention to patches
        cam = rollout[:, 0, 1:]  # (B*V*T, N-1)
        cam = self._process_cam(cam, B, V, T)
        return cam

    def _rollout_cls(self, attns, discard_ratio, head_fusion, B, V, T):
        """Direct CLS-to-patch attention tracking with log-space accumulation."""
        cls_patch_attention = None
        for layer_attn in tqdm(attns, desc="Computing CLS Attention Rollout"):
            # fuse heads
            if head_fusion == "mean":
                attn_heads = layer_attn.mean(dim=1)  # (B*T, N, N)
            elif head_fusion == "max":
                attn_heads, _ = layer_attn.max(dim=1)
            else:
                raise ValueError("head_fusion must be 'mean' or 'max'")
            
            # extract attention from CLS token to all patches
            cls_to_patch = attn_heads[:, 0, 1:]  # (B*T, N-1)
            
            # optionally discard low attentions
            if discard_ratio > 0:
                threshold = cls_to_patch.topk(int(cls_to_patch.size(1)*(1-discard_ratio)), dim=1)[0][:,-1].unsqueeze(-1)
                cls_to_patch = torch.where(cls_to_patch >= threshold, cls_to_patch, torch.zeros_like(cls_to_patch))
            
            # accumulate in log domain for numerical stability
            if cls_patch_attention is None:
                cls_patch_attention = torch.log(cls_to_patch + 1e-10)
            else:
                cls_patch_attention = cls_patch_attention + torch.log(cls_to_patch + 1e-10)
        
        # convert back from log domain
        cam = torch.exp(cls_patch_attention)  # (B*V*T, N-1)
        cam = self._process_cam(cam, B, V, T)
        return cam

    def _rollout_cls_weighted(self, attns, discard_ratio, head_fusion, B, V, T):
        """Full rollout weighted by layer-wise CLS-to-patch attention strength."""
        rollout = None
        cls_weights = []  # track how much CLS attends to patches per layer
        
        for layer_attn in tqdm(attns, desc="Computing CLS-Weighted Rollout"):
            # fuse heads
            if head_fusion == "mean":
                attn_heads = layer_attn.mean(dim=1)  # (B*T, N, N)
            elif head_fusion == "max":
                attn_heads, _ = layer_attn.max(dim=1)
            else:
                raise ValueError("head_fusion must be 'mean' or 'max'")
            
            # optionally discard low attentions
            if discard_ratio > 0:
                flat = attn_heads.flatten(1)
                threshold = flat.topk(int(flat.size(1)*(1-discard_ratio)), dim=1)[0][:,-1].unsqueeze(-1).unsqueeze(-1)
                attn_heads = torch.where(attn_heads >= threshold, attn_heads, torch.zeros_like(attn_heads))
            
            # track CLS attention to patches (average strength per sample)
            cls_to_patch = attn_heads[:, 0, 1:]  # (B*T, N-1)
            cls_weight = cls_to_patch.mean(dim=1, keepdim=True)  # (B*T, 1) - average attention per sample
            cls_weights.append(cls_weight)
            
            # add identity (residual connection)
            attn_heads = attn_heads + torch.eye(attn_heads.size(-1), device=attn_heads.device)
            attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
            
            # rollout via matrix multiplication
            if rollout is None:
                rollout = attn_heads
            else:
                rollout = rollout @ attn_heads
        
        # Extract CLS token attention to patches and weight by CLS focus
        cam = rollout[:, 0, 1:]  # (B*V*T, N-1)
        
        # Apply cumulative CLS weighting
        cumulative_cls_weight = torch.ones_like(cam[:, :1])  # (B*V*T, 1)
        for cls_weight in cls_weights:
            cumulative_cls_weight = cumulative_cls_weight * cls_weight
        
        cam = cam * cumulative_cls_weight  # weight the rollout by CLS attention strength
        cam = self._process_cam(cam, B, V, T)
        return cam


    def _convert_to_heatmap(self, heat: torch.Tensor) -> torch.Tensor:
        import matplotlib.pyplot as plt
        heat = F.interpolate(heat.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        heat_np = heat.squeeze(0).detach().cpu().numpy()  # (H,W)
        heat_color = plt.cm.jet(heat_np)[:,:,:3]  # drop alpha, (H,W,3)
        heat_color = torch.tensor(heat_color).permute(2,0,1)  # (3,H,W)
        return heat_color

    def _process_cam(self, cam, B, V, T):
        cam = cam / cam.max(dim=1, keepdim=True)[0]
        side = int(cam.shape[1]**0.5)
        cam = cam.view(B, V, T, side, side)
        return cam

    def _overlay_heatmaps(self, videos: torch.Tensor, cam: torch.Tensor, alpha=0.5) -> torch.Tensor:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        B, V, T, C, H, W = videos.shape

        cam_videos = []
        for b in range(B):
            views = []
            for v in range(V):
                frames = []
                for t in range(T):
                    im = videos[b, v, t].cpu() * std + mean
                    heat = cam[b, v, t].cpu().unsqueeze(0)
                    heat = self._convert_to_heatmap(heat)
                    overlay = (im * alpha + heat * (1 - alpha)).clip(0, 1) * 255
                    frames.append(overlay)
                views.append(torch.stack(frames))      # (T, C, H, W)
            cam_videos.append(torch.stack(views))     # (V, T, C, H, W)

        cam_videos = torch.stack(cam_videos)          # (B, V, T, C, H, W)
        return cam_videos.detach().cpu().numpy()


    def _log_wandb(self, cam_videos, cam, importance, fps=1):
        B, V, T, C, H, W = cam_videos.shape
        # combine temporal views into time dimension
        cam_videos = cam_videos.reshape(B, V*T, C, H, W)

        wandb.log({"explainer_video": wandb.Video(cam_videos, fps=fps, format="mp4")})

        # temporal importance chart
        table = wandb.Table(
            data=[[t, importance[b].mean(dim=0)[t].item(), f"video_{b}"] for b in range(B) for t in range(T)],
            columns=["frame","importance","series"]
        )
        line_chart = wandb.plot.line(
            table=table,
            x="frame",
            y="importance",
            stroke="series",
            title="Temporal Importance per Video/View"
        )
        wandb.log({"importance_line_chart": line_chart})
