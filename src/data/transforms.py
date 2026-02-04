"""
Transform classes and augmentation utilities for video data processing.
"""
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF, RandomErasing
from transformers import CLIPImageProcessor
from typing import Optional


class ExtractFramesTransform:
    """Extract frames from video at specified intervals."""
    
    def __init__(
            self,
            num_frames: Optional[int] = None,
            space_between_frames: Optional[int] = None
    ):
        self.num_frames = num_frames
        self.space_between_frames = space_between_frames
        if self.space_between_frames is not None and self.num_frames is not None:
            raise ValueError("Specify only one of num_frames or space_between_frames.")
    
    def __call__(self, video):
        if self.num_frames is None and self.space_between_frames is None:
            return video
        total_frames = video.shape[0]
        if self.space_between_frames is None:
            # evenly spaced frames
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames, device=video.device).long()
        else:
            indices = torch.arange(0, total_frames, step=self.space_between_frames, device=video.device)

        return video[indices]


class TemporalViewTransform:
    """Split video into multiple temporal views."""
    
    def __init__(self, num_views, num_frames):
        self.num_views = num_views
        self.num_frames = num_frames

    def __call__(self, video):
        """
        Takes a video tensor of shape [T, C, H, W] and returns
        a tensor of shape [num_views, num_frames, C, H, W]
        """
        if video is None:
            return None
        T, C, H, W = video.shape

        if self.num_frames * self.num_views > T:
            print(f"num_frames * num_views {self.num_frames * self.num_views} exceeds video length {T}.")
            return None
        # evenly spaced frames
        T_downsampled = self.num_frames * self.num_views
        indices = torch.linspace(0, T - 1, steps=T_downsampled, device=video.device).long()
        video = video[indices]

        # compute temporal indices for views
        chunk_len = T_downsampled // self.num_views  # = self.num_frames
        # some frames might be dropped if T is not divisible by num_views
        chunk_indices = torch.arange(T_downsampled).view(self.num_views, chunk_len)
        
        # gather temporal chunks: [num_views, chunk_len, C, H, W]
        views = video[chunk_indices]  # [num_views, chunk_len, C, H, W]
        return views  # [num_views, chunk_len, C, H, W]


class SpatialCropTransform:
    """Apply spatial crops (center, left, right) to video."""
    
    def __init__(self, num_spatial_views, size):
        if num_spatial_views not in [1, 3]:
            raise ValueError("num_spatial_views must be 1 or 3.")
        self.num_s = num_spatial_views
        self.size = size

        # define crop types
        self.crops = ['center', 'left', 'right'][:self.num_s]

    def __call__(self, video_chunks):
        if video_chunks is None:
            return None
        num_t, num_f, C, H, W = video_chunks.shape

        # scale so smallest side == self.size
        scale = self.size / min(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        if (H, W) != (new_h, new_w):
            video_chunks = F.interpolate(
                video_chunks.view(-1, C, H, W),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).view(num_t, num_f, C, new_h, new_w)
        H, W = new_h, new_w
        c = self.size

        # crop only along the larger dimension
        if H > W:  # vertical crops
            starts = [(H-c)//2] if self.num_s == 1 else [0, (H-c)//2, H-c]
            crop_coords = [(y, 0) for y in starts]
        else:      # horizontal crops
            starts = [(W-c)//2] if self.num_s == 1 else [0, (W-c)//2, W-c]
            crop_coords = [(0, x) for x in starts]

        crops = []
        for y, x in crop_coords:
            crops.append(video_chunks[..., y:y+c, x:x+c])
        crops = torch.stack(crops, dim=1)  # [num_t, num_s, num_f, C, c, c]
        return crops.view(-1, num_f, C, c, c)


class RandomAffine:
    """Apply random affine transformations to video clips."""
    
    def __init__(self, rotation=None, shear=None):
        self.rotation = rotation
        self.shear = shear

    def __call__(self, clip: torch.Tensor):
        """
        clip: (V, T, C, H, W)
        """
        V, T, C, H, W = clip.shape

        angle = 0.0
        shear = [0.0, 0.0]

        if self.rotation is not None:
            angle = random.uniform(-self.rotation, self.rotation)

        if self.shear is not None:
            s = self.shear
            shear = [random.uniform(-s, s), random.uniform(-s, s)]

        out = torch.empty_like(clip)

        for v in range(V):
            for t in range(T):
                out[v, t] = TF.affine(
                    clip[v, t],
                    angle=angle,
                    translate=[0, 0],
                    scale=1.0,
                    shear=shear,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )

        return out


class GaussianBlur:
    """Apply Gaussian blur to video clips."""
    
    def __init__(self, kernel_size: int, sigma: float):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.torch_transform = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        V, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)  # (V*T, C, H, W)
        x = self.torch_transform(x)
        x = x.view(V, T, C, H, W)
        return x


class CutoutTransform:
    """Apply random erasing/cutout to video clips."""
    
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.transform = RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Applies the same RandomErasing mask to all frames and views.
        
        Args:
            clip: Tensor of shape (V,T,C,H,W) or (T,C,H,W)
            
        Returns:
            Tensor with cutout applied
        """
        if random.random() > self.p:
            return clip  # skip

        if len(clip.shape) == 5:
            V, T, C, H, W = clip.shape
            # generate mask on first frame of first view
            masked = self.transform(clip[0, 0])
            # compute the difference to apply the same mask to others
            mask = masked != clip[0, 0]
            # add mask to all frames/views
            clip = clip.masked_fill(mask.unsqueeze(0).unsqueeze(0), 0)
            return clip
        else:
            raise ValueError(f"Clip must have 5 dimensions, got shape {clip.shape}")


class RotationTransform:
    """Apply random rotation to video clips."""
    
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Applies the same random rotation to all frames in all views.
        
        Args:
            clip: Tensor of shape (V,T,C,H,W) or (T,C,H,W)
            
        Returns:
            Rotated clip tensor
        """
        angle = random.uniform(-self.degrees, self.degrees)

        if len(clip.shape) == 5:
            V, T, C, H, W = clip.shape
            rotated_clip = torch.stack([
                torch.stack([TF.rotate(clip[v, t], angle=angle, fill=0) for t in range(T)])
                for v in range(V)
            ])
        elif len(clip.shape) == 4:
            T, C, H, W = clip.shape
            rotated_clip = torch.stack([TF.rotate(clip[t], angle=angle) for t in range(T)])
        else:
            raise ValueError(f"Clip must have 4 or 5 dimensions, got {len(clip.shape)}")
        return rotated_clip


def salt_and_pepper_normalized(x: torch.Tensor, prob: float, mean, std):
    """Apply salt and pepper noise to normalized video tensor.
    
    Args:
        x: Tensor of shape (V,T,C,H,W) AFTER normalization
        prob: Probability of noise
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Noisy tensor
    """
    V, T, C, H, W = x.shape
    if prob <= 0:
        return x

    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)

    # normalized values corresponding to 0 and 1 in image space
    black = (0.0 - mean) / std
    white = (1.0 - mean) / std

    rand = torch.rand_like(x)
    salt = torch.bitwise_and(prob > rand, rand > (prob / 2))
    pepper = rand < prob / 2

    out = x.clone()
    out[pepper] = black.expand_as(out)[pepper]
    out[salt] = white.expand_as(out)[salt]
    return out


def get_augmentations(augmentation_dict: Optional[dict], processor: CLIPImageProcessor):
    """Build list of augmentation transforms from configuration dictionary.
    
    Args:
        augmentation_dict: Dictionary specifying augmentation parameters
        processor: CLIP image processor for normalization parameters
        
    Returns:
        List of augmentation transforms
    """
    aug_transforms = []
    if augmentation_dict is not None:
        # Add augmentations before normalization
        if augmentation_dict.get("random-horizontal-flip", None) is not None:
            aug_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation_dict["random-horizontal-flip"]["p"]
                )
            )
        if augmentation_dict.get("random-vertical-flip", None) is not None:
            aug_transforms.append(
                transforms.RandomVerticalFlip(
                    p=augmentation_dict["random-vertical-flip"]["p"]
                )
            )
        if augmentation_dict.get("color-jitter", None) is not None:
            aug_transforms.append(
                transforms.ColorJitter(
                    brightness=augmentation_dict["color-jitter"]["brightness"],
                    contrast=augmentation_dict["color-jitter"]["contrast"],
                    saturation=augmentation_dict["color-jitter"]["saturation"],
                    hue=augmentation_dict["color-jitter"]["hue"],
                )
            )
        if augmentation_dict.get("gaussian-blur", None) is not None:
            aug_transforms.append(
                GaussianBlur(
                    kernel_size=augmentation_dict["gaussian-blur"]["kernel-size"],
                    sigma=augmentation_dict["gaussian-blur"]["sigma"],
                )
            )
        if augmentation_dict.get("salt-and-pepper-noise", None) is not None:
            aug_transforms.append(lambda x: salt_and_pepper_normalized(
                x,
                prob=augmentation_dict["salt-and-pepper-noise"]["probability"],
                mean=processor.image_mean,
                std=processor.image_std
            ))
        # if augmentation_dict.get("random-rotation", None) is not None:
        #     aug_transforms.append(
        #         RotationTransform(
        #             degrees=augmentation_dict["random-rotation"]["rotation-degrees"]
        #         )
        #     )
        if augmentation_dict.get("cutout", None) is not None:
            aug_transforms.append(
                CutoutTransform(
                    p=augmentation_dict["cutout"]["p"],
                    scale=augmentation_dict["cutout"]["scale"],
                    ratio=augmentation_dict["cutout"]["ratio"],
                    value=0.0,
                )
            )
    return aug_transforms
