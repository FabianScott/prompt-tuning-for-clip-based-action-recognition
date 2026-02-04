"""
DataLoader construction and transform building utilities.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPImageProcessor
from typing import Callable, Optional

from .utils import pad_collate, remove_black_strips
from .transforms import (
    get_augmentations,
    TemporalViewTransform,
    SpatialCropTransform
)



def build_dataloaders(
        train_set: Dataset,
        val_set: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None,
        shuffle_train: bool = True,
        seed: int = 42,
        start_idx_train: Optional[int] = None,
        start_idx_val: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders with deterministic shuffling.
    
    Args:
        train_set: Training dataset
        val_set: Validation dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        collate_fn: Custom collate function
        shuffle_train: Whether to shuffle training data
        seed: Random seed for reproducibility
        start_idx_train: Optional starting index for training subset
        start_idx_val: Optional starting index for validation subset
        device: Device for pin_memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    sampler_train = torch.utils.data.SubsetRandomSampler(
        list(range(start_idx_train, len(train_set))), generator=g
    ) if start_idx_train is not None else None
    sampler_val = torch.utils.data.SubsetRandomSampler(
        list(range(start_idx_val, len(val_set)))
    ) if start_idx_val is not None else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=pad_collate if collate_fn is None else collate_fn,
        generator=g,
        sampler=sampler_train,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        # multiprocessing_context='spawn'
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate if collate_fn is None else collate_fn,
        sampler=sampler_val,
        pin_memory=True if device == "cuda" else False,
        # multiprocessing_context='spawn'
    )

    return train_loader, val_loader


def build_dataloader(
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        collate_fn: Optional[Callable] = None
) -> DataLoader:
    """Build a single dataloader for a dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of dataloader workers
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate if collate_fn is None else collate_fn,
    )
    return dataloader


def add_view_dim(video):
    """Add a view dimension to the video tensor if not present.
    
    Args:
        video: Tensor of shape [T,C,H,W] or [V,T,C,H,W]
        
    Returns:
        Tensor of shape [V,T,C,H,W]
    """
    if len(video.shape) == 4:
        return video.unsqueeze(0)
    elif len(video.shape) == 5:
        return video
    else:
        raise ValueError(f"Video tensor must have 4 or 5 dimensions, got {len(video.shape)}")


def get_transform(
        num_temporal_views: int,
        num_frames_temporal: int,
        num_spatial_views: int,
        processor: CLIPImageProcessor,
        dtype=torch.bfloat16,
        augmentation_dict: Optional[dict] = None,
        do_remove_black_strips: bool = True,
):
    """Build transform pipeline based on view configuration.
    
    Args:
        num_temporal_views: Number of temporal views
        num_frames_temporal: Number of frames per temporal view
        num_spatial_views: Number of spatial crops
        processor: CLIP image processor
        dtype: Target dtype for video tensors
        augmentation_dict: Optional augmentation configuration
        do_remove_black_strips: Whether to remove black strips from videos
        
    Returns:
        Composed transform pipeline
    """
    # Apply remove_black_strips FIRST, before any resizing
    strip_removal = [remove_black_strips] if do_remove_black_strips else []
    
    base_transform = [
        transforms.Resize(processor.size['shortest_edge']),  # Single int preserves aspect ratio
        transforms.CenterCrop(processor.size['shortest_edge']),
        transforms.ConvertImageDtype(dtype),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ]

    aug_transforms = get_augmentations(augmentation_dict, processor)

    if num_temporal_views * num_spatial_views == 1:
        return transforms.Compose(
            strip_removal + base_transform + [add_view_dim] + aug_transforms  # Add view dimension
        )
    elif num_temporal_views > 1 and num_spatial_views == 1:
        return transforms.Compose(
            strip_removal + base_transform + [
                TemporalViewTransform(num_views=num_temporal_views, num_frames=num_frames_temporal)
            ] + aug_transforms
        )
    elif num_spatial_views > 1:
        return transforms.Compose(
            strip_removal + [
                transforms.ConvertImageDtype(dtype),
                TemporalViewTransform(num_views=num_temporal_views, num_frames=num_frames_temporal),
                SpatialCropTransform(num_spatial_views=num_spatial_views, size=processor.size['shortest_edge']),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ] + aug_transforms
        )
    elif num_temporal_views == 1 and num_spatial_views > 1:
        return transforms.Compose(
            strip_removal + [
                transforms.ConvertImageDtype(dtype),
                add_view_dim,  # Add temporal view dimension
                SpatialCropTransform(num_spatial_views=num_spatial_views, size=processor.size['shortest_edge']),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ] + aug_transforms
        )
