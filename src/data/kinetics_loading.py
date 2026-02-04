import os
from typing import Optional
import av
import cv2
import torch

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_video
from torchcodec.decoders import VideoDecoder
from torchvision import transforms
from ..configs.paths import (
    kinetics400_train_path,
    kinetics400_val_path,
    kinetics400_test_path,
    kinetics400_cache_root,
)
from .utils import load_video, read_cached_video, save_video_to_cache, do_class_based_split, resize_video

class Kinetics400(Dataset):
    def __init__(self, split, config: dict, reshaped_size=224, transform=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        if split == "train":
            self.root = kinetics400_train_path
        elif split == "val":
            self.root = kinetics400_val_path
        elif split == "test":
            self.root = kinetics400_test_path
        else:
            raise ValueError("split must be 'train','val','test'")

        # Validate sampling config
        nframes = config["num-frames-temporal"]
        spacing = config["space-between-frames"]
        if nframes and spacing:
            raise ValueError("Only one of num-frames-temporal or space-between-frames can be set")
        if nframes:
            self.num_frames_needed = nframes * config["num-temporal-views"]
        self.spacing = spacing
        self.transform = transform
        self.use_cache = config["dataset-config"]["use-cache"]
        self.save_to_cache = config["dataset-config"]["save-to-cache"]
        self.cache_extension = ".avi"  # Using MJPEG AVI for cached videos
        self.resized_protocol = config["dataset-config"]["resized-cache-protocol"]
        self.reshaped_size = reshaped_size
        # Collect samples
        self.class_names = sorted(os.listdir(self.root))
        self.samples = [
            (os.path.join(self.root, c, f), c)
            for c in self.class_names
            for f in os.listdir(os.path.join(self.root, c))
            if f.endswith(".mp4")
        ]
        self.samples = do_class_based_split(
            samples=self.samples,
            class_protocol=config["dataset-config"]["class-protocol"],
            num_classes=config["dataset-config"]["num-classes"]
        )
        self.class_names = sorted(set(label for _, label in self.samples))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Name depends on whether using resized cache or not
        cache_path = os.path.join(
            kinetics400_cache_root,
            f"{self.num_frames_needed}-{self.spacing}",
            f"{os.path.basename(path)}{self.cache_extension}",
        ) 
        cache_path_resized = os.path.join(
            kinetics400_cache_root,
            f"{self.num_frames_needed}-{self.spacing}",
            f"{self.resized_protocol}_{self.reshaped_size}",
            f"{os.path.basename(path)}{self.cache_extension}",
        ) if self.resized_protocol is not None else None
        # Start by trying to read from the resized cache if applicable
        video = read_cached_video(cache_path=cache_path_resized) if self.use_cache and self.resized_protocol is not None else None
        # Ensure the full size cache is used if resized cache is not available
        if video is None and self.use_cache:
            print(f"Resized cache not used or unavailable, trying full size cache for {path}") if self.resized_protocol is not None else None
            video = read_cached_video(cache_path=cache_path)
        # Revert to the original video if no cache available/used
        if video is None:
            video = load_video(path, self.num_frames_needed, self.spacing)  # [N,C,H,W] or None
            if video is None:
                return None
            if video.shape[0] < self.num_frames_needed:
                print(f"Video at {path} has only {video.shape[0]} frames, needed {self.num_frames_needed}. Skipping.")
                return None
        # Resize if required and save to the appropriate cache
        if self.resized_protocol is not None:
            video = resize_video(video, protocol=self.resized_protocol, size=self.reshaped_size)            
            if self.save_to_cache and not os.path.exists(cache_path_resized):
                save_video_to_cache(video, cache_path_resized)
        else:
            if self.save_to_cache:
                save_video_to_cache(video, cache_path)

        # Apply any transforms
        if self.transform:
            try:
                video = self.transform(video)
            except Exception as e:
                print(f"Error applying transform to video at {path}: {e.with_traceback(e.__traceback__)}")
                return None
        label_idx = self.class_to_idx[label]
        return video, label_idx



class VitaAuthorKineticsDataset(Dataset):
    # Class for testing the speed of the original vita author's implementation. 3/4x slower than mine
    def __init__(self, split, config: dict, transform=None, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        if split == "train":
            self.root = kinetics400_train_path
        elif split == "val":
            self.root = kinetics400_val_path
        elif split == "test":
            self.root = kinetics400_test_path
        else:
            raise ValueError("split must be 'train','val','test'")

        # Validate sampling config
        nframes = config["num-frames-temporal"]
        spacing = config["space-between-frames"]
        if nframes and spacing:
            raise ValueError("Only one of num-frames-temporal or space-between-frames can be set")
        if nframes:
            self.num_frames_needed = nframes * config["num-temporal-views"]
        self.spacing = spacing
        self.transform = transform
        self.device = device
        self.use_cache = config["dataset-config"]["use-cache"]
        self.save_to_cache = config["dataset-config"]["save-to-cache"]
        self.cache_extension = ".avi"  # Using MJPEG AVI for cached videos

        # Collect samples
        self.class_names = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = [
            (os.path.join(self.root, c, f), self.class_to_idx[c])
            for c in self.class_names
            for f in os.listdir(os.path.join(self.root, c))
            if f.endswith(".mp4")
        ]
        self.mean = 1
        self.std = 1
        self.spatial_size = 224
        self.sampling_rate = 24
        self.num_temporal_views = 4
        self.num_spatial_views = 3
        self.num_frames = 8

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        container = av.open(path)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]

        frames = [x.to_rgb().to_ndarray() for x in frames]
        frames = torch.as_tensor(np.stack(frames))
        frames = frames.float() / 255.

        frames = (frames - self.mean) / self.std
        frames = frames.permute(3, 0, 1, 2) # C, T, H, W
        
        if frames.size(-2) < frames.size(-1):
            new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
            new_height = self.spatial_size
        else:
            new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
            new_width = self.spatial_size
        frames = torch.nn.functional.interpolate(
            frames, size=(new_height, new_width),
            mode='bilinear', align_corners=False,
        )

        frames = self._generate_spatial_crops(frames)
        frames = sum([self._generate_temporal_crops(x) for x in frames], [])
        if len(frames) > 1:
            frames = torch.stack(frames)

        # Back to T, C, H, W
        frames = frames.permute(0, 2, 1, 3, 4)
        return frames, label


    def _generate_temporal_crops(self, frames):
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices