"""
Video loading, caching, and utility functions.
"""
import os
import cv2
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from torchvision import transforms
from typing import Optional


def pad_collate(batch, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Collate function that pads videos to same length.
    
    Args:
        batch: List of (video, label) tuples
        device: Device for tensors
        
    Returns:
        Tuple of (padded_videos, masks, labels)
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # or raise to let DataLoader skip the batch

    videos, labels = zip(*batch)
    max_len = max(v.shape[1] for v in videos)
    padded_videos, masks = [], []
    needs_mask = any(v.shape[1] != max_len for v in videos)
    for video in videos:
        V, T, C, H, W = video.shape
        if T < max_len:
            pad = torch.zeros(
                (max_len - T, C, H, W),
                dtype=video.dtype,
                device=video.device
            ).unsqueeze(0).expand(V, -1, -1, -1, -1)
            v_padded = torch.cat([video, pad], dim=1)
        else:
            v_padded = video
        padded_videos.append(v_padded)
        
        if needs_mask:
            mask = torch.cat((
                torch.ones(V, T, device=video.device),
                torch.zeros(V, max_len - T, device=video.device)
            ), dim=1)
            masks.append(mask)

    stacked_vids = torch.stack(padded_videos)
    stacked_masks = torch.stack(masks) if needs_mask else None
    labels_tensor = torch.tensor(labels, dtype=torch.long) if isinstance(labels[0], int) else torch.stack(labels)
    # if device is not None:
    #     stacked_vids = stacked_vids.to(device)
    #     stacked_masks = stacked_masks.to(device) if needs_mask else None
    #     labels = torch.tensor(labels, device=device)

    return (
        stacked_vids,
        stacked_masks,
        labels_tensor
    )


def load_video(path_or_bytes: str, num_frames_needed: Optional[int] = None, spacing: Optional[int] = None, indices: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """"
    Function to load video frames using torchcodec's VideoDecoder.
    Args:
        path (str): Path to the video file.
        num_frames_needed (Optional[int]): Number of frames to extract. If specified, frames will be evenly spaced.
        spacing (Optional[int]): Spacing between frames. If specified, frames will be extracted at this interval.
    Returns:
        torch.Tensor: Extracted video frames of shape [N, C, H, W] or None if loading fails.
    """
    try:
        decoder = VideoDecoder(path_or_bytes, device="cpu", seek_mode="approximate", dimension_order="NCHW")
    except Exception as e:
        print(f"Error initializing VideoDecoder for {path_or_bytes}: {e}")
        return None
    total_frames = decoder.metadata.num_frames
    if total_frames < num_frames_needed:
        print(f"Not enough frames in {path_or_bytes}, removing file")
        os.unlink(path_or_bytes)
        return None

    if num_frames_needed:
        indices = torch.linspace(0, max(total_frames - 1, 0),
                                min(num_frames_needed, total_frames)).long()
    elif spacing:
        indices = torch.arange(0, total_frames, spacing).long()
    else:
        indices = torch.arange(total_frames).long()

    # Now ensure indices are within bounds
    if any(indices >= total_frames):
        print(f"Warning: some frame indices exceed total frames in {path_or_bytes}. {indices}, {total_frames}.")
    
    try:
        frames = decoder.get_frames_at(indices).data  # uint8 [N,C,H,W]
    except Exception as e:
        print(f"Error decoding video {path_or_bytes}: {e}")
        return None

    if num_frames_needed and frames.shape[0] < num_frames_needed:
        print(f"Warning: only {frames.shape[0]} frames decoded from {path_or_bytes}, expected at least {num_frames_needed}")
        return None
    return frames

def read_cached_video(cache_path: str) -> Optional[torch.Tensor]:
    if os.path.exists(cache_path):
        if cache_path.endswith(".pt") or cache_path.endswith(".pth"):
            try:
                video = torch.load(cache_path)
                return video
            except Exception as e:
                print(f"Error loading cached video {cache_path}: {e}")
        else:
            try:
                # Read MJPEG using OpenCV
                cap = cv2.VideoCapture(cache_path)
                frames_list = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))  # C,H,W
                cap.release()
                video = torch.stack(frames_list).to(torch.uint8)  # [N,C,H,W]
                return video
            except Exception as e:
                print(f"Error loading cached video {cache_path}: {e}")
    return None

def save_video_to_cache(video: torch.Tensor, path: str):
    import os
    import cv2
    if path.endswith('.avi'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save frames to MJPEG AVI using OpenCV
        T, C, H, W = video.shape
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(path, fourcc, 30, (W, H))  # FPS=30
        for f in video:
            frame_np = f.permute(1, 2, 0).numpy()  # H,W,C
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
    elif path.endswith(".pt") or path.endswith(".pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(video, path)
    else:
        raise ValueError(f"Unsupported cache file extension in {path}")

def do_class_based_split(samples: list[tuple], class_protocol: str, num_classes: Optional[int]=None):
    """
    Given a dataset's samples list, keep only the classes desired according to the 
    desired classes protocol.
    """

    if class_protocol == "all":
        return samples
    elif class_protocol == "most-frequent":
        if num_classes is None:
            raise ValueError("desired_num_classes must be specified for 'most-frequent' protocol")
        from collections import Counter
        class_counts = Counter(label for _, label in samples)
        most_common = set(label for label, _ in class_counts.most_common(num_classes))
        filtered_samples = [s for s in samples if s[1] in most_common]
        return filtered_samples
    elif class_protocol == "least-frequent":
        if num_classes is None:
            raise ValueError("desired_num_classes must be specified for 'least-frequent' protocol")
        from collections import Counter
        class_counts = Counter(label for _, label in samples)
        least_common = set(label for label, _ in class_counts.most_common()[:-num_classes-1:-1])
        filtered_samples = [s for s in samples if s[1] in least_common]
        return filtered_samples
    elif class_protocol == "random":
        if num_classes is None:
            raise ValueError("desired_num_classes must be specified for 'random' protocol")
        import random
        all_classes = set(label for _, label in samples)
        random_classes = set(random.sample(all_classes, num_classes))
        filtered_samples = [s for s in samples if s[1] in random_classes]
        return filtered_samples
    else:
        raise ValueError(f"Unknown class_protocol: {class_protocol}")

def get_class_names_and_class_to_idx(samples):
    class_names = sorted(set(label for _, label in samples))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx

def resize_video(video: torch.Tensor, protocol: str, size: int=224) -> torch.Tensor:
    T, C, H, W = video.shape
    if protocol == "preserve-aspect-ratio":
        # Preserve aspect ratio and resize so shortest side is min_size
        if H < W:
            new_size = (size, int(W * (size / H)))
        else:
            new_size = (int(H * (size / W)), size)
    elif protocol == "interpolate-smallest-side":
        # Resize so shortest side is min_size, keep longest side at its current size
        if H < W:
            new_size = (size, W)
        else:
            new_size = (H, size)
    elif protocol == "interpolate-longest-side":
        # Resize so longest side is max_size, keep shortest side at its current size
        if H > W:
            new_size = (size, W)
        else:
            new_size = (H, size)
    else:
        raise ValueError(f"Unknown resize protocol: {protocol}")
    if new_size == (H, W):
        return video
    # Ensure both sides at least size
    new_size = (max(new_size[0], size), max(new_size[1], size))
    video = F.interpolate(video, size=new_size, mode='bilinear', align_corners=False)
    return video


def remove_black_strips(video: torch.Tensor) -> torch.Tensor:
    """
    Remove black strips from the top and bottom of the video frames.
    Args:
        video (torch.Tensor): Video tensor of shape [T, C, H, W].
    Returns:
        torch.Tensor: Cropped video tensor.
    """
    T, C, H, W = video.shape
    top, bottom = find_black_strips(video)
    # Crop video
    cropped_video = video[:, :, top:bottom, :]
    return cropped_video

def find_black_strips(video: torch.Tensor):
    """
    Find the coordinates of black strips in the video frames.
    Args:
        video (torch.Tensor): Video tensor of shape [T, C, H, W].
    Returns:
        tuple: (top, bottom) coordinates of the black strips.
    """
    T, C, H, W = video.shape
    gray_video = 0.2989 * video[:, 0] + 0.5870 * video[:, 1] + 0.1140 * video[:, 2]  # [T, H, W]
    row_energy = gray_video.max(dim=0).values.mean(dim=1)
    
    # Adaptive threshold based on dtype
    if video.dtype == torch.uint8:
        threshold = 15  # For uint8 (0-255), catch dark pixels < 15
    else:
        threshold = 0.05  # For float (0-1)
    
    non_black = row_energy > threshold
    non_black_indices = torch.where(non_black)[0]
    if len(non_black_indices) == 0:
        # All black, return full range
        return 0, H
    top = non_black_indices[0].item()
    bottom = non_black_indices[-1].item() + 1
    return top, bottom