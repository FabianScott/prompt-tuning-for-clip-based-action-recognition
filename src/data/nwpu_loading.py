import itertools
import os
import cv2
import sys
import json
import math
from matplotlib.pylab import annotations
import torch
import torchvision.io as io

import shutil
from typing import Optional
from ..configs.paths import (
    nwpu_raw_root, 
    nwpu_root, 
    nwpu_raw_train_path, 
    nwpu_gt_npz_path, 
    nwpu_raw_annotations_path,
    nwpu_annotations_dict_path,
    nwpu_proper_annotations_path,
    nwpu_raw_test_path,
    )
from .utils import load_video
import numpy as np
from collections import defaultdict, deque


def get_video_metadata(video_path: str):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        cap.release()
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()

    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count
    }

def generate_action_detection_annotations(
        anomaly_ann_path: str = nwpu_gt_npz_path, 
        video_base_dir: str = nwpu_raw_train_path, 
        action_ann: str = nwpu_raw_annotations_path, 
        savepath: Optional[str] = nwpu_annotations_dict_path
        ) -> dict:
    # Load ground truth
    gt_npz = np.load(anomaly_ann_path)
    vid_name_list = list(gt_npz.keys())

    # Load text file action_ann into a dictionary. The format is like
    # D001_03 climbing_fence
    # D001_04 climbing_fence
    action_dict = {}
    with open(action_ann, "r") as f:
        for line in f:
            line = line.strip()
            vid_name, action = line.split()
            action_dict[vid_name] = action

    # Keep only vid_name_list for those also in action_dict
    vid_name_list = [vid_name for vid_name in vid_name_list if vid_name in action_dict]

    # Directory where original videos are stored
    annotations = {}
    for vid_name in vid_name_list:
        anomaly_gt_array: np.ndarray = gt_npz[vid_name]

        metadata = get_video_metadata(os.path.join(video_base_dir, f"{vid_name}.avi"))

        # Find indices where label is abnormal (1)
        abnormal_indices = np.where(anomaly_gt_array == 1)[0]

        if len(abnormal_indices) > 0:
            # Convert abnormal_indices to action spans, like [[0, 100], [200, 300]]
            # Find where consecutive sequence breaks (i.e., gap > 1)
            diff = np.diff(abnormal_indices)
            split_points = np.where(diff > 1)[0] + 1

            # Split into contiguous groups
            contiguous_groups = np.split(abnormal_indices, split_points)

            # Create spans [start, end] from each group
            action_spans = [[int(group[0]), int(group[-1])] for group in contiguous_groups]  # np.int64 to int
        else:
            action_spans = []

        annotations[vid_name] = {
            "metadata": metadata,
            "action_type": action_dict[vid_name],
            "action_spans": action_spans}

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved annotations to {savepath}")
    
    return annotations

def create_pytorch_dataset_folder(annotations: dict, root: str = nwpu_root, video_dir: str = nwpu_raw_train_path):
    os.makedirs(root, exist_ok=True)

    for vid_name, ann in annotations.items():
        # Create class directory
        class_dir = os.path.join(root, ann["action_type"])
        os.makedirs(class_dir, exist_ok=True)

        # Put video file into class directory
        src_video_path = os.path.join(video_dir, f"{vid_name}.avi")
        dst_video_path = os.path.join(class_dir, f"{vid_name}.avi")
        if os.path.exists(src_video_path):
            shutil.copy(src_video_path, dst_video_path)
        else:
            print(f"Source video not found: {src_video_path}")
            continue

    print(f"Created PyTorch dataset structure at: {root}")

class NWUPDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, annotations_path: str = nwpu_annotations_dict_path, root: str = nwpu_root, transform: Optional[callable] = None, **kwargs):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.root = root
        self.transform = transform
        nframes = config["num-frames-temporal"]
        spacing = config["space-between-frames"]
        if nframes and spacing:
            raise ValueError("Only one of num-frames-temporal or space-between-frames can be set")
        if nframes:
            self.num_frames_needed = nframes * config["num-temporal-views"]
        self.spacing = spacing
        # Create a list of (video_path, label) tuples
        self.samples = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(ann["action_type"] for ann in self.annotations.values())))}

        for vid_name, annotation in self.annotations.items():
            video_path = os.path.join(root, annotation["action_type"], f"{vid_name}.avi")
            if os.path.exists(video_path):
                self.samples.append((video_path, annotation["action_type"]))
            else:
                print(f"Video file not found in dataset: {video_path}")

        self.class_names = sorted(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = load_video(video_path, num_frames_needed=self.num_frames_needed, spacing=self.spacing)
        if self.transform:
            video = self.transform(video)
        
        return video, label


class NWUPAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict,  transform=None, frames_needed: int = 32, split: str = "train", **kwargs):
        self.transform = transform
        self.frames_needed = frames_needed
        self.normal_video_proportion = config["dataset-config"]["normal-video-proportion"]
        self.max_clip_length = config["dataset-config"]["clip-length"]
        self.binary_classification = config["dataset-config"]["binary-classification"]
        self.abnormal_clip_label = config["dataset-config"]["abnormal-clip-label"]
        self.normal_clip_label = config["dataset-config"]["normal-clip-label"]
        self.num_normal_output = config["dataset-config"]["num-model-outputs-for-normal"]
        k_train = config["dataset-config"]["K-train"]
        k_val = config["dataset-config"]["K-val"]
        k_test = config["dataset-config"]["K-test"]

        with open(nwpu_proper_annotations_path, 'r') as f:
            annotations = json.load(f)

        self.action_counts = defaultdict(lambda: 0)
        abnormal_samples = []
        self.path_to_indices = {}

        for vid_name, vid_info in annotations.items():
            path = os.path.join(nwpu_raw_test_path, f"{vid_name}.avi")
            spans = annotations[vid_name]["action_spans"]
            action_type = annotations[vid_name]["action_type"]
            
            if isinstance(action_type, str):
                action_types = [action_type] * len(spans)

            for action_type, span in zip(action_types, spans):            
                if isinstance(span[0], int):
                    span = [span]  # Wrap single span in a list
                
                for s in span:
                    # make the abnormal frame indices by centering the span, but ensure clip is max_clip_length if possible
                    # action_type = self.abnormal_clip_label if self.binary_classification else action_type
                    clip_start = max(0, s[0] - self.max_clip_length // 2)
                    clip_end = clip_start + self.max_clip_length
                    abnormal_frame_indices = torch.linspace(clip_start, clip_end, frames_needed).long()
                    if len(abnormal_frame_indices) == 0 or len(abnormal_frame_indices) < frames_needed:
                        print(f"Skipping {path} due to insufficient frames for abnormal span {s}")
                        continue
                    # self.paths_and_indices_per_action[action_type].append((path, abnormal_frame_indices))
                    self.action_counts[action_type] += 1
                    self.path_to_indices[path] = abnormal_frame_indices
                    abnormal_samples.append((path, action_type))

       # Add normal samples
        self.num_abnormal_samples = len(abnormal_samples)
        normal_samples, normal_path_to_indices = load_normal_nwpu_paths(
            num_videos=int(self.num_abnormal_samples * self.normal_video_proportion),
            max_clip_length=self.max_clip_length,
            num_frames_needed=self.frames_needed,
            label=self.normal_clip_label
        )

        self.action_counts[self.normal_clip_label] += len(normal_samples)
        self.path_to_indices.update(normal_path_to_indices)

        for action, num in self.action_counts.items():
            print(f"Action: {action} has {num} examples.")

        split_list = split_nwpu_dataset_classwise(
            normal_samples=normal_samples,
            abnormal_samples=abnormal_samples,
            k_train=k_train,
            k_val=k_val,
            k_test=k_test,
            normal_video_proportion=self.normal_video_proportion
        )
        self.normal_labels = [f"{self.normal_clip_label} {i}" for i in range(self.num_normal_output)]
        if self.binary_classification:
            self.class_names = self.normal_labels
            self.class_to_idx = {cls_name: 0 for cls_name in self.normal_labels + [self.normal_clip_label]}
        else:
            self.class_names = sorted(self.action_counts.keys())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
            # For multi-class there is already a single normal label in the class_names so add self.normal_labels[:-1]
            self.class_to_idx.update({nl: self.class_to_idx[self.normal_clip_label] for nl in self.normal_labels[:-1]})

        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.samples = [(path, self._class_to_idx(cls)) for path, cls in split_list[split]]

    def _class_to_idx(self, class_name: str) -> int:
        if class_name in self.normal_labels:
            return self.class_to_idx[self.normal_clip_label]
        elif self.binary_classification:
            return 0 if class_name in self.normal_labels + [self.normal_clip_label] else 1
        return self.class_to_idx[class_name]   

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        indices = self.path_to_indices[path]
        video = load_video(path, num_frames_needed=self.frames_needed, indices=indices)
        if video is None:
            return None
        if self.transform:
            video = self.transform(video)
        return video, label

def split_nwpu_dataset_classwise(normal_samples, abnormal_samples, k_train, k_val, k_test, normal_video_proportion):
    """
    Splits normal and abnormal samples into train, val, and test sets.
    Abnormal samples are distributed class-wise as evenly as possible.
    Normal samples are added proportionally to each split.
    """
    if k_train is None or k_val is None or k_test is None:
        raise ValueError("k_train, k_val, and k_test must be specified for NWPU dataset splitting.")
    # Group abnormal samples by class
    ab_by_class = defaultdict(list)
    for path, label in abnormal_samples:
        ab_by_class[label].append((path, label))

    splits = {"test": [], "val": [], "train": []}
    normal_idx = 0

    def select_normal(num_abnormal, normal_list, start_idx):
        num_normal = int(num_abnormal * normal_video_proportion)
        selected = normal_list[start_idx:start_idx + num_normal]
        return selected, start_idx + num_normal

    def assign_split(num_needed, split_name):
        assigned = []
        for cls, samples in ab_by_class.items():
            take = math.ceil(len(samples) / sum([k_train, k_val, k_test]) * num_needed)
            take = min(take, len(samples))
            assigned.extend(samples[:take])
            ab_by_class[cls] = samples[take:]  # remove assigned
        return assigned
    # Test split
    test_abnormal = assign_split(k_test, "test")
    test_normal, normal_idx = select_normal(len(test_abnormal), normal_samples, normal_idx)
    splits["test"] = test_abnormal + test_normal

    # Validation split
    val_abnormal = assign_split(k_val, "val")
    val_normal, normal_idx = select_normal(len(val_abnormal), normal_samples, normal_idx)
    splits["val"] = val_abnormal + val_normal

    # Train split
    train_abnormal = assign_split(k_train, "train")
    train_normal, normal_idx = select_normal(len(train_abnormal), normal_samples, normal_idx)
    splits["train"] = train_abnormal + train_normal

    return splits

def load_normal_nwpu_paths(num_videos: int, max_clip_length: int, num_frames_needed: int, min_clip_length: int = 1, label: str = "normal"):
    filenames = os.listdir(nwpu_raw_train_path)

    paths_per_camera = defaultdict(deque)
    for f in filenames:
        if f.endswith('.avi'):
            cam = f.split('_')[0]
            paths_per_camera[cam].append(os.path.join(nwpu_raw_train_path, f))

    cameras = [q for q in paths_per_camera.values() if q]
    selected = []
    path_to_indices = {}
    leftover_frames = defaultdict(int)

    while cameras and len(selected) < num_videos:
        for q in list(cameras):
            if not q:
                cameras.remove(q)
                continue

            path = q.popleft()
            meta = get_video_metadata(path)
            available_length = meta["frame_count"] - leftover_frames[path]
            max_length = int(max_clip_length * meta["fps"])
            min_length = int(min_clip_length * meta["fps"])

            if available_length < min_length:
                continue  # skip if remaining clip too short

            indices = get_indices_for_nwpu(
                video_length=available_length,
                max_clip_length=max_length,
                num_frames_needed=num_frames_needed
            )
            selected.append((path, label))
            path_to_indices[path] = indices

            leftover_frames[path] = max(0, available_length - len(indices))
            if leftover_frames[path] > 0:
                q.appendleft(path)  # push back for next round

            if len(selected) == num_videos:
                break

    if len(selected) < num_videos:
        print(f"Warning: found only {len(selected)} normal videos, requested {num_videos}")

    return selected, path_to_indices



def get_indices_for_nwpu(video_length: int, max_clip_length: int, num_frames_needed: int) -> torch.Tensor:
    """
    Return a linspace across the video length, capped at max_clip_length.
    Will return the middle portion of the video if video_length > max_clip_length.
    Args:
        video_length (int): Total number of frames in the video.
        max_clip_length (int): Maximum length of the clip to consider.
        num_frames_needed (int): Number of frames to sample.
    Returns:
        torch.Tensor: Tensor of frame indices.
    """
    start_frame = max(0, (video_length - max_clip_length) // 2)
    end_frame = min(video_length - 1, start_frame + max_clip_length - 1)
    return torch.linspace(start_frame, end_frame, num_frames_needed).long()