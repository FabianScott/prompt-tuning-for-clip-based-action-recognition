import os
import torchvision.io as io
from .dataset_builders import build_dataset, split_trainlist
from ..configs.paths import ucf101_annotations_unzipped_classlist, ucf101_annotations_unzipped_trainlist1, ucf101_annotations_unzipped_testlist1, ucf_101_root, ucf101_annotations_unzipped_testlist2, ucf101_annotations_unzipped_testlist3
from .utils import load_video, resize_video

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchcodec.decoders import VideoDecoder


from typing import Callable, Optional


class UCF101Dataset(Dataset):
    def __init__(self, config: dict, reshaped_size=224, root: str = ucf_101_root, split="train", transform=None, **kwargs):
        self.root_dir = root
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
        self.resized_protocol = config["dataset-config"]["resized-cache-protocol"]
        self.val_proportion = config["dataset-config"]["val-proportion"]
        self.reshaped_size = reshaped_size


        if split == "test":
            annotations_file = ucf101_annotations_unzipped_testlist1
        elif split == "train" or split == "val":
            train_file = ucf101_annotations_unzipped_trainlist1 + f"-train-{"{:.2f}".format(self.val_proportion)}.txt"
            val_file = ucf101_annotations_unzipped_trainlist1 + f"-val-{"{:.2f}".format(self.val_proportion)}.txt"
            if not os.path.exists(train_file) or not os.path.exists(val_file):
                split_trainlist(
                    src_path=ucf101_annotations_unzipped_trainlist1,
                    val_ratio=self.val_proportion,
                    train_out=train_file,
                    val_out=val_file
                )
            annotations_file = train_file if split == "train" else val_file    
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'val'.")

        classes_file=ucf101_annotations_unzipped_classlist
        # Parse annotations
        self.samples = []
        class_to_idx = {}
        
        with open(classes_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                class_to_idx[name] = int(idx) - 1  # zero-based
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                vid_rel_path = parts[0]
                if len(parts) > 1:
                    label = int(parts[1]) - 1  # convert to 0-based
                else:
                    # fallback: extract class from folder name
                    label = class_to_idx[vid_rel_path.split("/")[0]]
                self.samples.append((vid_rel_path, label))

        self.class_names = sorted(list(class_to_idx.keys()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_rel_path, label = self.samples[idx]
        path = os.path.join(self.root_dir, vid_rel_path)
        
        video = load_video(
            path_or_bytes=path,
            num_frames_needed=self.num_frames_needed,
            spacing=self.spacing,
        )
        if video is None:
            return None
        if self.resized_protocol is not None:
            video = resize_video(video, protocol=self.resized_protocol, size=self.reshaped_size)
        if self.transform:
            video = self.transform(video)

        return video, label


def build_and_split_ucf101_dataset(
    data_dir: str,
    processor: CLIPProcessor,
    train_len: Optional[int]=None,
    train_prop: Optional[float]=None,
    val_len: Optional[int]=None,
    val_prop: Optional[float]=None,
    train_split_function: Callable=torch.utils.data.random_split,
    transform: Optional[Callable]=None,
    ) -> tuple[Dataset, Dataset]:

    # Ensure only one of train_len and train_prop is define
    if train_len is not None and train_prop is not None:
        raise ValueError("Specify only one of train_len or train_prop")
    if val_len is not None and val_prop is not None:
        raise ValueError("Specify only one of val_len or val_prop")
    dataset = build_dataset(
        data_dir=data_dir,
        processor=processor,
        transform=transform,
    )
    num_classes = len(dataset.class_names)
    print(f"Dataset has {len(dataset)} videos from {num_classes} classes.")

    train_len = train_len if train_prop is None else int(len(dataset) * train_prop)
    val_len = val_len if val_prop is None else int(len(dataset) * val_prop)
    print(f"Training set size: {train_len}, Validation set size: {val_len}, Full dataset size: {len(dataset)}")
    train_set, val_set = train_split_function(dataset, [train_len, val_len])

    return train_set, val_set