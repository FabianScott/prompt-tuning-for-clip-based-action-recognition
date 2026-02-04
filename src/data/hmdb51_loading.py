import os
import torch
import torchvision.io as io
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from ..configs.paths import hmdb51_root, hmdb51_classlist_file, hmdb51_trainlist, hmdb51_vallist


class HMDB51Dataset(Dataset):
    def __init__(self, config, root: str = hmdb51_root, split="train",  
                 classes_file=hmdb51_classlist_file, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.samples = []
        self.num_frames_needed = config["num-frames-temporal"] * config["num-temporal-views"]
        self.resized_protocol = config["dataset-config"]["resized-cache-protocol"]
        
        class_to_idx = {}
        with open(classes_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                class_to_idx[name] = int(idx)
        
        self.class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        # Parse annotations, test set is in vallist, no validation set provided so use K-shot from train set
        if split == "train":
            annotations_file = hmdb51_trainlist
        elif split == "test":
            annotations_file = hmdb51_vallist
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'.")
        
        with open(annotations_file, 'r') as f:
            for line in f:
                path, _, label_idx = line.strip().split()

                class_name = path.split("\\")[0]  # handles class\video_folder format
                class_idx = class_to_idx[class_name]

                folder_path = os.path.join(root, path.replace("\\", "/"))
                self.samples.append((folder_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        try:
            listed_dir = os.listdir(folder_path)
        except FileNotFoundError:
            print(f"Folder {folder_path} not found. Ignoring this sample.")
            return None

        frames = sorted([os.path.join(folder_path, f) for f in listed_dir if f.endswith('.jpg')])
        video = torch.stack([io.read_image(f).float() / 255.0 for f in frames]) # (T, C, H, W)
        
        if video.shape[0] < self.num_frames_needed:
            print(f"Video at {folder_path} has fewer frames ({video.shape[0]}) than num_frames_needed ({self.num_frames_needed}). Ignoring this sample.")
            return None

        video = self.transform(video) if self.transform else video
        return video, label