"""
Dataset building and construction utilities.
"""
import random
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from typing import Optional, Callable
from collections import defaultdict

from ..configs.datasets_and_methods import dataset_names


def build_dataset(
        name: str,
        processor: CLIPProcessor,
        config: dict,
        split: str = "train",
        transform: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        use_videomix: bool = False,
        use_augmentation: bool = False,
        **kwargs,
    ) -> Dataset:
    """Build a video dataset by name.
    
    Args:
        name: Dataset name (ucf101, hmdb51, kinetics400, nwpu)
        processor: CLIP processor for preprocessing
        config: Configuration dictionary
        split: Data split (train/val/test)
        transform: Optional custom transform
        data_dir: Optional data directory
        use_videomix: Whether to wrap with VideoMix
        use_augmentation: Whether to apply augmentations
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Constructed dataset
    """
    from .dataloading import get_transform
    
    use_augmentation = config["use-augmentation"] if "use-augmentation" in config else use_augmentation
    transform = get_transform(
        num_temporal_views=config["num-temporal-views"],
        num_frames_temporal=config["num-frames-temporal"],
        num_spatial_views=config["num-spatial-views"],
        processor=processor.image_processor,
        augmentation_dict=config["augmentation-dict"] if use_augmentation else None,
        do_remove_black_strips=config["dataset-config"]["remove-black-strips"]
    ) if transform is None else transform

    if name.lower() == "ucf101":
        from .ucf101_loading import UCF101Dataset
        from ..configs.paths import ucf101_annotations_unzipped_classlist

        dataset = UCF101Dataset(
            classes_file=ucf101_annotations_unzipped_classlist,
            transform=transform,
            split=split,
            config=config,
            **kwargs
        )
        print(f"Built UCF101 {split} dataset with {len(dataset)} samples and {len(dataset.class_names)} classes.")
    elif name.lower() == "hmdb51":
        from .hmdb51_loading import HMDB51Dataset
        dataset = HMDB51Dataset(
            transform=transform,
            split=split,
            config=config,
            **kwargs
        )
        print(f"Built HMDB51 {split} dataset with {len(dataset)} samples and {len(dataset.class_names)} classes.")
    elif name.lower() == "kinetics400":
        from .kinetics_loading import Kinetics400, VitaAuthorKineticsDataset
        dataset = Kinetics400(
            transform=transform,
            split=split,
            config=config,
            reshaped_size=processor.image_processor.size['shortest_edge'],
            **kwargs
        )
        print(f"Built Kinetics400 {split} dataset with {len(dataset)} samples and {len(dataset.class_names)} classes.")
    elif name.lower() == "nwpu":
        from .nwpu_loading import NWUPAnomalyDataset
        dataset = NWUPAnomalyDataset(
            transform=transform,
            config=config,
            split=split,
            **kwargs
        )
        print(f"Built NWUP Anomaly-Only {split} dataset with {len(dataset)} samples and {len(dataset.class_names)} classes.")
    else:
        raise ValueError(f"Dataset {name} not recognized. Available: {dataset_names}")
    
    if use_videomix:
        from .videomix import VideoMixDataset
        dataset = VideoMixDataset(base_dataset=dataset, config=config)
        print(f"Wrapped dataset with VideoMix of type {config['videomix-type']}, with probability {config['videomix-prob']}.")
    
    return dataset


def build_K_shot_datasets(
    dataset: Dataset,
    train_examples_per_class: Optional[int] = 2,
    val_examples_per_class: Optional[int] = 20,
    remove_class_if_insufficient: bool = False,
    use_excess_for: str = "none",
) -> tuple[Dataset, Dataset, list[str]]:
    """Build K-shot train and validation datasets.
    
    Ensures a balanced validation set with the parameter val_examples_per_class, 
    default is 20. The Dataset class must have a 'samples' attribute with 
    (data, label) tuples and a class_indices attribute mapping class indices 
    to sample indices.
    
    Args:
        dataset: Base dataset to split
        train_examples_per_class: Number of training examples per class
        val_examples_per_class: Number of validation examples per class
        remove_class_if_insufficient: Whether to remove classes with insufficient data
        use_excess_for: Where to allocate excess samples ("train", "val", or "none")
        
    Returns:
        Tuple of (train_set, val_set, class_names)
    """
    num_classes = len(dataset.class_names)
    classes = []
    # Collect indices per class
    class_indices = {cls_idx: [] for cls_idx in range(num_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for cls_idx, indices in class_indices.items():
        if hasattr(dataset, "normal_clip_label") and dataset.class_names[cls_idx] in [dataset.normal_clip_label]:
            train_count = len(indices) // 2
            val_count = len(indices) - train_count
            print(f"Splitting class {dataset.class_names[cls_idx]} in half!")
        elif len(indices) < (train_examples_per_class + val_examples_per_class):
            train_count = min(train_examples_per_class, len(indices))
            val_count = min(val_examples_per_class, len(indices) - train_count)
            print(f"Not enough data from class {dataset.class_names[cls_idx]} ({len(indices)} samples).")
            if remove_class_if_insufficient:
                print(f"Removing class {dataset.class_names[cls_idx]} from dataset.")
                continue
            else:
                print(f"Using {train_count} for training and {val_count} for validation.")
        elif len(indices) == (train_examples_per_class + val_examples_per_class) or use_excess_for == "none":
            train_count = train_examples_per_class
            val_count = val_examples_per_class
        elif len(indices) > (train_examples_per_class + val_examples_per_class):
            excess = len(indices) - (train_examples_per_class + val_examples_per_class)
            if use_excess_for == "train":
                train_count = train_examples_per_class + excess
                val_count = val_examples_per_class
            elif use_excess_for == "val":
                train_count = train_examples_per_class
                val_count = val_examples_per_class + excess
            else:
                train_count = train_examples_per_class
                val_count = val_examples_per_class
        
        classes.append(dataset.class_names[cls_idx])
        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count:train_count + val_count])

    if remove_class_if_insufficient:
        print(f"Final dataset has {len(classes)} classes and {len(train_indices)} train and {len(val_indices)} val samples after removing insufficient ones.")
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    return train_set, val_set, sorted(classes)


def build_balanced_subset(
    dataset: Dataset,
    examples_per_class: Optional[int] = 16,
    skip_first_K: int = 0,
) -> tuple[Dataset, list[str]]:
    """Build a balanced subset with fixed number of examples per class.
    
    Args:
        dataset: Base dataset
        examples_per_class: Number of examples per class (None for all)
        skip_first_K: Number of examples to skip from each class
        
    Returns:
        Tuple of (subset, class_names)
    """
    # Collect indices per class
    class_indices = {el: [] for el in dataset.class_to_idx.values()}
    for idx, (_, label) in enumerate(dataset.samples):
        if isinstance(label, str):
            label_idx = dataset.class_to_idx[label]
        else:
            label_idx = label
        class_indices[label_idx].append(idx)

    indices_list = []

    for cls_idx, indices in class_indices.items():
        # safe skip — max 0
        start = min(skip_first_K, len(indices))   # skip only if data exists

        remaining = len(indices) - start
        num_samples = min(examples_per_class, remaining) if examples_per_class is not None else remaining

        if examples_per_class is not None and remaining < examples_per_class:
            print(f"Only {remaining} usable samples in class {dataset.class_names[cls_idx]} after skip. Using {num_samples}")

        indices_list.extend(indices[start:start + num_samples])

    subset = torch.utils.data.Subset(dataset, indices_list)
    print(f"Built balanced subset with {len(subset)} samples and {len(dataset.class_names)} classes. Skipping {skip_first_K} samples per class when possible.")
    return subset, dataset.class_names


def build_train_and_val_set(args, config, processor):
    """Build training and validation datasets based on configuration.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        processor: CLIP processor
        
    Returns:
        Tuple of (train_set, val_set, train_classes, val_classes)
    """
    if config["dataset-config"]["has-validation"]:
        train_set = build_dataset(
            name=args.dataset,
            data_dir=args.data_dir,
            processor=processor,
            split='train',
            config=config,
            use_videomix=config["videomix-type"] is not None,
            use_augmentation=config["use-augmentation"]
        )
        train_classes = train_set.class_names
        if config["dataset-config"]["K-train"] is not None and args.dataset.lower() != "nwpu":
            # Use balanced subset of training set for validation if no val split
            train_set, train_classes = build_balanced_subset(
                dataset=train_set,
                examples_per_class=args.K_train,
                skip_first_K=config["dataset-config"]["skip-first-K-train"]
            )
        val_set = build_dataset(
            name=args.dataset,
            data_dir=args.data_dir,
            processor=processor,
            split='val',
            config=config,
            use_augmentation=False
       )
        val_classes = val_set.class_names
        if config["dataset-config"]["K-val"] is not None and args.dataset.lower() != "nwpu":
            val_set, val_classes = build_balanced_subset(
                dataset=val_set,
                examples_per_class=args.val_examples_per_class,
                skip_first_K=config["dataset-config"]["skip-first-K-val"]
            )    # Not K-shot, just build standard datasets, requires there to be a directory for each split        
    else:
        dataset = build_dataset(
            name=args.dataset,
            data_dir=args.data_dir,
            processor=processor,
            config=config,
            split='train',
        )
        if config["dataset-config"]["K-train"] is not None and config["dataset-config"]["K-val"] is not None:
            train_set, val_set, train_classes = build_K_shot_datasets(
                dataset=dataset,
                train_examples_per_class=args.K_train,
                val_examples_per_class=args.val_examples_per_class,
                remove_class_if_insufficient=config["dataset-config"]["remove-class-if-insufficient"],
                use_excess_for=config["dataset-config"]["use-excess-for"],
            )
        else:
            raise ValueError("Dataset has no validation split, but K-shot parameters not provided.")
        val_classes = train_classes  # For K-shot, validate on same classes as training, to be split later
        if config["videomix-type"] is not None:
            from .videomix import VideoMixDataset
            train_set = VideoMixDataset(base_dataset=train_set, config=config)
            print(f"Wrapped training set with VideoMix of type {config['videomix-type']}, with probability {config['videomix-prob']}.")

    if config["dataset-config"].get("class-weighting", None) is not None:
        from .weighting import put_class_weights_in_config
        put_class_weights_in_config(
            config=config,
            class_weighting=config["dataset-config"].get("class-weighting", None),
            samples=train_set.samples,
            class_idxs=range(len(train_set.class_names))
        )
    return train_set, val_set, train_classes, val_classes


def split_trainlist(src_path, train_out, val_out, val_ratio, seed=0):
    """Split a trainlist file into train and val files.
    
    Ensures no overlap of groups within classes (for HMDB51-style data).
    
    Args:
        src_path: Path to source trainlist file
        train_out: Output path for training list
        val_out: Output path for validation list
        val_ratio: Ratio of groups to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_out, val_out) paths
    """
    random.seed(seed)

    # class -> group_id -> lines
    by_class_group = defaultdict(lambda: defaultdict(list))

    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, cls = line.rsplit(" ", 1)
            group_id = path.split("_g")[1].split("_")[0]  # e.g. g16
            by_class_group[int(cls)][group_id].append(line)

    train_lines, val_lines = [], []

    for cls, groups in by_class_group.items():
        group_ids = list(groups.keys())
        random.shuffle(group_ids)

        n_val = max(1, int(len(group_ids) * val_ratio))
        val_groups = set(group_ids[:n_val])

        for gid, lines in groups.items():
            if gid in val_groups:
                val_lines.extend(lines)
            else:
                train_lines.extend(lines)

    with open(train_out, "w") as f:
        f.write("\n".join(train_lines))
    with open(val_out, "w") as f:
        f.write("\n".join(val_lines))

    return train_out, val_out
