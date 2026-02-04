import os
import wandb
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Callable
from transformers import CLIPProcessor


def test_dataloader(
        method_default: str, 
        dataset_default: str, 
        train_first: bool = True, 
        fixed_config: Optional[dict] = None, 
        break_after: Optional[int] = None,
        idx_to_plot: Optional[int] = 0,
        ):
    args, method_config = parse_run_args(method_default=method_default, dataset_default=dataset_default, override_args=fixed_config)
    run_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    run_folder = os.path.join(args.out, f"testing-dataloader/{args.model[-7:]}/K-{args.K_train}/", run_time_str)
    os.makedirs(run_folder, exist_ok=False)
    print(f"Run will be saved at: {run_folder}")
    # Fix tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Load a quick processor to build dataloaders and get classes
    # We will instantiate full trainer after we know class_names
    tmp_processor = CLIPProcessor.from_pretrained(args.model, cache_dir=CLIP_MODEL_CACHE_DIR)

    train_set, val_set, train_classes, val_classes = build_train_and_val_set(args=args, config=method_config, processor=tmp_processor)
    try:
        test_set = build_dataset(
                name=dataset_default,
                split="test",
                processor=tmp_processor,
                config=method_config
            )
    except Exception as e:
        print(f"Test set could not be built: {e}")
        test_set = None

    train_loader, val_loader = build_dataloaders(
        train_set=train_set,
        val_set=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False
    )
    if train_first:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train loader")
        for i, batch in pbar:
            if batch is None:
                print(f"Batch {i} is None")
                continue
            if break_after is not None and i >= break_after:
                break
            vids, mask, labels = batch
            print(f"Batch {i}: vids shape {vids.shape}, mask shape {mask.shape if mask is not None else None}, labels shape {labels.shape}")
            if idx_to_plot is not None and i == idx_to_plot:
                plot_video_examples(
                    examples=vids,
                    labels=labels,
                    class_names=train_classes,
                    nrow=4,
                    save_path=os.path.join(run_folder, "train_batch_examples.png"),
                    show=False,
                    title="Training Batch Examples"
                )
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation loader")
        for i, batch in pbar:
            vids, mask, labels = batch
            if break_after is not None and i >= break_after:
                break
            print(f"Batch {i}: vids shape {vids.shape}, mask shape {mask.shape if mask is not None else None}, labels shape {labels.shape}")
            if idx_to_plot is not None and i == idx_to_plot:
                plot_video_examples(
                    examples=vids,
                    labels=labels,
                    class_names=val_classes,
                    nrow=4,
                    save_path=os.path.join(run_folder, "val_batch_examples.png"),
                    show=False,
                    title="Validation Batch Examples"
                )
    else:
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation loader")
        for i, batch in pbar:
            if batch is None:
                print(f"Batch {i} is None")
                continue
            vids, mask, labels = batch
            print(f"Batch {i}: vids shape {vids.shape}, mask shape {mask.shape if mask is not None else None}, labels shape {labels.shape}")
            if break_after is not None and i >= break_after:
                break
            if idx_to_plot is not None and i == idx_to_plot:
                plot_video_examples(
                    examples=vids,
                    labels=labels,
                    class_names=val_classes,
                    nrow=4,
                    save_path=os.path.join(run_folder, "val_batch_examples.png"),
                    show=False,
                    title="Validation Batch Examples"
                )
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train loader")
        for i, batch in pbar:
            if batch is None:   
                print(f"Batch {i} is None")
                continue
            vids, mask, labels = batch
            print(f"Batch {i}: vids shape {vids.shape}, mask shape {mask.shape if mask is not None else None}, labels shape {labels.shape}")
            if break_after is not None and i >= break_after:
                break
            if idx_to_plot is not None and i == idx_to_plot:
                plot_video_examples(
                    examples=vids,
                    labels=labels,
                    class_names=train_classes,
                    nrow=4,
                    save_path=os.path.join(run_folder, "train_batch_examples.png"),
                    show=False,
                    title="Training Batch Examples"
                )
    if test_set is not None:
        test_loader = build_dataloader(
            dataset=test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test loader")
        for i, batch in pbar:
            if batch is None:   
                print(f"Batch {i} is None")
                continue
            vids, mask, labels = batch
            print(f"Batch {i}: vids shape {vids.shape}, mask shape {mask.shape if mask is not None else None}, labels shape {labels.shape}")
            if break_after is not None and i >= break_after:
                break
            if idx_to_plot is not None and i == idx_to_plot:
                plot_video_examples(
                    examples=vids,
                    labels=labels,
                    class_names=test_set.class_names,
                    nrow=4,
                    save_path=os.path.join(run_folder, "test_batch_examples.png"),
                    show=False,
                    title="Test Batch Examples"
                )

if __name__ == "__main__":
    import sys
    import torchvision
    from pathlib import Path
    
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))
    print(f"Project root: {project_root}")
    from src.configs.parse_run_args import get_model_class, parse_run_args, load_checkpoint_given_trainer
    from src.data.dataset_builders import build_train_and_val_set, build_dataset
    from src.data.dataloading import build_dataloaders, build_dataloader
    from src.configs.paths import CLIP_MODEL_CACHE_DIR
    from src.plots import plot_video_examples

    fixed_config = {
        "use-augmentation": False,
        "dataset-config": {
            "batch-size": 16,
            "num-workers": 0,
            "K-train": None,
            "K-val": 4,
        },
        "videomix-type": None # "spatial"
    }
    break_after = 1

    test_dataloader(
        method_default="video_coop", 
        dataset_default="ucf101", 
        train_first=True, 
        fixed_config=fixed_config,
        break_after=break_after,
        idx_to_plot=0,
        )
    
    # Plain : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_13-57-43/train_batch_examples.png
    # Flip : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_13-59-12/train_batch_examples.png
    # Color Jitter : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_14-00-43/train_batch_examples.png
    # Gaussian Blur : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_14-03-14/train_batch_examples.png
    # Salt and Pepper : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_14-06-34/train_batch_examples.png
    # Cutout : /work3/fasco/data/processed/prompts/video_coop/ucf101/testing-dataloader/patch16/K-None/2026-01-20_14-08-04/train_batch_examples.png