import os
import torch
import wandb

from datetime import datetime
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, SubsetRandomSampler


from ..configs.parse_run_args import get_model_class, parse_run_args, load_checkpoint_given_path, get_config_and_path_from_keys, load_model_from_keys
from ..configs.read_token import read_token
from ..data.dataloading import build_dataloaders
from ..data.dataset_builders import build_train_and_val_set
from ..configs.paths import WANDB_CACHE_DIR, CLIP_MODEL_CACHE_DIR
from ..eval.evaluation import run_evaluation
from .utils import enable_profiling


def train_model(method_default: str, dataset_default: str, test_best: bool = True, fixed_config: dict = None):
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)
    args, method_config = parse_run_args(method_default=method_default, dataset_default=dataset_default, override_args=fixed_config)
    if args.continue_from is not None:
        loaded_config, checkpoint_path = get_config_and_path_from_keys(keys=args.continue_from)

    if method_config["use-profiling"]:
         enable_profiling()
    
    if "dataset-config" in fixed_config:
        method_config["dataset-config"].update(fixed_config["dataset-config"])

    torch.autograd.set_detect_anomaly(method_config["debug"])

    class_protocol = method_config['dataset-config']['class-protocol']
    extra_name_str = f"-{args.temporal_pooling}" if args.temporal_pooling != "mean" else ""
    extra_name_str += f"-{class_protocol}-{method_config['dataset-config']['num-classes']}" if class_protocol != "all" else ""
    run_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    run_folder = os.path.join(args.out, f"{args.model[-7:]}{extra_name_str}/K-{args.K_train}/", run_time_str)
    os.makedirs(run_folder, )
    print(f"Run will be saved at: {run_folder}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Fix tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR
    # Load a quick processor to build dataloaders and get classes
    # We will instantiate full trainer after we know class_names
    tmp_processor = CLIPProcessor.from_pretrained(args.model, cache_dir=CLIP_MODEL_CACHE_DIR, force_download=False, use_fast=True)

    train_set, val_set, train_classes, val_classes = build_train_and_val_set(args=args, config=method_config, processor=tmp_processor)

    train_loader, val_loader = build_dataloaders(
        train_set=train_set,
        val_set=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if method_config["start-train-step"] > 0:
        indices = list(range(method_config["start-train-step"] * method_config["batch-size"], len(train_set)))
        train_loader1 = DataLoader(train_set, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices), collate_fn=train_loader.collate_fn)
    else:
        train_loader1 = None

    if method_config["use-wandb"]:
        wandb.login(key=read_token('tokens/wandb.txt', 'Weights & Biases'))
    wandb_logger = wandb.init(
        project="Thesis - Video Classification", 
        config=method_config,
        name=f"{args.prompt_optimisation_method}-{args.dataset}-{args.model[-7:]}-K{args.K_train}{extra_name_str}",
        ) if method_config["use-wandb"] else None
    
    if args.continue_from is not None:
        model = load_checkpoint_given_path(
            method_name=args.prompt_optimisation_method.lower(),
            checkpoint_path=checkpoint_path,
            method_config=method_config,
            device=device,
            class_names=train_classes,
            wandb_logger=wandb_logger,
        )
    else:
        model_class = get_model_class(args.prompt_optimisation_method.lower())
        model = model_class(
            config=method_config,
            class_names=train_classes,
            device=device,
            eval_class_names=val_classes,
            wandb_logger=wandb_logger,
        )

    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        output_folder=run_folder,
        train_loader1=train_loader1
    )
    wandb.finish()

    if test_best:
        # Load the test set of the dataset if available
        print("Running evaluation on the best model saved during training...")
        try:
            run_evaluation(
                eval_config={
                    "model_name": args.prompt_optimisation_method.lower(),
                    "dataset_train": args.dataset,
                    "run_id": run_time_str.strip("/"),
                    "dataset_test": args.dataset,
                    "K-test": args.K_test,
                    "num-classes": method_config["dataset-config"]["num-classes"],
                    "class-protocol": method_config["dataset-config"]["class-protocol"],
                    "use-handcrafted-prompts": method_config.get("use-handcrafted-prompts", False),
                    "debug": method_config["debug"],
                    "split": "test",
                    "use-augmentation": False,
                },
                meta_config={
                    "batch-size": args.batch_size,
                    "num-workers": args.num_workers,
                    "use-wandb": method_config["use-wandb"],
                    "re-run": False
                },
                checkpoint_path=os.path.join(run_folder, "best.pt")
            )
        except Exception as e:
            print(f"Could not run evaluation on test set: {e}")

    return model