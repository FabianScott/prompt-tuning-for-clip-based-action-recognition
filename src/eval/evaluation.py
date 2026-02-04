import torch
from time import time

from ..plots import plot_metrics_grid
from ..configs.parse_run_args import load_checkpoint_given_path
from ..configs.read_config import read_method_config
from ..modeling.utils import calculate_class_metrics
from .eval_utils import (
    setup_evaluation_environment,
    generate_test_path,
    load_config_and_checkpoint,
    prepare_method_config,
    build_evaluation_dataset,
    build_evaluation_dataloader,
    init_wandb_logger,
    save_metrics_json,
    load_or_compute_metrics
)


def run_evaluation(eval_config, meta_config, checkpoint_path: str = None):
    test_name, metrics_save_path = generate_test_path(eval_config)
    metrics_file = f"{metrics_save_path}/metrics_per_class.json"
    
    def compute_metrics():
        setup_evaluation_environment()
        print(f"Evaluating and saving metrics for {test_name} in {metrics_save_path}")
        
        # Load config and checkpoint
        if checkpoint_path is None:
            method_config, ckpt_path = load_config_and_checkpoint(eval_config)
        else:
            method_config, ckpt_path = load_config_and_checkpoint(eval_config)
            ckpt_path = checkpoint_path
        
        # Prepare method config
        base_config = read_method_config(eval_config["model_name"], use_augmentation=eval_config["use-augmentation"])
        method_config = prepare_method_config(method_config, eval_config, base_config)
        eval_config["train-config"] = method_config
        method_config["continue-from"] = ckpt_path
        
        # Build dataset and dataloader
        dataset, class_names = build_evaluation_dataset(eval_config, method_config)
        dataloader = build_evaluation_dataloader(dataset, meta_config)
        
        # Initialize wandb
        wandb_logger = init_wandb_logger(meta_config, method_config, "EVAL-" + test_name, ["eval"])
        
        # Load model
        model = load_checkpoint_given_path(
            method_name=eval_config["model_name"],
            checkpoint_path=ckpt_path,
            method_config=method_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            class_names=class_names,
            wandb_logger=wandb_logger,
        )
        
        # Evaluate
        acc, loss, logits_cat, labels_cat = model.eval(
            dataloader=dataloader,
            class_names=class_names,
            use_hand_crafted_text_features=eval_config["use-handcrafted-prompts"],
        )
        
        scores_per_class = calculate_class_metrics(
            logits_cat=logits_cat,
            labels_cat=labels_cat,
            class_names=class_names,
            logger=None,
            loss=loss
        )
        
        save_metrics_json(scores_per_class, metrics_file)
        return scores_per_class
    
    scores_per_class = load_or_compute_metrics(
        metrics_file,
        compute_metrics,
        force_rerun=meta_config.get("re-run", False)
    )

    t = time()
    plot_metrics_grid(
        scores_per_class,
        savename=test_name + f"{eval_config['run_id']}.png",
        bins=20,
    )
    print(f"Plotting took {time()-t:.2f} seconds")