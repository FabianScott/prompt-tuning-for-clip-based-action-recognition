
import torch 
def clean_run_name_for_keys(name):
    to_remove = ["ucf101", "video", "/", "best", "true", "val", "ep", "0", "16", "No Hand Features"] + dataset_names
    to_replace = [("_", " "), ("-", " "), ("With Hand Features", "H"),]

    for item in to_remove:
        name = name.replace(item, "")
    for old, new in to_replace:
        name = name.replace(old, new)
    return name.title().strip()

def flop_analysis(
        model_keys, 
        name_for_display,
        dataset_name, 
        plot_savename: str, 
        num_classes=None,
        device=torch.device("cpu")
        ):
    
    plot_title = f"GFLOPs for {name_for_display} vs Validation Accuracy on UCF101"
    json_savename = plot_savename.replace(".png", ".json")

    model_accuracies = []
    gflops_list = []
    keys_used = []
    dataset, dataloader, idx = None, None, None
    for keys in tqdm(model_keys, desc="Calculating GFLOPs for models"):
        try:
            method_config, checkpoint_path = get_config_and_path_from_keys(keys=keys)
        except Exception as e:
            print(f"Error loading model for keys {keys}: {e}")
            continue
        method_name = keys.split("/")[0]

        dataset = build_dataset(
            name=dataset_name,
            split="test",
            config=method_config,
            processor = CLIPProcessor.from_pretrained(method_config["model"], cache_dir=CLIP_MODEL_CACHE_DIR, force_download=False, use_fast=True)
        ) if dataset is None else dataset

        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        ) if dataloader is None else dataloader
        model = load_checkpoint_given_path(
            method_name=method_name,
            checkpoint_path=checkpoint_path,
            method_config=method_config,
            device=device,
            class_names=dataset.class_names if num_classes is None else ["cls"] * num_classes,
            wandb_logger=None,
        )
        model.eval_mode()
        model = CoOpForwardWrapper(model).to(device)
        inputs = None # Get videos and video_masks, no labels
        idx = 0 if idx is None else idx
        while inputs is None:
            try:
                inputs = next(iter(dataloader))[:2]
            except Exception as e:
                print(f"Error getting batch {idx} from dataloader: {e}")
                idx += 1
        with torch.no_grad(): 
            flops_dict, skips = flop_count(
                model,
                inputs=inputs,
                supported_ops=supported_ops
            )

        gflops = sum(flops_dict.values())
        acc = max(method_config["training-history"]["val-acc"])
        model_accuracies.append(acc)
        gflops_list.append(gflops)
        keys_used.append(keys)
        print(f"Model {keys} with acc {acc:.2f} has {gflops:.2f} GFLOPs, skipped : {skips}")

    model_list = [
        {"name": clean_run_name_for_keys(key), "gflops": g, "accuracy": a}
        for key, g, a in zip(keys_used, gflops_list, model_accuracies)
    ]

    with open(json_savename, "w") as f:
        json.dump(model_list, f, indent=4)
        print(f"Saved GFLOPs data to {json_savename}")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_gflops_vs_accuracy(
        models=model_list,
        ax=ax,
        alpha=0.9,
        linewidth=2.0,
        fontsize=14,
        grid=True,
        cmap="tab10",
        ylim=(0.7, 1.01),
        xlim=(1700, max(3000, max(gflops_list)+100))
    )
    plt.title(plot_title, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_savename), exist_ok=True)
    plt.savefig(plot_savename, dpi=400)
    print(f"Saved GFLOPs vs Accuracy plot to {plot_savename}")
    
    return model_list, plot_savename, json_savename


if __name__ == "__main__":
    import os
    import sys
    import json
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    from src.configs.parse_run_args import get_config_and_path_from_keys, load_checkpoint_given_path
    from src.data.dataset_builders import build_dataset
    from src.data.dataloading import build_dataloader
    from src.modeling.flops import supported_ops
    from src.configs.datasets_and_methods import method_names, dataset_names
    from fvcore.nn import flop_count
    from transformers import CLIPProcessor
    from src.modeling.flops import CoOpForwardWrapper

    from src.configs.paths import CLIP_MODEL_CACHE_DIR
    from src.plots import plot_gflops_vs_accuracy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_range = [1, 51, 101, 400, 600, 1000]
    model_keys = [
        "vidop/ucf101/best-with-hand-features-true-val",
        "vidop/ucf101/best-no-hand-features-attention-0",
        "video_coop/ucf101/best-mean",
        "video_coop/ucf101/best-attention-0",
        "dual_coop/ucf101/best-mean",
        "dual_coop/ucf101/best-attention-0",
        "vilt/ucf101/best-mean",
        "vita/ucf101/ep-16",
        "stt/ucf101/best-true-val",
    ]  
    rerun = True

    output_lists = []
    dataset_name = "ucf101"
    for num_classes in tqdm(classes_range, desc="Analyzing GFLOPs across class counts"):
        plot_savename = f"figures/gflops/gflops_vs_val_accuracy_{num_classes}_classes.png"
        json_savename = plot_savename.replace(".png", ".json")
        if os.path.exists(json_savename) and not rerun:
            print(f"Loading existing GFLOPs data from {json_savename}")
            with open(json_savename, "r") as f:
                model_list = json.load(f)
            output_lists.append(model_list)
            continue
        model_list, _, json_savename = flop_analysis(
            model_keys=model_keys,
            dataset_name=dataset_name,
            num_classes=num_classes,
            name_for_display=f"{num_classes} class" + ("es" if num_classes > 1 else ""),
            plot_savename=plot_savename,
            device=device,
        )
        output_lists.append(model_list)

    # Make a line-plot of the gflops vs num_classes for each model
    model_names = [model["name"] for model in output_lists[0]]
    plt.figure(figsize=(8, 4))
    # Plot GFLOPs vs number of classes for each model
    for model_name in model_names:
        gflops_across_classes = []
        for model_list in output_lists:
            # Find the model in this list with the matching name
            match = next((m for m in model_list if m["name"] == model_name), None)
            gflops_across_classes.append(match["gflops"] if match else None)
        
        # Linear fit
        x = np.array(classes_range)
        y = np.array(gflops_across_classes)
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        print(f"{model_name}: GFLOPs = {slope:.2f} * NumClasses + {intercept:.2f}")

        plt.plot(classes_range, gflops_across_classes, marker='o', label=f"{model_name}: {slope:.2f}•N + {intercept:.0f}", alpha=0.8, linewidth=2.0)
    
    # Compute max GFLOPs at each dataset class point
    dataset_points = [51, 101, 400]
    max_gflops_at_points = []
    for c in dataset_points:
        # Find index in classes_range closest to c
        idx = min(range(len(classes_range)), key=lambda i: abs(classes_range[i]-c))
        max_gflops = max(output_lists[i][j]["gflops"] 
                        for j in range(len(model_names)) 
                        for i, cls in enumerate(classes_range) if i == idx)
        max_gflops_at_points.append(max_gflops)

    # Draw vertical lines
    for c, y_max in zip(dataset_points, max_gflops_at_points):
        plt.vlines(c, ymin=0, ymax=y_max, label=None, colors='gray', linestyles='dashed', alpha=0.5)
    
    # Regular ticks every 100 except 100 and 400
    regular_ticks = [i for i in range(0, max(classes_range)+1, 100) if i not in [100, 400]]
    # Add dataset points
    tick_positions = regular_ticks + [51, 101, 400] 
    tick_positions = sorted(tick_positions)

    # Set labels: default numbers for regular ticks, dataset names for specific points
    tick_labels = []
    for pos in tick_positions:
        # if pos == 51:
        #     tick_labels.append("HMDB51")
        # elif pos == 101:
        #     tick_labels.append("UCF101")
        # elif pos == 400:
        #     tick_labels.append("Kinetics400")
        # else:
        tick_labels.append(str(pos))
    
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.xlabel("Number of Classes")
    plt.ylabel("GFLOPs")
    plt.title("GFLOPs vs Number of Classes")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    plt.tight_layout()
    plotname = "figures/gflops/gflops_vs_num_classes.png"
    os.makedirs(os.path.dirname(plotname), exist_ok=True)
    plt.savefig(plotname, dpi=400)
    print(f"Saved GFLOPs vs Number of Classes plot to {plotname}")
    plt.show()
