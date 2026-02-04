# Adapted from https://github.com/cvdfoundation/kinetics-dataset/blob/main/arrange_by_classes.py

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# KINETICS-400
"""     videos      csv         replace
test:   38685       39805       0
train:  240258      246534      1392
val:    19881       19906       0
"""

SPLITS = ['test', 'train', 'val']

def load_label(csv):
    table = np.loadtxt(csv, skiprows=1, dtype=str, delimiter=',')
    return {k: v.replace('"', '') for k, v in zip(table[:, 1], table[:, 0])}

def collect_dict(path, split, replace_videos):
    split_video_path = path / split
    split_csv = load_label(path / f'annotations/{split}.csv')
    split_videos = list(split_video_path.glob('*.mp4'))
    split_videos = {str(p.stem)[:11]:p for p in split_videos}
    # replace paths for corrupted videos
    match_dict = {k: replace_videos[k] for k in split_videos.keys() & replace_videos.keys()}
    split_videos.update(match_dict)
    # collect videos with labels from csv: dict with {video_path: class}
    split_final = {split_videos[k]:split_csv[k] for k in split_csv.keys() & split_videos.keys()}
    return split_final

def main():
    from ...configs.paths import kinetics400_root
    path = Path(kinetics400_root)
    assert path.exists(), f'Provided path:{path} does not exist'
    from ...configs.paths import kinetics400_pytorch_root
    # collect videos in replacement
    replace = list((path / 'replacement/replacement_for_corrupted_k400').glob('*.mp4'))
    replace_videos = {str(p.stem)[:11]:p for p in replace}

    video_parent = Path(kinetics400_pytorch_root)

    for split in SPLITS:
        print(f'Working on: {split}')
        # create output path
        split_video_path = video_parent / split
        split_video_path.mkdir(exist_ok=True, parents=True)
        split_final = collect_dict(path, split, replace_videos)
        print(f'Found {len(split_final)} videos in split: {split}')
        labels = set(split_final.values())
        # create label directories 
        for label in labels:
            label_pth = split_video_path / label
            label_pth.mkdir(exist_ok=True, parents=True)
        # symlink videos to respective labels 
        for vid_pth, label in tqdm(split_final.items(), desc=f'Progress {split}'):
            dst_vid = split_video_path / label / vid_pth.name
            if dst_vid.is_symlink():
                dst_vid.unlink()
            dst_vid.symlink_to(vid_pth.resolve(), target_is_directory=False)

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))
    print(f"Project root: {project_root}")
    main()