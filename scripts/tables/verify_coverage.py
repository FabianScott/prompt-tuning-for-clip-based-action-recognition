#!/usr/bin/env python3
"""
Verification script to ensure all LaTeX tables have corresponding generation scripts.
"""

# Mapping of LaTeX table labels to their generation scripts
TABLE_MAPPING = {
    "tab:ucf101_results": "ucf101_main_results.py",
    "tab:ucf101_results_videomix": "ucf101_videomix_results.py",
    "tab:ucf101_1_temporal_view": "1_temporal_view.py",
    "tab:classes_under_threshold_per_model": "distribution_tables.py",
    "tab:overlap_classes_by_threshold": "distribution_tables.py",
    "tab:ucf101_augmentation": "ucf101_combined_augmentation.py",
    "tab:ucf101_results_augmented_train": "train_augmented.py",
    "tab:ucf101_augmentations_separate_1": "individual_augmentation.py",
    "tab:ucf101_augmentations_separate_2": "individual_augmentation.py",
    "tab:kinetics400_zero-shot": "kinetics400_zero_shot.py",
    "tab:kinetics_kshot": "kinetics400_kshot.py",
    "tab:hmdb51_zero-shot": "hmdb51_zero_shot.py",
    "tab:hmdb51_kshot": "hmdb51_kshot.py",
    "tab:ucf101_removed_black_strips": "removed_black_strip.py",
}

if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("TABLE GENERATION SCRIPT VERIFICATION")
    print("=" * 80)
    print()
    
    tables_dir = os.path.join(os.path.dirname(__file__))
    
    print(f"Checking {len(TABLE_MAPPING)} tables...")
    print()
    
    all_exist = True
    for table_label, script_name in sorted(TABLE_MAPPING.items()):
        script_path = os.path.join(tables_dir, script_name)
        exists = os.path.exists(script_path)
        status = "✓" if exists else "✗"
        
        if not exists:
            all_exist = False
            
        print(f"{status} {table_label:45s} -> {script_name}")
    
    print()
    print("=" * 80)
    if all_exist:
        print("✓ All tables have generation scripts!")
    else:
        print("✗ Some scripts are missing!")
    print("=" * 80)
