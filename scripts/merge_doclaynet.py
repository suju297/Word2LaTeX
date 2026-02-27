import os
import shutil
import random
from pathlib import Path

def merge_doclaynet():
    # Setup paths
    base_dir = Path("data")
    src_dir = base_dir / "doclaynet_dataset"
    dst_dir = base_dir / "merged_dataset"
    
    if not src_dir.exists():
        print(f"Source not found: {src_dir}")
        return

    # Get all images
    # DocLayNet images are .png
    images = list((src_dir / "images").glob("*.png"))
    print(f"Found {len(images)} DocLayNet images")
    
    if not images:
        return

    # Shuffle
    random.shuffle(images)
    
    # Split
    total = len(images)
    val_count = int(total * 0.1)  # 10% val
    train_count = total - val_count
    
    train_imgs = images[:train_count]
    val_imgs = images[train_count:]
    
    print(f"Merging {len(train_imgs)} to train, {len(val_imgs)} to val...")
    
    # Copy helper
    def copy_files(img_list, split_name):
        count = 0
        for img_path in img_list:
            # Image
            dst_img_path = dst_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # Label (change ext to .txt)
            label_name = img_path.stem + ".txt"
            src_label = src_dir / "labels" / label_name
            if src_label.exists():
                dst_label_path = dst_dir / split_name / "labels" / label_name
                shutil.copy2(src_label, dst_label_path)
                count += 1
        return count

    # Do the copy
    n_train = copy_files(train_imgs, "train")
    n_val = copy_files(val_imgs, "val")
    
    print(f"Copied {n_train} train labels, {n_val} val labels")

if __name__ == "__main__":
    merge_doclaynet()
