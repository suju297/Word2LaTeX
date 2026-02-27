"""Dataset Downloader - Download and convert public document layout datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import shutil

logger = logging.getLogger(__name__)

# Unified class mapping for all datasets
UNIFIED_CLASSES = {
    0: "Text",
    1: "Title", 
    2: "List",
    3: "Table",
    4: "Figure",
    5: "Caption",
    6: "Footer",
    7: "Header",
    8: "Formula",
}

# DocLayNet class mapping (11 classes -> unified)
DOCLAYNET_MAPPING = {
    1: 5,   # Caption -> Caption
    2: 6,   # Footnote -> Footer
    3: 8,   # Formula -> Formula
    4: 2,   # List-item -> List
    5: 6,   # Page-footer -> Footer
    6: 7,   # Page-header -> Header
    7: 4,   # Picture -> Figure
    8: 1,   # Section-header -> Title
    9: 3,   # Table -> Table
    10: 0,  # Text -> Text
    11: 1,  # Title -> Title
}

# DocBank class mapping (13 classes -> unified)
DOCBANK_MAPPING = {
    "abstract": 0,
    "author": 0,
    "caption": 5,
    "date": 0,
    "equation": 8,
    "figure": 4,
    "footer": 6,
    "list": 2,
    "paragraph": 0,
    "reference": 0,
    "section": 1,
    "table": 3,
    "title": 1,
}

# PubLayNet class mapping (5 classes -> unified)
# PubLayNet uses 1-indexed category IDs
PUBLAYNET_MAPPING = {
    1: 0,   # text -> Text
    2: 1,   # title -> Title
    3: 2,   # list -> List
    4: 3,   # table -> Table
    5: 4,   # figure -> Figure
}


class DatasetDownloader:
    """Download and convert public document layout datasets."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.labels_dir = output_dir / "labels"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {"images": 0, "annotations": 0}

    def download_doclaynet(self, split: str = "train", limit: Optional[int] = None) -> None:
        """Download DocLayNet COCO annotations and convert to YOLO format.
        
        Uses direct download from IBM cloud storage.
        
        Args:
            split: Dataset split (train, val, test)
            limit: Maximum number of samples to download (for testing)
        """
        import urllib.request
        import zipfile
        import tempfile
        
        try:
            from PIL import Image
            from tqdm import tqdm
        except ImportError:
            print("Required: pip install pillow tqdm")
            return
        
        # DocLayNet download URLs
        CORE_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
        
        # Check if already downloaded
        cache_dir = Path.home() / ".cache" / "doclaynet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = cache_dir / "DocLayNet_core.zip"
        extract_dir = cache_dir / "DocLayNet_core"
        
        if not extract_dir.exists():
            if not zip_path.exists():
                print(f"Downloading DocLayNet (~1.5GB)...")
                print(f"This may take a while. Saving to {zip_path}")
                
                # Download with progress
                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\rDownloading: {percent:.1f}% ({downloaded / 1e6:.1f}MB / {total_size / 1e6:.1f}MB)", end="")
                
                urllib.request.urlretrieve(CORE_URL, zip_path, download_progress)
                print("\nDownload complete!")
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)
        
        # Load COCO annotations - check multiple locations
        possible_dirs = [
            extract_dir / "DocLayNet_core",
            extract_dir,
            extract_dir.parent  # Check cache_dir root
        ]
        
        base_dir = None
        for d in possible_dirs:
            if (d / "COCO").exists():
                base_dir = d
                break
        
        if not base_dir:
            print(f"COCO annotations not found in {possible_dirs}")
            return

        coco_path = base_dir / "COCO" / f"{split}.json"
        png_dir = base_dir / "PNG"
        
        if not coco_path.exists():
            print(f"COCO annotations not found at {coco_path}")
            print(f"Available files: {list((extract_dir / 'COCO').glob('*'))}")
            return
        
        print(f"Loading COCO annotations from {coco_path}...")
        with open(coco_path) as f:
            coco = json.load(f)
        
        # Build image_id -> annotations mapping
        img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        images = coco["images"]
        if limit:
            images = images[:limit]
        
        print(f"Processing {len(images)} images...")
        
        for img_info in tqdm(images, desc=f"DocLayNet {split}"):
            img_id = img_info["id"]
            filename = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]
            
            # Copy image
            src_img = png_dir / filename
            if not src_img.exists():
                continue
            
            dst_filename = f"doclaynet_{split}_{img_id}.png"
            dst_img = self.images_dir / dst_filename
            shutil.copy(src_img, dst_img)
            
            # Convert annotations to YOLO format
            anns = img_to_anns.get(img_id, [])
            yolo_lines = []
            
            for ann in anns:
                cat_id = ann["category_id"]
                bbox = ann["bbox"]  # [x, y, width, height] in pixels
                
                unified_class = DOCLAYNET_MAPPING.get(cat_id)
                if unified_class is None:
                    continue
                
                x, y, w, h = bbox
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                
                yolo_lines.append(f"{unified_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                self.stats["annotations"] += 1
            
            # Save YOLO label
            label_path = self.labels_dir / f"doclaynet_{split}_{img_id}.txt"
            label_path.write_text("\n".join(yolo_lines))
            self.stats["images"] += 1
        
        print(f"DocLayNet {split}: {self.stats['images']} images, {self.stats['annotations']} annotations")

    def download_docbank(self, limit: Optional[int] = None) -> None:
        """Download DocBank from GitHub (token-level annotations).
        
        Note: DocBank has token-level bboxes, not region-level like DocLayNet.
        This method downloads a subset and aggregates tokens into region-level bboxes.
        
        DocBank download links from: https://doc-analysis.github.io/docbank-page/
        """
        try:
            from PIL import Image
            from tqdm import tqdm
        except ImportError:
            print("Required: pip install pillow tqdm")
            return
        
        cache_dir = Path.home() / ".cache" / "docbank"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check multiple possible locations
        possible_paths = [
            Path.home() / ".cache" / "docbank_repo" / "DocBank_samples" / "DocBank_samples",  # Cloned repo
            Path.home() / ".cache" / "docbank" / "DocBank_500K_ori_img",  # Manual download
            cache_dir / "DocBank_samples",  # Alternative location
        ]
        
        docbank_dir = None
        for path in possible_paths:
            if path.exists():
                docbank_dir = path
                print(f"Found DocBank at {docbank_dir}")
                break
        
        if docbank_dir:
            self._process_docbank_local(docbank_dir, limit)
        else:
            print("")
            print("⚠️  DocBank not found locally.")
            print("")
            print("To get DocBank samples, run:")
            print("  git clone git@github.com:doc-analysis/DocBank.git ~/.cache/docbank_repo")
            print("")
            print("For full dataset (30GB+):")
            print("  1. Visit: https://doc-analysis.github.io/docbank-page/")
            print("  2. Download DocBank_500K_ori_img.zip")
            print("  3. Extract to ~/.cache/docbank/")
            print("")
            return
    
    def _process_docbank_local(self, docbank_dir: Path, limit: Optional[int] = None) -> None:
        """Process locally downloaded DocBank data."""
        from PIL import Image
        from tqdm import tqdm
        
        # DocBank structure: one txt file per image
        txt_files = list(docbank_dir.glob("**/*.txt"))
        print(f"Found {len(txt_files)} annotation files")
        
        if limit:
            txt_files = txt_files[:limit]
        
        for txt_path in tqdm(txt_files, desc="DocBank"):
            try:
                # Find corresponding image (same name, .jpg or .png)
                img_path = txt_path.with_suffix(".jpg")
                if not img_path.exists():
                    img_path = txt_path.with_suffix(".png")
                if not img_path.exists():
                    continue
                
                # Load and parse DocBank annotation format
                # Format: token x0 y0 x1 y1 R G B font_name label
                tokens = []
                with open(txt_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 10:
                            token = parts[0]
                            bbox = [float(x) for x in parts[1:5]]
                            label = parts[-1]
                            tokens.append({"bbox": bbox, "label": label})
                
                if not tokens:
                    continue
                
                # Load image to get dimensions
                img = Image.open(img_path)
                width, height = img.size
                
                # Aggregate tokens into regions
                labels = [t["label"] for t in tokens]
                bboxes = [t["bbox"] for t in tokens]
                regions = self._aggregate_docbank_tokens(bboxes, labels, width, height)
                
                # Save image
                img_filename = f"docbank_{txt_path.stem}.jpg"
                dst_img = self.images_dir / img_filename
                img.save(str(dst_img), quality=95)
                
                # Save YOLO labels
                yolo_lines = []
                for region in regions:
                    label_name = region["label"]
                    unified_class = DOCBANK_MAPPING.get(label_name.lower())
                    if unified_class is None:
                        continue
                    xc, yc, w, h = region["bbox"]
                    yolo_lines.append(f"{unified_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    self.stats["annotations"] += 1
                
                label_path = self.labels_dir / f"docbank_{txt_path.stem}.txt"
                label_path.write_text("\n".join(yolo_lines))
                self.stats["images"] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {txt_path}: {e}")
        
        print(f"DocBank: {self.stats['images']} images, {self.stats['annotations']} annotations")
    
    def _aggregate_docbank_tokens(self, bboxes: List, labels: List, 
                                   width: int, height: int) -> List[Dict]:
        """Aggregate token-level bboxes into region-level bboxes.
        
        Groups consecutive tokens with the same label and merges their bboxes.
        """
        if not bboxes or not labels:
            return []
        
        # Map numeric labels to string labels if needed
        label_names = [
            "abstract", "author", "caption", "date", "equation", 
            "figure", "footer", "list", "paragraph", "reference",
            "section", "table", "title"
        ]
        
        regions = []
        current_label = None
        current_bbox = None  # [x0, y0, x1, y1]
        
        for bbox, label in zip(bboxes, labels):
            # Convert numeric label to string if needed
            if isinstance(label, int) and label < len(label_names):
                label_str = label_names[label]
            else:
                label_str = str(label).lower()
            
            if label_str == current_label and current_bbox is not None:
                # Extend current region
                x0, y0, x1, y1 = bbox
                current_bbox = [
                    min(current_bbox[0], x0),
                    min(current_bbox[1], y0),
                    max(current_bbox[2], x1),
                    max(current_bbox[3], y1)
                ]
            else:
                # Save previous region if exists
                if current_label is not None and current_bbox is not None:
                    # Convert to normalized center format
                    x0, y0, x1, y1 = current_bbox
                    w = (x1 - x0) / width
                    h = (y1 - y0) / height
                    xc = (x0 + x1) / 2 / width
                    yc = (y0 + y1) / 2 / height
                    
                    # Only add if reasonable size
                    if w > 0.01 and h > 0.005:
                        regions.append({
                            "label": current_label,
                            "bbox": [xc, yc, w, h]
                        })
                
                # Start new region
                current_label = label_str
                current_bbox = list(bbox)
        
        # Don't forget last region
        if current_label is not None and current_bbox is not None:
            x0, y0, x1, y1 = current_bbox
            w = (x1 - x0) / width
            h = (y1 - y0) / height
            xc = (x0 + x1) / 2 / width
            yc = (y0 + y1) / 2 / height
            
            if w > 0.01 and h > 0.005:
                regions.append({
                    "label": current_label,
                    "bbox": [xc, yc, w, h]
                })
        
        return regions

    def merge_with_synthetic(self, synthetic_dir: Path) -> None:
        """Merge external dataset with our synthetic Gemini-labeled data."""
        synthetic_images = synthetic_dir / "images"
        synthetic_labels = synthetic_dir / "labels"
        
        if not synthetic_images.exists():
            print(f"Synthetic data not found at {synthetic_dir}")
            return
        
        print(f"Merging synthetic data from {synthetic_dir}...")
        
        for img_path in synthetic_images.glob("*.jpg"):
            # Copy image
            dst_img = self.images_dir / f"synthetic_{img_path.name}"
            shutil.copy(img_path, dst_img)
            
            # Copy label
            label_path = synthetic_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = self.labels_dir / f"synthetic_{img_path.stem}.txt"
                shutil.copy(label_path, dst_label)
                self.stats["images"] += 1

    def download_publaynet(self, split: str = "train", limit: Optional[int] = None) -> None:
        """Download PubLayNet from HuggingFace and convert to YOLO format.
        
        PubLayNet has 5 classes: text, title, list, table, figure.
        Total ~360K images but loads efficiently via streaming.
        
        Args:
            split: Dataset split (train, val)
            limit: Maximum number of samples (0 = all)
        """
        try:
            from datasets import load_dataset
            from PIL import Image
            from tqdm import tqdm
        except ImportError:
            print("Required: pip install datasets pillow tqdm")
            return
        
        print(f"Loading PubLayNet {split} split from HuggingFace...")
        print("Note: PubLayNet has ~340K train / ~12K val images")
        
        try:
            # Stream to avoid downloading entire dataset at once
            dataset = load_dataset(
                "jordanparker6/publaynet", 
                split=split,
                streaming=True
            )
        except Exception as e:
            print(f"Failed to load PubLayNet: {e}")
            print("Trying alternative source...")
            try:
                dataset = load_dataset(
                    "DILHTWD/publaynet",
                    split=split,
                    streaming=True
                )
            except Exception as e2:
                print(f"PubLayNet not available: {e2}")
                return
        
        print(f"Processing PubLayNet {split}...")
        
        count = 0
        for sample in tqdm(dataset, desc=f"PubLayNet {split}"):
            if limit and count >= limit:
                break
            
            try:
                image = sample.get("image")
                # PubLayNet format: list of annotation dicts with bbox, category_id
                annotations = sample.get("annotations", [])
                
                if image is None or not annotations:
                    continue
                
                width, height = image.size
                
                # Convert to YOLO format
                yolo_lines = []
                for ann in annotations:
                    cat_id = ann.get("category_id")
                    bbox = ann.get("bbox")  # [x, y, width, height] in pixels
                    
                    if not bbox or cat_id is None:
                        continue
                    
                    unified_class = PUBLAYNET_MAPPING.get(cat_id)
                    if unified_class is None:
                        continue
                    
                    # PubLayNet bbox format: [x, y, width, height] in pixels
                    x, y, w, h = bbox
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    yolo_lines.append(f"{unified_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    self.stats["annotations"] += 1
                
                if not yolo_lines:
                    continue
                
                # Save image
                img_filename = f"publaynet_{split}_{count}.jpg"
                img_path = self.images_dir / img_filename
                image.save(str(img_path), quality=95)
                
                # Save YOLO label
                label_path = self.labels_dir / f"publaynet_{split}_{count}.txt"
                label_path.write_text("\n".join(yolo_lines))
                self.stats["images"] += 1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing PubLayNet sample {count}: {e}")
                continue
        
        print(f"PubLayNet {split}: {self.stats['images']} images, {self.stats['annotations']} annotations")

    def create_dataset_yaml(self, name: str = "document_layout") -> Path:
        """Create YOLO dataset.yaml configuration file."""
        yaml_content = f"""# Document Layout Detection Dataset
path: {self.output_dir.absolute()}
train: images
val: images

names:
  0: Text
  1: Title
  2: List
  3: Table
  4: Figure
  5: Caption
  6: Footer
  7: Header
  8: Formula
"""
        yaml_path = self.output_dir / f"{name}.yaml"
        yaml_path.write_text(yaml_content)
        print(f"Created dataset config: {yaml_path}")
        return yaml_path

    def print_stats(self) -> None:
        """Print dataset statistics."""
        print(f"\n=== Dataset Statistics ===")
        print(f"Total images: {self.stats['images']}")
        print(f"Total annotations: {self.stats['annotations']}")
        print(f"Output directory: {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare document layout datasets")
    parser.add_argument("--output", type=str, default="data/combined_dataset", help="Output directory")
    parser.add_argument("--doclaynet", action="store_true", help="Download DocLayNet")
    parser.add_argument("--docbank", action="store_true", help="Download DocBank")
    parser.add_argument("--publaynet", action="store_true", help="Download PubLayNet")
    parser.add_argument("--synthetic", type=str, help="Path to synthetic dataset to merge")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples per dataset (0=all)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(Path(args.output))
    
    if args.doclaynet:
        downloader.download_doclaynet(split=args.split, limit=args.limit or None)
    
    if args.docbank:
        downloader.download_docbank(limit=args.limit or None)
    
    if args.publaynet:
        downloader.download_publaynet(split=args.split, limit=args.limit or None)
    
    if args.synthetic:
        downloader.merge_with_synthetic(Path(args.synthetic))
    
    downloader.create_dataset_yaml()
    downloader.print_stats()


if __name__ == "__main__":
    main()
