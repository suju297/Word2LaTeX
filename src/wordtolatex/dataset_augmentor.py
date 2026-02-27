"""Dataset Augmentation - Expand synthetic dataset with transformations."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

class DatasetAugmentor:
    """Augment YOLO/COCO dataset with image and bbox transformations."""
    
    def __init__(self, 
                 input_dir: Path, 
                 output_dir: Path,
                 augmentations_per_image: int = 3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentations_per_image = augmentations_per_image
        
        self.images_in = input_dir / "images"
        self.labels_in = input_dir / "labels"
        self.images_out = output_dir / "images"
        self.labels_out = output_dir / "labels"
        
        # Create output dirs
        self.images_out.mkdir(parents=True, exist_ok=True)
        self.labels_out.mkdir(parents=True, exist_ok=True)

    def augment(self) -> None:
        """Run augmentation pipeline."""
        image_files = list(self.images_in.glob("*.jpg")) + list(self.images_in.glob("*.png"))
        print(f"Found {len(image_files)} images to augment")
        
        # Copy originals first
        for img_path in image_files:
            # Copy image
            img = Image.open(img_path)
            img.save(self.images_out / img_path.name)
            
            # Copy label
            label_path = self.labels_in / f"{img_path.stem}.txt"
            if label_path.exists():
                (self.labels_out / label_path.name).write_text(label_path.read_text())
        
        # Generate augmentations
        for img_path in tqdm(image_files, desc="Augmenting"):
            label_path = self.labels_in / f"{img_path.stem}.txt"
            bboxes = self._load_yolo_labels(label_path) if label_path.exists() else []
            
            img = Image.open(img_path)
            
            for aug_idx in range(self.augmentations_per_image):
                aug_img, aug_bboxes = self._apply_augmentations(img, bboxes)
                
                # Save with augmentation suffix
                aug_name = f"{img_path.stem}_aug{aug_idx}"
                aug_img.save(self.images_out / f"{aug_name}.jpg", quality=95)
                self._save_yolo_labels(self.labels_out / f"{aug_name}.txt", aug_bboxes)
        
        print(f"Augmentation complete. Output: {self.output_dir}")
        print(f"Total images: {len(image_files) * (1 + self.augmentations_per_image)}")

    def _load_yolo_labels(self, path: Path) -> List[Dict]:
        """Load YOLO format labels."""
        bboxes = []
        if not path.exists():
            return bboxes
            
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                bboxes.append({
                    "class_id": int(parts[0]),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4])
                })
        return bboxes

    def _save_yolo_labels(self, path: Path, bboxes: List[Dict]) -> None:
        """Save YOLO format labels."""
        lines = []
        for bbox in bboxes:
            # Validate bbox is within bounds
            xc, yc, w, h = bbox["x_center"], bbox["y_center"], bbox["width"], bbox["height"]
            
            # Clamp to [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))
            
            # Skip if bbox is mostly outside image
            if xc - w/2 > 1.0 or xc + w/2 < 0.0:
                continue
            if yc - h/2 > 1.0 or yc + h/2 < 0.0:
                continue
            
            lines.append(f"{bbox['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        path.write_text("\n".join(lines))

    def _apply_augmentations(self, img: Image.Image, bboxes: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """Apply augmentations appropriate for clean digital documents (Word→PDF).
        
        Since source documents are digitally rendered (not scanned), we focus on:
        - Scale/zoom variations
        - Minor color jitter (display differences)
        - Cutout/occlusion (robustness)
        - Aspect ratio variations
        """
        aug_img = img.copy()
        aug_bboxes = copy.deepcopy(bboxes)
        
        # === PHOTOMETRIC (minor - documents are clean) ===
        
        # 1. Minor brightness variation (display calibration differences)
        if random.random() < 0.4:
            factor = random.uniform(0.95, 1.05)  # Very subtle
            aug_img = ImageEnhance.Brightness(aug_img).enhance(factor)
        
        # 2. Minor contrast variation
        if random.random() < 0.4:
            factor = random.uniform(0.95, 1.05)
            aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)
        
        # 3. Background color shift (slightly off-white paper color)
        if random.random() < 0.3:
            aug_img = self._apply_background_tint(aug_img)
        
        # === GEOMETRIC ===
        
        # 4. Scale/zoom variation (simulate different view sizes)
        if random.random() < 0.4:
            scale = random.uniform(0.85, 1.15)
            aug_img, aug_bboxes = self._apply_scale(aug_img, aug_bboxes, scale)
        
        # 5. Random cutout/occlusion (improves robustness)
        if random.random() < 0.3:
            aug_img = self._apply_cutout(aug_img, num_holes=random.randint(1, 3))
        
        # 6. Slight aspect ratio change (different paper proportions)
        if random.random() < 0.2:
            aug_img, aug_bboxes = self._apply_aspect_ratio_change(aug_img, aug_bboxes)
        
        # 7. Random edge padding (margin variations)
        if random.random() < 0.25:
            aug_img, aug_bboxes = self._apply_margin_variation(aug_img, aug_bboxes)
        
        return aug_img, aug_bboxes

    def _apply_background_tint(self, img: Image.Image) -> Image.Image:
        """Apply slight background tint (cream, gray, blue-white paper colors)."""
        img_array = np.array(img).astype(np.float32)
        
        # Various paper tints
        tints = [
            [255, 255, 250],  # Slight cream
            [252, 252, 255],  # Slight blue-white
            [250, 250, 250],  # Off-white
            [255, 253, 248],  # Warm white
        ]
        tint = np.array(random.choice(tints), dtype=np.float32)
        blend = random.uniform(0.01, 0.04)  # Very subtle
        
        # Only apply to light pixels (background, not text)
        light_mask = img_array.mean(axis=2) > 200
        
        for c in range(3):
            img_array[:, :, c] = np.where(
                light_mask,
                img_array[:, :, c] * (1 - blend) + tint[c] * blend,
                img_array[:, :, c]
            )
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _apply_scale(self, img: Image.Image, bboxes: List[Dict], scale: float) -> Tuple[Image.Image, List[Dict]]:
        """Apply zoom in/out while keeping image size constant."""
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create canvas and center
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        
        if scale > 1:
            # Crop center of scaled image
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            canvas = scaled.crop((left, top, left + w, top + h))
            
            # Adjust bboxes (shift towards center)
            adjusted = []
            for bbox in bboxes:
                new_bbox = {
                    "class_id": bbox["class_id"],
                    "x_center": (bbox["x_center"] - 0.5) * scale + 0.5,
                    "y_center": (bbox["y_center"] - 0.5) * scale + 0.5,
                    "width": bbox["width"] * scale,
                    "height": bbox["height"] * scale
                }
                adjusted.append(new_bbox)
        else:
            # Paste scaled image on white canvas
            canvas.paste(scaled, (offset_x, offset_y))
            
            # Adjust bboxes
            adjusted = []
            for bbox in bboxes:
                new_bbox = {
                    "class_id": bbox["class_id"],
                    "x_center": offset_x / w + bbox["x_center"] * scale,
                    "y_center": offset_y / h + bbox["y_center"] * scale,
                    "width": bbox["width"] * scale,
                    "height": bbox["height"] * scale
                }
                adjusted.append(new_bbox)
        
        return canvas, adjusted

    def _apply_cutout(self, img: Image.Image, num_holes: int = 2) -> Image.Image:
        """Apply random rectangular cutouts (occlusion augmentation)."""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        for _ in range(num_holes):
            # Random hole size (2-8% of image)
            hole_h = random.randint(int(h * 0.02), int(h * 0.08))
            hole_w = random.randint(int(w * 0.02), int(w * 0.08))
            
            # Random position
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            
            # Fill with white (document background)
            img_array[y:y+hole_h, x:x+hole_w] = 255
        
        return Image.fromarray(img_array)

    def _apply_aspect_ratio_change(self, img: Image.Image, bboxes: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """Apply slight aspect ratio distortion."""
        w, h = img.size
        
        # ±5% stretch in one dimension
        stretch = random.uniform(0.95, 1.05)
        if random.random() < 0.5:
            new_w = int(w * stretch)
            new_h = h
        else:
            new_w = w
            new_h = int(h * stretch)
        
        stretched = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Resize back to original
        final = stretched.resize((w, h), Image.LANCZOS)
        
        # Bboxes stay in normalized coords, so no adjustment needed after resize back
        return final, bboxes

    def _apply_margin_variation(self, img: Image.Image, bboxes: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        """Add random padding/margins (simulates different document margins)."""
        w, h = img.size
        
        # Add 2-5% padding on random sides
        pad = random.uniform(0.02, 0.05)
        pad_px = int(min(w, h) * pad)
        
        sides = random.sample(["top", "bottom", "left", "right"], k=random.randint(1, 2))
        
        new_h = h + (pad_px if "top" in sides else 0) + (pad_px if "bottom" in sides else 0)
        new_w = w + (pad_px if "left" in sides else 0) + (pad_px if "right" in sides else 0)
        
        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        
        offset_x = pad_px if "left" in sides else 0
        offset_y = pad_px if "top" in sides else 0
        
        canvas.paste(img, (offset_x, offset_y))
        
        # Resize back to original size
        final = canvas.resize((w, h), Image.LANCZOS)
        
        # Adjust bboxes
        scale_x = w / new_w
        scale_y = h / new_h
        
        adjusted = []
        for bbox in bboxes:
            new_bbox = {
                "class_id": bbox["class_id"],
                "x_center": (offset_x / new_w + bbox["x_center"] * (w / new_w)),
                "y_center": (offset_y / new_h + bbox["y_center"] * (h / new_h)),
                "width": bbox["width"] * scale_x,
                "height": bbox["height"] * scale_y
            }
            adjusted.append(new_bbox)
        
        return final, adjusted

    def _apply_paper_aging(self, img: Image.Image) -> Image.Image:
        """Add slight yellow/sepia tint to simulate aged paper."""
        img_array = np.array(img).astype(np.float32)
        
        # Blend towards warm color
        tint = np.array([255, 250, 235], dtype=np.float32)  # Warm white
        blend = random.uniform(0.02, 0.08)
        
        img_array = img_array * (1 - blend) + tint * blend
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _apply_salt_pepper_noise(self, img: Image.Image, amount: float = 0.002) -> Image.Image:
        """Add salt-and-pepper noise (common in scanned docs)."""
        img_array = np.array(img)
        
        # Salt (white pixels)
        salt_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[salt_mask] = 255
        
        # Pepper (black pixels)  
        pepper_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[pepper_mask] = 0
        
        return Image.fromarray(img_array)

    def _apply_shadow_gradient(self, img: Image.Image) -> Image.Image:
        """Apply diagonal shadow gradient (simulates uneven lighting)."""
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        
        # Create gradient mask
        direction = random.choice(["top", "bottom", "left", "right", "corner"])
        
        if direction == "top":
            gradient = np.linspace(0.75, 1.0, h).reshape(-1, 1)
        elif direction == "bottom":
            gradient = np.linspace(1.0, 0.75, h).reshape(-1, 1)
        elif direction == "left":
            gradient = np.linspace(0.75, 1.0, w).reshape(1, -1)
        elif direction == "right":
            gradient = np.linspace(1.0, 0.75, w).reshape(1, -1)
        else:  # corner
            y_grad = np.linspace(0.85, 1.0, h).reshape(-1, 1)
            x_grad = np.linspace(0.85, 1.0, w).reshape(1, -1)
            gradient = y_grad * x_grad
        
        # Expand gradient to match image shape (h, w, 3)
        if gradient.shape == (h, 1):
            # Vertical gradient: broadcast across width and channels
            gradient = np.tile(gradient, (1, w))[:, :, np.newaxis]
        elif gradient.shape == (1, w):
            # Horizontal gradient: broadcast across height and channels
            gradient = np.tile(gradient, (h, 1))[:, :, np.newaxis]
        else:
            # Corner gradient: add channel dimension
            gradient = gradient[:, :, np.newaxis]
        
        # Apply to all channels (broadcast automatically)
        img_array = img_array * gradient
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _apply_jpeg_compression(self, img: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression artifacts."""
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_resolution_degradation(self, img: Image.Image, scale: float) -> Image.Image:
        """Downscale and upscale to simulate low-res scan."""
        w, h = img.size
        small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def _apply_perspective_warp(self, img: Image.Image) -> Image.Image:
        """Apply slight perspective distortion (phone camera angle)."""
        w, h = img.size
        
        # Small random offsets for corners
        margin = 0.03  # 3% max distortion
        
        # Original corners: TL, TR, BR, BL
        src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
        
        # Distorted corners
        dst_pts = [
            (random.uniform(0, margin * w), random.uniform(0, margin * h)),
            (w - random.uniform(0, margin * w), random.uniform(0, margin * h)),
            (w - random.uniform(0, margin * w), h - random.uniform(0, margin * h)),
            (random.uniform(0, margin * w), h - random.uniform(0, margin * h)),
        ]
        
        # PIL doesn't have perspective transform directly, use simple approach
        # For production, use cv2.getPerspectiveTransform
        # Approximation: just slight rotation/scale
        angle = random.uniform(-2, 2)
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255), expand=False)

    def _random_crop(self, img: Image.Image, bboxes: List[Dict], min_keep: float = 0.92) -> Tuple[Image.Image, List[Dict]]:
        """Random crop that preserves most of the image."""
        w, h = img.size
        
        # Calculate crop margins
        max_crop = 1.0 - min_keep
        left = random.uniform(0, max_crop / 2) * w
        top = random.uniform(0, max_crop / 2) * h
        right = w - random.uniform(0, max_crop / 2) * w
        bottom = h - random.uniform(0, max_crop / 2) * h
        
        # Crop image
        cropped = img.crop((int(left), int(top), int(right), int(bottom)))
        
        # Resize back to original size (keeps training consistent)
        cropped = cropped.resize((w, h), Image.LANCZOS)
        
        # Adjust bboxes (in normalized coords)
        new_w = right - left
        new_h = bottom - top
        
        adjusted_bboxes = []
        for bbox in bboxes:
            # Convert from normalized to pixel coords
            xc_px = bbox["x_center"] * w
            yc_px = bbox["y_center"] * h
            bw_px = bbox["width"] * w
            bh_px = bbox["height"] * h
            
            # Shift by crop offset
            xc_px -= left
            yc_px -= top
            
            # Re-normalize to cropped size
            xc_norm = xc_px / new_w
            yc_norm = yc_px / new_h
            w_norm = bw_px / new_w
            h_norm = bh_px / new_h
            
            adjusted_bboxes.append({
                "class_id": bbox["class_id"],
                "x_center": xc_norm,
                "y_center": yc_norm,
                "width": w_norm,
                "height": h_norm
            })
        
        return cropped, adjusted_bboxes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment YOLO dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output augmented dataset directory")
    parser.add_argument("--copies", type=int, default=3, help="Number of augmented copies per image")
    
    args = parser.parse_args()
    
    augmentor = DatasetAugmentor(
        Path(args.input),
        Path(args.output),
        augmentations_per_image=args.copies
    )
    augmentor.augment()

if __name__ == "__main__":
    main()
