#!/usr/bin/env python3
"""Test layout analysis models on IEEE_TEMPLATE.

Tests Surya, DocTR, and PaddleOCR for layout detection capabilities on M4 Mac.

Usage: python scripts/test_layout_models.py
"""

import time
from pathlib import Path

# Test PDF
TEST_PDF = Path("tests/corpus/IEEE_TEMPLATE/ref.pdf")


def test_surya():
    """Test Surya layout detection."""
    print("\n" + "="*60)
    print("TESTING SURYA")
    print("="*60)
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from PIL import Image
        import fitz  # PyMuPDF to render PDF to images
        
        # Check device
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Open PDF and render to images
        print(f"Loading PDF: {TEST_PDF}")
        doc = fitz.open(str(TEST_PDF))
        images = []
        for i in range(min(2, len(doc))):  # First 2 pages
            pix = doc[i].get_pixmap(dpi=144)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        print(f"Rendered {len(images)} pages to images")
        
        # Load model (two-step: Foundation -> Layout)
        print("Loading Surya models...")
        start = time.time()
        foundation = FoundationPredictor(device=device)
        layout_predictor = LayoutPredictor(foundation)
        print(f"Models loaded in {time.time() - start:.1f}s")
        
        # Run detection
        print("Running layout detection...")
        start = time.time()
        results = layout_predictor(images)
        elapsed = time.time() - start
        
        # Show results
        for i, result in enumerate(results):
            print(f"\nPage {i+1}: {len(result.bboxes)} regions detected")
            for bbox in result.bboxes[:10]:  # First 10 boxes
                print(f"  - {bbox.label}: ({bbox.bbox[0]:.0f},{bbox.bbox[1]:.0f}) - ({bbox.bbox[2]:.0f},{bbox.bbox[3]:.0f})")
        
        print(f"\n✅ Surya completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ Surya failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_doctr():
    """Test DocTR layout detection."""
    print("\n" + "="*60)
    print("TESTING DOCTR")
    print("="*60)
    
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Check device
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load PDF
        print(f"Loading PDF: {TEST_PDF}")
        doc = DocumentFile.from_pdf(str(TEST_PDF))
        print(f"Loaded {len(doc)} pages")
        
        # Load model
        print("Loading DocTR model...")
        start = time.time()
        model = ocr_predictor(pretrained=True)
        print(f"Model loaded in {time.time() - start:.1f}s")
        
        # Run detection (first 2 pages)
        print("Running OCR + layout detection...")
        start = time.time()
        result = model(doc[:2])
        elapsed = time.time() - start
        
        # Show results
        for i, page in enumerate(result.pages):
            print(f"\nPage {i+1}: {len(page.blocks)} blocks")
            for j, block in enumerate(page.blocks[:5]):
                words = sum(len(line.words) for line in block.lines)
                print(f"  - Block {j+1}: {words} words, confidence={block.lines[0].words[0].confidence:.2f}")
        
        print(f"\n✅ DocTR completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ DocTR failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_paddleocr():
    """Test PaddleOCR/PaddleX layout detection."""
    print("\n" + "="*60)
    print("TESTING PADDLEX (LAYOUT PARSING)")
    print("="*60)
    
    try:
        import os
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"  # Skip connectivity check
        
        from paddlex import create_pipeline
        import fitz
        import numpy as np
        from PIL import Image
        
        print(f"Loading PDF: {TEST_PDF}")
        
        # Render PDF to images
        doc = fitz.open(str(TEST_PDF))
        images = []
        for i in range(min(2, len(doc))):
            pix = doc[i].get_pixmap(dpi=144)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Save temp image (PaddleX needs file path)
            temp_path = f"/tmp/paddle_test_page_{i}.png"
            img.save(temp_path)
            images.append(temp_path)
        print(f"Rendered {len(images)} pages")
        
        # Create layout parsing pipeline
        print("Loading PaddleX layout parsing pipeline...")
        start = time.time()
        pipeline = create_pipeline("layout_parsing")
        print(f"Pipeline loaded in {time.time() - start:.1f}s")
        
        # Run detection
        print("Running layout detection...")
        start = time.time()
        
        for i, img_path in enumerate(images):
            result = pipeline.predict(img_path)
            # PaddleX returns generator, convert to list
            result_list = list(result)
            if result_list:
                res = result_list[0]
                # Get layout boxes
                layout_info = res.get('layout_parsing_result', {})
                regions = layout_info.get('parsing_result', [])
                print(f"\nPage {i+1}: {len(regions)} regions detected")
                for j, region in enumerate(regions[:10]):
                    label = region.get('layout_label', 'unknown')
                    bbox = region.get('layout_bbox', [0,0,0,0])
                    print(f"  - {label}: ({bbox[0]:.0f},{bbox[1]:.0f}) - ({bbox[2]:.0f},{bbox[3]:.0f})")
            else:
                print(f"\nPage {i+1}: No results")
        
        elapsed = time.time() - start
        print(f"\n✅ PaddleX completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ PaddleX failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Layout Model Comparison Test")
    print(f"Test PDF: {TEST_PDF}")
    print(f"Exists: {TEST_PDF.exists()}")
    
    # Test each model
    results = {}
    results["surya"] = test_surya()
    results["doctr"] = test_doctr()
    results["paddleocr"] = test_paddleocr()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"  {model}: {status}")


if __name__ == "__main__":
    main()
