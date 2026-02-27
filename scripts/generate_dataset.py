#!/usr/bin/env python3
"""Generate synthetic dataset for layout analysis using Gemini."""

import argparse
import sys
from pathlib import Path
from wordtolatex.dataset_generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic layout dataset using Gemini Vision.")
    parser.add_argument("--input", type=str, required=True, help="Input corpus directory containing PDFs")
    parser.add_argument("--output", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to process (0 = all)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)
        
    print(f"Generating dataset from {input_dir} to {output_dir}...")
    
    try:
        generator = DatasetGenerator(output_dir)
        
        # Pass the limit argument to process_corpus
        generator.process_corpus(input_dir, limit=args.limit)
        
        print("\nDataset generation complete.")
        print(f"Images: {output_dir}/images")
        print(f"YOLO Labels: {output_dir}/labels")
        print(f"COCO JSON: {output_dir}/annotations.json")
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
