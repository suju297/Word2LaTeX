"""IR Block Mapper - Link PDF regions to IR blocks.

This module is responsible for identifying which intermediate representation (IR)
block corresponds to a specific visual difference detected in the PDF.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Any

from wordtolatex.ir import (
    Document, 
    Block, 
    HeadingBlock, 
    ParagraphBlock, 
    TableBlock, 
    ListBlock,
    ListItem
)
from wordtolatex.vision_diff import RegionDiff, DiffType


class BlockMapper:
    """Maps visual regions to document structure."""
    
    def __init__(self, doc: Document):
        self.doc = doc
        self.blocks = doc.blocks
        # Pre-compute block text (normalized) for faster matching
        self._block_cache = []
        for block in self.blocks:
            self._block_cache.append({
                "id": block.id,
                "text": self._get_block_text(block),
                "type": block.block_type,
                "block": block
            })

    def find_block_for_diff(self, diff: RegionDiff) -> Optional[Block]:
        """Find the IR block corresponding to the given diff."""
        
        target_text = diff.ref_text if diff.diff_type == DiffType.MISSING else diff.gen_text
        if not target_text:
            return None
            
        # 1. Try exact/high-similarity match
        best_match = None
        best_score = 0.0
        
        # Determine strictness based on whether it's a heading
        heading_boost = diff.is_heading_like
        
        for item in self._block_cache:
            # Type filtering optimization
            if heading_boost and not isinstance(item["block"], HeadingBlock):
                continue
                
            score = self._compute_similarity(target_text, item["text"])
            
            # Special handling for numbered headings (IR usually lacks prefix)
            if diff.has_numbering and isinstance(item["block"], HeadingBlock):
                # Try stripping prefix from diff text
                stripped_target = self._strip_numbering(target_text)
                score_stripped = self._compute_similarity(stripped_target, item["text"])
                score = max(score, score_stripped)
            
            if score > best_score:
                best_score = score
                best_match = item["block"]
        
        # Thresholds
        threshold = 0.8 if not heading_boost else 0.6  # Lower threshold for headings (prefixes)
        
        if best_score > threshold:
            return best_match
            
        return None

    def _get_block_text(self, block: Block) -> str:
        """Extract plain text from a block."""
        if hasattr(block, "content") and isinstance(block.content, list):
            # Paragraph, Heading, ListItem
            return "".join(span.text for span in block.content).strip()
        elif isinstance(block, ListBlock):
            # Concatenate list items
            return " ".join(self._get_block_text(item) for item in block.items)
        elif isinstance(block, ListItem):
             return "".join(span.text for span in block.content).strip()
        elif hasattr(block, "rows"): # Table
            # Just grab some text for matching
            text = []
            for row in block.rows:
                for cell in row.cells:
                     # Recursive but simplified
                     pass # Implementing full table text is expensive/complex here
            return "TABLE" # Placeholder
            
        return ""

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity (0-1)."""
        if not text1 or not text2:
            return 0.0
            
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return 1.0
            
        # Jaccard
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union

    def _strip_numbering(self, text: str) -> str:
        """Strip numbering prefix like '1.', 'I.', 'A.' for better matching."""
        # Simple patterns
        text = re.sub(r'^[IVXivx]+\.\s*', '', text) # Roman
        text = re.sub(r'^\d+(\.)?\s*', '', text)    # Arabic
        text = re.sub(r'^[A-Z]\.\s*', '', text)     # Letter
        return text.strip()

def map_diffs_to_blocks(doc: Document, diffs: List[RegionDiff]) -> List[Tuple[RegionDiff, Optional[Block]]]:
    """Batch map diffs to blocks."""
    mapper = BlockMapper(doc)
    results = []
    for diff in diffs:
        block = mapper.find_block_for_diff(diff)
        results.append((diff, block))
    return results
