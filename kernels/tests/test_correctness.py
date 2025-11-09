"""
test_correctness.py
Validate GPU Smith-Waterman against CPU baseline (Biopython)
"""

import sys
sys.path.insert(0, '../smith_waterman')

import numpy as np
from Bio import pairwise2
from smith_waterman import SmithWatermanGPU

# TODO: Implement correctness tests
# Test cases:
# 1. Identical sequences (should have max score)
# 2. Completely different sequences (should have low score)
# 3. Known alignment pairs with expected scores
# 4. Edge cases: very short sequences, very long sequences
