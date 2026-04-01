#!/usr/bin/env python3
"""
Algospeak Inspection Script

This script filters labeled posts to extract only those that contain algospeak detections.
It reads from the JSONL output of data_processing.py and creates a subset containing
only posts where algospeak terms were found.

FUNCTIONALITY:
- Reads labeled_posts.jsonl (output from data_processing.py)
- Filters for posts with non-empty algospeak_hits
- Saves filtered results to algospeak_labeled_only.jsonl
- Prints count of algospeak-detected posts

RUN:
    python inspect_algospeek.py
    # Input: generated_data/labeled_posts.jsonl
    # Output: generated_data/algospeak_labeled_only.jsonl

DEPENDENCIES:
- Requires generated_data/labeled_posts.jsonl from data_processing.py
"""

import json

INFILE = "generated_data/labeled_posts.jsonl"
OUTFILE = "generated_data/algospeak_labeled_only.jsonl"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

count = 0
with open(INFILE, "r", encoding="utf-8") as f, open(OUTFILE, "w", encoding="utf-8") as out:
    for line in f:
        rec = json.loads(line)
        if rec.get("algospeak_hits"):   # non-empty list
            count += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved {count} algospeak-labeled posts to {OUTFILE}")
