#!/usr/bin/env python3
"""
Convert llm_labeled_posts.jsonl to CSV format for easier analysis.
"""

#!/usr/bin/env python3
"""
Convert JSONL to CSV Format

This script converts the JSONL output from LLM labeling to CSV format for easier analysis
in spreadsheet applications like Excel or Google Sheets.

FUNCTIONALITY:
- Reads generated_data/llm_labeled_posts.jsonl (output from llm_label_posts.py)
- Converts all records to CSV format
- Handles complex data types (lists converted to comma-separated strings)
- Maintains all columns and data integrity

RUN EXAMPLES:
    # Convert to CSV with default names
    python jsonl_to_csv.py

    # Custom input/output files
    python jsonl_to_csv.py --input my_labels.jsonl --output my_labels.csv

OUTPUT:
- CSV file with all labeling data
- Automatic handling of lists (spans, decoded fields)
- UTF-8 encoding for international characters

DEPENDENCIES:
- Requires llm_labeled_posts.jsonl from llm_label_posts.py
- Python csv module (built-in)

USE CASES:
- Data analysis in Excel/Google Sheets
- Import into data science tools (pandas, R, etc.)
- Sharing labeled datasets with collaborators
"""

import argparse
import csv
import json
from typing import Any, Dict, List


def jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """Convert JSONL file to CSV format."""
    data = []

    # Read JSONL file
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                data.append(rec)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    if not data:
        print("No data found in JSONL file")
        return

    # Get all unique keys from the data
    all_keys = set()
    for rec in data:
        all_keys.update(rec.keys())

    # Define the field order (prioritize important fields first)
    field_order = [
        "hash",
        "text",
        "is_coded",
        "domain",
        "mechanism",
        "confidence",
        "spans",
        "decoded",
        "reasoning",
        "needs_review",
        "model_used",
        # Add any other fields that might exist
    ]

    # Add any remaining keys not in the predefined order
    remaining_keys = all_keys - set(field_order)
    field_order.extend(sorted(remaining_keys))

    # Convert lists to strings for CSV compatibility
    def format_field(value: Any) -> str:
        if isinstance(value, list):
            # Convert lists to comma-separated strings, or JSON if complex
            if all(isinstance(item, str) for item in value):
                return ", ".join(value)
            else:
                return json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            return str(value).lower()
        elif value is None:
            return ""
        else:
            return str(value)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()

        for rec in data:
            # Format each field
            formatted_rec = {}
            for key in field_order:
                formatted_rec[key] = format_field(rec.get(key, ""))

            writer.writerow(formatted_rec)

    print(f"Converted {len(data)} records to CSV: {csv_path}")
    print(f"Fields: {', '.join(field_order)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL to CSV")
    parser.add_argument("--input", "-i", default="generated_data/llm_labeled_posts.jsonl",
                       help="Input JSONL file path")
    parser.add_argument("--output", "-o", default="generated_data/llm_labeled_posts.csv",
                       help="Output CSV file path")

    args = parser.parse_args()

    jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()