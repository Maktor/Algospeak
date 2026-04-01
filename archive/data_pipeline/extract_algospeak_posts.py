#!/usr/bin/env python3
"""
Extract posts labeled as algospeak ("yes" or "unsure") from JSONL file.
Shows post text and LLM reasoning for each.
"""

#!/usr/bin/env python3
"""
Extract Algospeak Posts from LLM Labels

This script filters and displays posts that were labeled as containing algospeak ("yes" or "unsure")
from the LLM labeling output. Useful for analyzing detected algospeak patterns.

FUNCTIONALITY:
- Reads llm_labeled_posts.jsonl (output from llm_label_posts.py)
- Filters for posts with is_coded="yes" or is_coded="unsure"
- Displays detailed analysis including text, labels, confidence, reasoning
- Optional: Saves results to JSON or formatted text file

RUN EXAMPLES:
    # Display algospeak posts to console
    python extract_algospeak_posts.py

    # Save detailed JSON results
    python extract_algospeak_posts.py --output algospeak_analysis.json

    # Save formatted text analysis
    python extract_algospeak_posts.py --save-output algospeak_report.txt

OUTPUT:
- Console display of all algospeak posts with full details
- Optional JSON file with structured data
- Optional text file with formatted analysis

DEPENDENCIES:
- Requires generated_data/llm_labeled_posts.jsonl from llm_label_posts.py
"""

import argparse
import json
from typing import Any, Dict


def extract_algospeak_posts(jsonl_path: str, output_path: str = None, save_output_path: str = None) -> None:
    """Extract and display posts labeled as 'yes' or 'unsure'."""
    extracted = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                is_coded = rec.get("is_coded")

                if is_coded in ["yes", "unsure"]:
                    post_data = {
                        "line": line_num,
                        "text": rec.get("text", ""),
                        "is_coded": is_coded,
                        "confidence": rec.get("confidence", 0.0),
                        "domain": rec.get("domain", ""),
                        "mechanism": rec.get("mechanism", ""),
                        "reasoning": rec.get("reasoning", ""),
                        "spans": rec.get("spans", []),
                        "decoded": rec.get("decoded", []),
                        "needs_review": rec.get("needs_review", False),
                    }
                    extracted.append(post_data)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    # Prepare output
    output_lines = []
    output_lines.append(f"Found {len(extracted)} algospeak posts (yes/unsure):\n")

    for i, post in enumerate(extracted, 1):
        output_lines.append(f"--- Post {i} ---")
        output_lines.append(f"Text: {post['text']}")
        output_lines.append(f"Label: {post['is_coded']} (confidence: {post['confidence']})")
        output_lines.append(f"Domain: {post['domain']}")
        output_lines.append(f"Mechanism: {post['mechanism']}")
        output_lines.append(f"Spans: {post['spans']}")
        output_lines.append(f"Decoded: {post['decoded']}")
        output_lines.append(f"Needs Review: {post['needs_review']}")
        output_lines.append(f"Reasoning: {post['reasoning']}")
        output_lines.append("")

    output_text = "\n".join(output_lines)

    # Display results
    print(output_text)

    # Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed results to: {output_path}")

    # Save formatted output if requested
    if save_output_path:
        os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
        with open(save_output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Saved formatted output to: {save_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract algospeak posts from JSONL")
    parser.add_argument("--input", "-i", default="generated_data/llm_labeled_posts.jsonl",
                       help="Input JSONL file path")
    parser.add_argument("--output", "-o", default=None,
                       help="Optional output JSON file for detailed results")
    parser.add_argument("--save-output", "-s", default=None,
                       help="Optional file to save the formatted text output")

    args = parser.parse_args()

    extract_algospeak_posts(args.input, args.output, args.save_output)


if __name__ == "__main__":
    main()