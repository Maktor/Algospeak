#!/usr/bin/env python3
"""
Bluesky Data Collection Script

This script connects to the Bluesky firehose (Jetstream) to collect real-time social media posts
for algospeak analysis. It filters for English-language posts and saves them to a text file.

FUNCTIONALITY:
- Connects to Bluesky's WebSocket firehose API
- Filters posts by language (English only using langdetect)
- Collects specified number of posts
- Saves cleaned post text to file (one post per line)

RUN:
    python test_pull.py
    # Collects 10,000 English posts by default
    # Output: posts_10000.txt

CUSTOMIZE:
    smoke_test(max_posts=500)  # Collect fewer posts for testing

REQUIREMENTS:
- websocket-client: pip install websocket-client
- langdetect: pip install langdetect

API:
- Uses Bluesky's public Jetstream API (wss://jetstream2.us-east.bsky.network)
- No authentication required for read-only access
"""

import json
from websocket import create_connection
from langdetect import detect, LangDetectException

URL = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

def smoke_test(max_posts=25):
    ws = create_connection(URL)
    seen = 0
    with open("posts_10000.txt", "w", encoding="utf-8") as f:
        try:
            while seen < max_posts:
                msg = ws.recv()
                evt = json.loads(msg)

                # Jetstream event: record is nested in commit structure
                commit = evt.get("commit", {})
                rec = commit.get("record", {})
                if rec.get("$type") == "app.bsky.feed.post":
                    text = rec.get("text", "")
                    # Filter for English language posts only
                    try:
                        if detect(text) == "en":
                            post_text = text.replace("\n", " ")
                            print(post_text)
                            f.write(post_text + "\n")
                            seen += 1
                    except LangDetectException:
                        # Skip posts where language detection fails
                        pass
        finally:
            ws.close()

if __name__ == "__main__":
    smoke_test(10000)
