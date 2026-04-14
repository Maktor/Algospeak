import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "original_datasets"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HATEBASE_TRAIN_PATH = DATA_DIR / "hatebase_train.parquet"
HATEBASE_TEST_PATH = DATA_DIR / "hatebase_test.parquet"
BLUESKY_PATH = DATA_DIR / "bluesky.csv"

OUT_PATH = OUT_DIR / "combined_for_annotation.csv"

for p in [HATEBASE_TRAIN_PATH, HATEBASE_TEST_PATH, BLUESKY_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}. Put it in dataset_cleaning/original_datasets/")

def load_hatebase(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")

    # Hatebase columns you showed: tweet, category, data, class
    # Standardize to your new schema
    out = pd.DataFrame({
        "text": df["tweet"].astype(str),
        "source": df["data"].astype(str),          # e.g., GoEmotions, HateXplain, Slur, etc.
        "hb_class": df["class"].astype(str),       # keep safe/unsafe for reference (optional)
        "hb_category": df["category"].astype(str), # keep original category for reference (optional)
    })

    out["classification"] = ""  # empty column for your team labels
    return out

def load_bluesky(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 👇 IMPORTANT: pick the column that contains the post text
    # In your screenshot I can’t reliably read the header names.
    # Common names are: "text", "post", "content", "body"
    # Replace BLUESKY_TEXT_COL with the correct column name from your file.
    BLUESKY_TEXT_COL = "text"  # <-- change if needed

    out = pd.DataFrame({
        "text": df[BLUESKY_TEXT_COL].astype(str),
        "source": "bluesky",
    })

    out["hb_class"] = ""       # bluesky doesn’t have Hatebase labels
    out["hb_category"] = ""
    out["classification"] = "" # empty for annotation
    return out

def normalize_for_dedup(df: pd.DataFrame) -> pd.Series:
    # normalize ONLY for duplicate detection (do not overwrite text)
    return (
        df["text"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

def main():
    # Load
    hb_train = load_hatebase(HATEBASE_TRAIN_PATH)
    hb_test  = load_hatebase(HATEBASE_TEST_PATH)
    bluesky  = load_bluesky(BLUESKY_PATH)

    print("Hatebase train:", hb_train.shape)
    print("Hatebase test :", hb_test.shape)
    print("Bluesky      :", bluesky.shape)

    # Combine
    combined = pd.concat([hb_train, hb_test, bluesky], ignore_index=True)
    print("Combined (before drops):", combined.shape)

    # Drop OWS (only affects hatebase rows)
    combined = combined[combined["source"].str.lower() != "ows"].copy()
    print("After dropping OWS:", combined.shape)

    # Remove empty texts
    combined["text"] = combined["text"].astype(str)
    combined = combined[combined["text"].str.strip() != ""].copy()
    print("After dropping empty text:", combined.shape)

    # Dedupe on normalized text
    combined["_text_norm"] = normalize_for_dedup(combined)
    dup_rows = combined.duplicated("_text_norm").sum()
    print("Duplicate rows to remove:", dup_rows)

    combined = combined.drop_duplicates(subset="_text_norm", keep="first").copy()
    combined = combined.drop(columns=["_text_norm"])
    print("Final combined shape:", combined.shape)

    # Quick distribution check
    print("\nSources (top 15):")
    print(combined["source"].value_counts().head(15))

    # Save
    combined.to_parquet(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH.resolve())

if __name__ == "__main__":
    main()