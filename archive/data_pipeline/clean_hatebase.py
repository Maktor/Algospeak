#Loading datset
import pandas as pd

hatebase = pd.read_parquet(
    '/Users/nataliaherrera/Downloads/train-00000-of-00001.parquet',
    engine='pyarrow'
)

print(hatebase.head())

#Inpsect data features
print("\nShape:", hatebase.shape)

print("\nColumns:")
print(hatebase.columns)

print("\nMissing values:")
print(hatebase.isnull().sum())

print("\nClass distribution:")
print(hatebase['class'].value_counts())

# Normalize text ONLY for duplicate detection
hatebase['tweet_norm'] = (
    hatebase['tweet']
    .astype(str)
    .str.strip()
    .str.lower()
)

# Count duplicates
duplicate_count = hatebase.duplicated('tweet_norm').sum()
print("\nNumber of duplicate tweets:", duplicate_count)

# Show rows that are duplicates
duplicates = hatebase[hatebase.duplicated('tweet_norm', keep=False)]

# Remove duplicates (keep first occurrence)
hatebase_clean = hatebase.drop_duplicates(subset='tweet_norm')

print("Shape after removing duplicates:", hatebase_clean.shape)

# Drop helper column
hatebase_clean = hatebase_clean.drop(columns=['tweet_norm'])

# Save cleaned dataset
hatebase_clean.to_parquet("hatebase_cleaned.parquet", index=False)

print("\nCleaned dataset saved as hatebase_cleaned.parquet")

print("\nClass distribution AFTER cleaning:")
print(hatebase_clean['class'].value_counts())
