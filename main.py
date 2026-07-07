import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)
print(os.listdir(path))

import re
import string

def clean_text(text):
    """
    Cleans raw review text to prepare it for sentiment analysis.
    Lowercasing and removing punctuation reduce noise so the model
    treats words like 'Great' and 'great!' as the same token.
    """
    text = text.lower()  # normalize case so 'Good' and 'good' are treated the same
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation (commas, periods, etc.)
    text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces into one and trim edges
    return text

# Apply the cleaning function to every review and store the result in a new column
df['clean_review'] = df['review'].apply(clean_text)

# Quick sanity check: compare original vs cleaned text for the first few rows
print(df[['review', 'clean_review']].head())