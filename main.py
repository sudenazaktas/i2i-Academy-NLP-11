import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)
print(os.listdir(path))

# Load the dataset into a pandas DataFrame
csv_path = os.path.join(path, "IMDB Dataset.csv")
df = pd.read_csv(csv_path)

print(df.shape)
print(df.columns.tolist())

import re
import string

def clean_text(text):
    
    #Cleans raw review text to prepare it for sentiment analysis.
    #Lowercasing and removing punctuation reduce noise so the model
    #treats words like 'Great' and 'great!' as the same token.
    text = text.lower()  # normalize case so 'Good' and 'good' are treated the same
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation (commas, periods, etc.)
    text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces into one and trim edges
    return text

# Apply the cleaning function to every review and store the result in a new column
df['clean_review'] = df['review'].apply(clean_text)

# Quick sanity check: compare original vs cleaned text for the first few rows
print(df[['review', 'clean_review']].head())

from textblob import TextBlob

def get_sentiment_score(text):
    
    #Uses TextBlob's pre-trained sentiment model to score a piece of text.
    #Polarity ranges from -1.0 (very negative) to +1.0 (very positive).
    return TextBlob(text).sentiment.polarity

# Calculate a sentiment score for every cleaned review
df['sentiment_score'] = df['clean_review'].apply(get_sentiment_score)

# Quick sanity check: look at a few reviews alongside their scores
print(df[['clean_review', 'sentiment_score']].head())

def label_sentiment(score):
    
    #Converts a numeric polarity score into a human-readable label.
    #Scores close to zero (between -0.1 and 0.1) are treated as neutral
    #since TextBlob rarely returns an exact 0 even for balanced text.

    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply the labeling function based on each review's sentiment score
df['ai_label'] = df['sentiment_score'].apply(label_sentiment)

# Print final statistics: how many reviews fall into each category
print(df['ai_label'].value_counts())