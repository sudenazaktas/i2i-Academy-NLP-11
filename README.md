# i2i-Academy-NLP-11

A Python script that performs sentiment analysis on the IMDB 50K Movie Reviews dataset using a pre-trained NLP model.

## What it does
- Downloads the dataset directly from Kaggle using `kagglehub`
- Cleans review text (removes HTML tags, punctuation, and normalizes case)
- Calculates sentiment polarity scores using TextBlob
- Labels each review as Positive, Negative, or Neutral based on the score
- Compares AI-generated labels against the dataset's true labels (achieves 78% accuracy)

## Requirements
```bash
pip install kagglehub pandas textblob
```

## Usage
```bash
python main.py
```
