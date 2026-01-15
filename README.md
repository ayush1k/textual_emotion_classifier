# Emotion Detection from Text

## Introduction
This project focuses on building and evaluating machine learning models to classify emotions from textual data. The goal is to predict one of six emotions (sadness, anger, love, surprise, fear, joy) based on a given text input.

## Dataset
The dataset used is `train.txt`, which contains text entries and their corresponding emotions. The dataset has 16,000 entries and two columns: 'text' and 'emotion'.

### Emotion Mapping
During preprocessing, the categorical emotion labels were converted into numerical representations:
```
{'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
```

## Preprocessing Steps
Before training the models, the text data underwent several preprocessing steps:
1.  **Lowercasing**: All text was converted to lowercase.
2.  **Punctuation Removal**: All punctuation marks were removed from the text.
3.  **Number Removal**: Numerical digits were removed.
4.  **Emoji Removal**: Non-ASCII characters (assumed to be emojis in this context) were removed.
5.  **Stopword Removal**: Common English stopwords (e.g., 'the', 'is', 'a') were removed to focus on more meaningful words.

## Models Used
Three different machine learning models were trained and evaluated:
1.  **Multinomial Naive Bayes with CountVectorizer (BoW)**: Uses a Bag-of-Words representation of the text features.
2.  **Multinomial Naive Bayes with TfidfVectorizer (TF-IDF)**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) representation of the text features.
3.  **Logistic Regression with TfidfVectorizer (TF-IDF)**: Uses TF-IDF representation with a Logistic Regression classifier.

## Results
Each model's performance was evaluated based on accuracy on the test set.

-   **Multinomial Naive Bayes (CountVectorizer)**: `0.768125`
-   **Multinomial Naive Bayes (TfidfVectorizer)**: `0.6609375`
-   **Logistic Regression (TfidfVectorizer)**: `0.8628125`

The **Logistic Regression model with TF-IDF features** achieved the highest accuracy among the tested models.

## Usage
To run this code:
1.  Ensure you have the `train.txt` dataset in the same directory as your notebook or script.
2.  Install the required Python libraries:
    ```bash
    pip install pandas numpy scikit-learn nltk seaborn matplotlib
    ```
3.  Run the cells sequentially. The code will perform data loading, preprocessing, model training, and evaluation.

## Dependencies
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `nltk`
-   `scikit-learn`
