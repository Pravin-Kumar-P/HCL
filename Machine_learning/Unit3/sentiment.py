import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tabulate import tabulate
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
file_path = "chat.csv"

try:
    df = pd.read_csv(file_path, sep='\t', header=None, engine='python')
    print("Dataset loaded successfully!")

    # Rename first column as 'response'
    if 0 in df.columns:
        df.rename(columns={0: 'response'}, inplace=True)
        df = df[['response']].copy()
    else:
        print("Error: Could not find text column.")
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
    df = pd.read_csv(file_path, header=None, engine='python', sep=None)
    df.rename(columns={0: 'response'}, inplace=True)
    df = df[['response']].copy()

print(df.head())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"human \d+: ", "", text)     
    text = re.sub(r"[^a-z\s]", "", text)       
    text = re.sub(r"\s+", " ", text).strip()   
    return text

df['clean_text'] = df['response'].apply(clean_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['clean_text'].apply(get_sentiment)

X = df['clean_text']
y = df['Sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

y_pred = svm_model.predict(X_test_tfidf)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))

sample_results = pd.DataFrame({
    "Response": X_test.sample(10, random_state=42),
})
sample_results["Predicted Sentiment"] = svm_model.predict(vectorizer.transform(sample_results["Response"]))

print("\n Sample Predictions:")
print(tabulate(sample_results, headers='keys', tablefmt='fancy_grid', showindex=False))
