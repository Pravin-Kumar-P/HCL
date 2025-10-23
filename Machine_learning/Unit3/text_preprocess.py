import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
#nltk.download('punkt')
#nltk.download('punkt_tab')  
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
text = "The quick brown fox jumps over the lazy dog near the river bank."
print("Original Text:\n", text)
# 1 TOKENIZATION
tokens = word_tokenize(text.lower())
print("\nTokenized Words:\n", tokens)
# 2 STOP-WORD REMOVAL
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in tokens if w.isalnum() and w not in stop_words]
print("\nEnglish Stop Words Example (some common ones):\n", list(stop_words)[:20])
print("\nAfter Stop-word Removal:\n", filtered_words)
# 3️ STEMMING
ps = PorterStemmer()
stemmed_words = [ps.stem(w) for w in filtered_words]
print("\nAfter Stemming:\n", stemmed_words)
# 4️ LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
print("\nAfter Lemmatization:\n", lemmatized_words)
