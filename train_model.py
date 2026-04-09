import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle

real_news = [
    "Government announces new education policy in India",
    "ISRO successfully launches satellite",
    "Supreme Court gives new judgement",
    "Elections conducted peacefully",
    "New hospital opened in Delhi"
]

fake_news = [
    "Click here to win free iPhone now",
    "Miracle cure doctors hide",
    "Earn money without work",
    "Shocking secret revealed",
    "Get rich overnight trick"
]

texts = real_news + fake_news
labels = [1]*len(real_news) + [0]*len(fake_news)

df = pd.DataFrame({"text": texts, "label": labels})

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])

model = LinearSVC()
model.fit(X, df['label'])

pickle.dump(model, open('fake_news_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained")