import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample training data
data = pd.DataFrame({
    'bug': [
        'app crashes on submit',
        'UI not responsive on mobile',
        'slow loading dashboard',
        'login fails randomly'
    ],
    'severity': ['High', 'Medium', 'Low', 'High']
})

X = data['bug']
y = data['severity']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

joblib.dump(model, 'model/bug_classifier.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
