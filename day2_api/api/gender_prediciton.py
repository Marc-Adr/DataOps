import pandas_gbq
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from google.oauth2 import service_account
import json
import joblib

with open("/home/ensai/Documents/dataops/ensai-2025-f0ebdf104921.json") as source:
    info = json.load(source)

credentials = service_account.Credentials.from_service_account_info(info)

# Load data from BigQuery
data = pandas_gbq.read_gbq("""SELECT sexe, preusuel as prenom, sum(cast(nombre as int)) as total 
                           FROM ml.prenoms group by prenom, sexe""",
                           project_id="ensai-2025", credentials= credentials)

# Preprocessing
X = data['prenom']
y = data['sexe']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))  # Use character n-grams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer to .bin files
joblib.dump(model, 'gender_prediction_model.bin')
joblib.dump(vectorizer, 'tfidf_vectorizer.bin')