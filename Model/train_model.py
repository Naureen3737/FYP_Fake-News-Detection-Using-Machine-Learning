import pandas as pd
import numpy as np
import joblib
import re
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
import warnings
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset with correct encoding
df = pd.read_csv("train.csv")


# Ensure label consistency
df['label'] = df['label'].str.strip().str.upper()

# Keep only REAL and FAKE labels
df = df[df['label'].isin(['REAL', 'FAKE'])]

# Extract features and labels
X = df['text'].fillna("")  # Replace NaN with empty string
y = df['label']

# Advanced preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove special characters and lowercase
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Apply preprocessing
X_cleaned = X.apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.7,
    min_df=2,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTE on vectorized data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# PAC Model with GridSearchCV
pac_model = PassiveAggressiveClassifier(random_state=42)
param_grid = {
    'max_iter': [50, 100, 200],
    'C': [0.01, 0.1, 1, 10],
    'loss': ['hinge', 'squared_hinge']
}

grid_search = GridSearchCV(
    estimator=pac_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_resampled, y_train_resampled)
best_pac_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_pac_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔍 Accuracy: {round(accuracy * 100, 2)}%")

print("\n📊 Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

print("\n🧮 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
print(conf_matrix)

# Optional: Save reports to file
# with open("classification_report.txt", "w") as f:
#     f.write(report)
# np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",", fmt="%d")

# Cross-validation
cross_val_accuracy = cross_val_score(best_pac_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print(f"\n🔁 Cross-validation Accuracy: {round(np.mean(cross_val_accuracy) * 100, 2)}%")

# Save model and vectorizer
joblib.dump(best_pac_model, 'improved_pac_model.pkl', compress=3)
joblib.dump(vectorizer, 'improved_vectorizer.pkl', compress=3)
joblib.dump(accuracy, 'improved_accuracy.pkl', compress=3)
print("\n✅ Improved Model, Vectorizer, and Accuracy saved successfully!")