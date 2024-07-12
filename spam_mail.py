import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
# You can download the dataset from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    return ' '.join(words)

# Apply preprocessing to messages
df['message'] = df['message'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Classify user-provided message
def classify_message(message):
    preprocessed_message = preprocess_text(message)
    message_vec = vectorizer.transform([preprocessed_message])
    prediction = model.predict(message_vec)
    return prediction[0]

# Ask the user for a message
user_message = input("Enter an SMS message to classify: ")
classification = classify_message(user_message)
print(f'The message is classified as: {classification}')
