# from scratch

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultinomialNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_word_counts = {}
        self.class_totals = {}
        self.class_priors = {}
        self.vocab_size = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]
            self.class_word_counts[c] = X_c.sum(axis=0)
            self.class_totals[c] = self.class_word_counts[c].sum()
            self.class_priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                # Add 1 for Laplace smoothing
                likelihood = np.sum(x * np.log((self.class_word_counts[c] + 1) / (self.class_totals[c] + self.vocab_size)))
                class_probs[c] = prior + likelihood
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

# Sample data
texts = [
    "Buy cheap meds now", "Limited time offer!", "Win money instantly", "Congratulations, you won a prize",
    "Get rich quick", "Earn money from home", "Cheap loan available", "Exclusive deal just for you",
    "Meeting agenda for Monday", "Project submission deadline", "Let’s catch up tomorrow", "Team lunch on Friday",
    "Client meeting rescheduled", "Budget report due next week", "Update on project status", "Please review the document"
]
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1,  # spam
                   0, 0, 0, 0, 0, 0, 0, 0]) # ham

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and evaluate
nb = MultinomialNaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy (from scratch):", accuracy_score(y_test, y_pred))










#using scikit-learn

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "Buy cheap meds now",         # spam
    "Limited time offer!",        # spam
    "Meeting agenda for Monday", # not spam
    "Project submission deadline", # not spam
    "Win money instantly",       # spam
    "Let’s catch up tomorrow"    # not spam
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text to word count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
