import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer
# Load the dataset from your Google Drive
df = pd.read_csv('/content/drive/MyDrive/Day-1(7th april)/PlayTennis.csv')
df.head()
# Check for missing values
print(df.info())
print(df.isnull().sum())

# Countplots for categorical features
# Plot countplots for all feature columns dynamically
cols = df.columns[:-1]
n = len(cols)
rows = (n + 1) // 2  # Automatically calculate rows for 2 columns per row

plt.figure(figsize=(12, 4 * rows))
for i, col in enumerate(cols):
    plt.subplot(rows, 2, i + 1)
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Pairplot to see interactions (optional, mostly useful for numeric)
# sns.pairplot(df, hue='PlayTennis')
# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical features
categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('encoder', OneHotEncoder(), categorical_features)
], remainder='passthrough')

X_encoded = preprocessor.fit_transform(X)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Train Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# For binary classification
y_prob = dt_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


plt.figure(figsize=(16, 10))
plot_tree(dt_classifier,
          feature_names=preprocessor.get_feature_names_out(),
          class_names=label_encoder.classes_,
          filled=True,
          rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt_classifier, X_encoded, y_encoded, cv=strat_kfold, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Average Accuracy:", np.mean(cv_scores))

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='--')
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.grid(True)
plt.show()
