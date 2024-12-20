import pandas as pd
import time, os, sys
from logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Redirect console output to a text file
sys.stdout = Logger("logs/model_development_log.txt")

start_time = time.time()
print("Loading the preprocessed dataset...")
# Load the preprocessed dataset
dataset = pd.read_csv("data/selected_features_dataset.csv")
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Specifying features and labels...")
# Specify features and labels
features = dataset[["Message_Length", "Word_Count", "Punctuation_Count", "Capital_Letter_Ratio", "Spam_Keyword_Count"]]
labels = dataset["Label"]
print(f"Features and labels specified in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Splitting the data into training and testing sets...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
print(f"Data split in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Defining the models...")
# Define the models
random_forest_model = RandomForestClassifier(n_estimators=300, random_state=42)
logistic_regression_model = LogisticRegression(random_state=42)
naive_bayes_model = MultinomialNB()
knn_model = KNeighborsClassifier(n_neighbors=5)
print(f"Models defined in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Training the Random Forest model...")
# Train the models
random_forest_model.fit(X_train, y_train)
print(f"Random Forest model trained in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Training the Logistic Regression model...")
logistic_regression_model.fit(X_train, y_train)
print(f"Logistic Regression model trained in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Training the Naive Bayes model...")
naive_bayes_model.fit(X_train, y_train)
print(f"Naive Bayes model trained in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Training the K-Nearest Neighbors model...")
knn_model.fit(X_train, y_train)
print(f"K-Nearest Neighbors model trained in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Making predictions with Random Forest model...")
# Make predictions
random_forest_predictions = random_forest_model.predict(X_test)
print(f"Predictions made with Random Forest model in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Making predictions with Logistic Regression model...")
logistic_regression_predictions = logistic_regression_model.predict(X_test)
print(f"Predictions made with Logistic Regression model in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Making predictions with Naive Bayes model...")
naive_bayes_predictions = naive_bayes_model.predict(X_test)
print(f"Predictions made with Naive Bayes model in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Making predictions with K-Nearest Neighbors model...")
knn_predictions = knn_model.predict(X_test)
print(f"Predictions made with K-Nearest Neighbors model in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Evaluating the Random Forest model...")
# Evaluate the models
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, random_forest_predictions)}")
print(classification_report(y_test, random_forest_predictions))
print(f"Random Forest model evaluated in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Evaluating the Logistic Regression model...")
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, logistic_regression_predictions)}")
print(classification_report(y_test, logistic_regression_predictions))
print(f"Logistic Regression model evaluated in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Evaluating the Naive Bayes model...")
print("Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, naive_bayes_predictions)}")
print(classification_report(y_test, naive_bayes_predictions))
print(f"Naive Bayes model evaluated in {time.time() - start_time:.2f} seconds.\n")

start_time = time.time()
print("Evaluating the K-Nearest Neighbors model...")
print("K-Nearest Neighbors:")
print(f"Accuracy: {accuracy_score(y_test, knn_predictions)}")
print(classification_report(y_test, knn_predictions))
print(f"K-Nearest Neighbors model evaluated in {time.time() - start_time:.2f} seconds.\n")

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

start_time = time.time()
print("Saving the models...")
# Save the models (optional)
import joblib
joblib.dump(random_forest_model, "models/random_forest_model.pkl")
joblib.dump(logistic_regression_model, "models/logistic_regression_model.pkl")
joblib.dump(naive_bayes_model, "models/naive_bayes_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")
print(f"Models saved successfully in {time.time() - start_time:.2f} seconds.\n")

# Close the log file
sys.stdout.log.close()
