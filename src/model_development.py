import pandas as pd
import time, os, sys
from logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Redirect console output to a text file
sys.stdout = Logger("logs/model_development_log.txt")

start_time = time.time()
print("\n==================== Loading the Preprocessed Dataset ====================")
# Load the preprocessed dataset
dataset = pd.read_csv("data/selected_features_dataset.csv")
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.\n")
load_time = time.time() - start_time

start_time = time.time()
print("\n==================== Specifying Features and Labels ====================")
# Specify features and labels
features = dataset[["Message_Length", "Word_Count", "Punctuation_Count", "Capital_Letter_Ratio", "Spam_Keyword_Count"]]
labels = dataset["Label"]
print(f"Features and labels specified in {time.time() - start_time:.2f} seconds.\n")
specify_time = time.time() - start_time

start_time = time.time()
print("\n==================== Splitting the Data into Training and Testing Sets ====================")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
print(f"Data split in {time.time() - start_time:.2f} seconds.\n")
split_time = time.time() - start_time

start_time = time.time()
print("\n==================== Defining the Models ====================")
# Define the models
random_forest_model = RandomForestClassifier(n_estimators=300, random_state=42)
logistic_regression_model = LogisticRegression(random_state=42)
naive_bayes_model = MultinomialNB()
knn_model = KNeighborsClassifier(n_neighbors=5)
print(f"Models defined in {time.time() - start_time:.2f} seconds.\n")
define_time = time.time() - start_time

start_time = time.time()
print("\n==================== Training the Models ====================")
# Train the models
train_start_time = time.time()
print("Training the Random Forest model...")
random_forest_model.fit(X_train, y_train)
print(f"Random Forest model trained in {time.time() - train_start_time:.2f} seconds.\n")
train_rf_time = time.time() - train_start_time

train_start_time = time.time()
print("Training the Logistic Regression model...")
logistic_regression_model.fit(X_train, y_train)
print(f"Logistic Regression model trained in {time.time() - train_start_time:.2f} seconds.\n")
train_lr_time = time.time() - train_start_time

train_start_time = time.time()
print("Training the Naive Bayes model...")
naive_bayes_model.fit(X_train, y_train)
print(f"Naive Bayes model trained in {time.time() - train_start_time:.2f} seconds.\n")
train_nb_time = time.time() - train_start_time

train_start_time = time.time()
print("Training the K-Nearest Neighbors model...")
knn_model.fit(X_train, y_train)
print(f"K-Nearest Neighbors model trained in {time.time() - train_start_time:.2f} seconds.\n")
train_knn_time = time.time() - train_start_time

start_time = time.time()
print("\n==================== Making Predictions ====================")
# Make predictions
predict_start_time = time.time()
print("Making predictions with Random Forest model...")
random_forest_predictions = random_forest_model.predict(X_test)
print(f"Predictions made with Random Forest model in {time.time() - predict_start_time:.2f} seconds.\n")
predict_rf_time = time.time() - predict_start_time

predict_start_time = time.time()
print("Making predictions with Logistic Regression model...")
logistic_regression_predictions = logistic_regression_model.predict(X_test)
print(f"Predictions made with Logistic Regression model in {time.time() - predict_start_time:.2f} seconds.\n")
predict_lr_time = time.time() - predict_start_time

predict_start_time = time.time()
print("Making predictions with Naive Bayes model...")
naive_bayes_predictions = naive_bayes_model.predict(X_test)
print(f"Predictions made with Naive Bayes model in {time.time() - predict_start_time:.2f} seconds.\n")
predict_nb_time = time.time() - predict_start_time

predict_start_time = time.time()
print("Making predictions with K-Nearest Neighbors model...")
knn_predictions = knn_model.predict(X_test)
print(f"Predictions made with K-Nearest Neighbors model in {time.time() - predict_start_time:.2f} seconds.\n")
predict_knn_time = time.time() - predict_start_time

start_time = time.time()
print("\n==================== Evaluating the Models ====================")
# Evaluate the models
evaluate_start_time = time.time()
print("Evaluating the Random Forest model...")
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, random_forest_predictions)}")
print(classification_report(y_test, random_forest_predictions))
print(f"Random Forest model evaluated in {time.time() - evaluate_start_time:.2f} seconds.\n")
evaluate_rf_time = time.time() - evaluate_start_time

evaluate_start_time = time.time()
print("Evaluating the Logistic Regression model...")
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, logistic_regression_predictions)}")
print(classification_report(y_test, logistic_regression_predictions))
print(f"Logistic Regression model evaluated in {time.time() - evaluate_start_time:.2f} seconds.\n")
evaluate_lr_time = time.time() - evaluate_start_time

evaluate_start_time = time.time()
print("Evaluating the Naive Bayes model...")
print("Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, naive_bayes_predictions)}")
print(classification_report(y_test, naive_bayes_predictions))
print(f"Naive Bayes model evaluated in {time.time() - evaluate_start_time:.2f} seconds.\n")
evaluate_nb_time = time.time() - evaluate_start_time

evaluate_start_time = time.time()
print("Evaluating the K-Nearest Neighbors model...")
print("K-Nearest Neighbors:")
print(f"Accuracy: {accuracy_score(y_test, knn_predictions)}")
print(classification_report(y_test, knn_predictions))
print(f"K-Nearest Neighbors model evaluated in {time.time() - evaluate_start_time:.2f} seconds.\n")
evaluate_knn_time = time.time() - evaluate_start_time

# Define a function to perform grid search and evaluate models
def evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search_start_time = time.time()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Grid search and evaluation completed in {time.time() - grid_search_start_time:.2f} seconds.\n")
    return best_model, accuracy, report

# Define a function to perform randomized search and evaluate models
def evaluate_model_randomized(model, param_distributions, X_train, y_train, X_test, y_test, n_iter=100):
    randomized_search_start_time = time.time()
    randomized_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    randomized_search.fit(X_train, y_train)
    best_model = randomized_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Randomized search and evaluation completed in {time.time() - randomized_search_start_time:.2f} seconds.\n")
    return best_model, accuracy, report

# Experiment with different algorithms and hyperparameters
print("\n==================== Optimizing Models ====================")
optimize_start_time = time.time()
print("Optimizing Random Forest model...")
best_rf_model, rf_accuracy, rf_report = evaluate_model(RandomForestClassifier(random_state=42), {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}, X_train, y_train, X_test, y_test)
optimize_rf_time = time.time() - optimize_start_time
print(f"Best Random Forest model accuracy: {rf_accuracy}")
print(rf_report)

optimize_start_time = time.time()
print("Optimizing Logistic Regression model...")
best_lr_model, lr_accuracy, lr_report = evaluate_model(LogisticRegression(random_state=42), {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [300, 400, 500]
}, X_train, y_train, X_test, y_test)
optimize_lr_time = time.time() - optimize_start_time
print(f"Best Logistic Regression model accuracy: {lr_accuracy}")
print(lr_report)

optimize_start_time = time.time()
print("Optimizing Naive Bayes model...")
best_nb_model, nb_accuracy, nb_report = evaluate_model(MultinomialNB(), {
    'alpha': [0.01, 0.1, 1, 10]
}, X_train, y_train, X_test, y_test)
optimize_nb_time = time.time() - optimize_start_time
print(f"Best Naive Bayes model accuracy: {nb_accuracy}")
print(nb_report)

optimize_start_time = time.time()
print("Optimizing K-Nearest Neighbors model...")
best_knn_model, knn_accuracy, knn_report = evaluate_model(KNeighborsClassifier(), {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}, X_train, y_train, X_test, y_test)
optimize_knn_time = time.time() - optimize_start_time
print(f"Best K-Nearest Neighbors model accuracy: {knn_accuracy}")
print(knn_report)

# Experiment with different algorithms and hyperparameters using RandomizedSearchCV
print("Optimizing Random Forest model with RandomizedSearchCV...")
best_rf_model, rf_accuracy, rf_report = evaluate_model_randomized(RandomForestClassifier(random_state=42), {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'bootstrap': [True, False]
}, X_train, y_train, X_test, y_test)
print(f"Best Random Forest model accuracy: {rf_accuracy}")
print(rf_report)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

start_time = time.time()
print("\n==================== Saving the Models ====================")
# Save the models (optional)
import joblib
joblib.dump(random_forest_model, "models/random_forest_model.pkl")
joblib.dump(logistic_regression_model, "models/logistic_regression_model.pkl")
joblib.dump(naive_bayes_model, "models/naive_bayes_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")
print(f"Models saved successfully in {time.time() - start_time:.2f} seconds.\n")
save_models_time = time.time() - start_time

# Save the best models
print("Saving the best models...")
joblib.dump(best_rf_model, "models/best_random_forest_model.pkl")
joblib.dump(best_lr_model, "models/best_logistic_regression_model.pkl")
joblib.dump(best_nb_model, "models/best_naive_bayes_model.pkl")
joblib.dump(best_knn_model, "models/best_knn_model.pkl")
print("Best models saved successfully.")

# Ensure the predictions directory exists
os.makedirs("predictions", exist_ok=True)

# Save prediction results
start_time = time.time()
print("\n==================== Saving Prediction Results ====================")
# Save Random Forest predictions
rf_results = pd.DataFrame({"Actual": y_test, "Predicted": random_forest_predictions})
rf_results.to_csv("predictions/random_forest_predictions.csv", index=False)
print("Random Forest predictions saved.")

# Save Logistic Regression predictions
lr_results = pd.DataFrame({"Actual": y_test, "Predicted": logistic_regression_predictions})
lr_results.to_csv("predictions/logistic_regression_predictions.csv", index=False)
print("Logistic Regression predictions saved.")

# Save Naive Bayes predictions
nb_results = pd.DataFrame({"Actual": y_test, "Predicted": naive_bayes_predictions})
nb_results.to_csv("predictions/naive_bayes_predictions.csv", index=False)
print("Naive Bayes predictions saved.")

# Save K-Nearest Neighbors predictions
knn_results = pd.DataFrame({"Actual": y_test, "Predicted": knn_predictions})
knn_results.to_csv("predictions/knn_predictions.csv", index=False)
print("K-Nearest Neighbors predictions saved.")

print(f"Prediction results saved successfully in {time.time() - start_time:.2f} seconds.\n")
save_predictions_time = time.time() - start_time

# Print summary of model evaluations and optimizations
print("\n==================== Summary of Model Evaluations and Optimizations ====================")
print("\nInitial Model Evaluations:")
print("----------------------------------------------------------------------------------------")
print(f"Random Forest Accuracy: {accuracy_score(y_test, random_forest_predictions)}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logistic_regression_predictions)}")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, naive_bayes_predictions)}")
print(f"K-Nearest Neighbors Accuracy: {accuracy_score(y_test, knn_predictions)}")

print("\nOptimized Model Evaluations:")
print("----------------------------------------------------------------------------------------")
print(f"Best Random Forest Accuracy: {rf_accuracy}")
print(f"Best Logistic Regression Accuracy: {lr_accuracy}")
print(f"Best Naive Bayes Accuracy: {nb_accuracy}")
print(f"Best K-Nearest Neighbors Accuracy: {knn_accuracy}")
print("========================================================================================")

# Print summary of times taken by each operation
print("\n==================== Summary of Times Taken by Each Operation ====================")
print(f"Loading dataset: {load_time:.2f} seconds")
print(f"Specifying features and labels: {specify_time:.2f} seconds")
print(f"Splitting data: {split_time:.2f} seconds")
print(f"Defining models: {define_time:.2f} seconds")
print(f"Training Random Forest model: {train_rf_time:.2f} seconds")
print(f"Training Logistic Regression model: {train_lr_time:.2f} seconds")
print(f"Training Naive Bayes model: {train_nb_time:.2f} seconds")
print(f"Training K-Nearest Neighbors model: {train_knn_time:.2f} seconds")
print(f"Making predictions with Random Forest model: {predict_rf_time:.2f} seconds")
print(f"Making predictions with Logistic Regression model: {predict_lr_time:.2f} seconds")
print(f"Making predictions with Naive Bayes model: {predict_nb_time:.2f} seconds")
print(f"Making predictions with K-Nearest Neighbors model: {predict_knn_time:.2f} seconds")
print(f"Evaluating Random Forest model: {evaluate_rf_time:.2f} seconds")
print(f"Evaluating Logistic Regression model: {evaluate_lr_time:.2f} seconds")
print(f"Evaluating Naive Bayes model: {evaluate_nb_time:.2f} seconds")
print(f"Evaluating K-Nearest Neighbors model: {evaluate_knn_time:.2f} seconds")
print(f"Optimizing Random Forest model: {optimize_rf_time:.2f} seconds")
print(f"Optimizing Logistic Regression model: {optimize_lr_time:.2f} seconds")
print(f"Optimizing Naive Bayes model: {optimize_nb_time:.2f} seconds")
print(f"Optimizing K-Nearest Neighbors model: {optimize_knn_time:.2f} seconds")
print(f"Saving models: {save_models_time:.2f} seconds")
print(f"Saving predictions: {save_predictions_time:.2f} seconds")
print("========================================================================================")

# Close the log file
sys.stdout.log.close()
