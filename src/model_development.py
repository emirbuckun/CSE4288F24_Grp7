import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed dataset
df = pd.read_csv("data/selected_features_dataset.csv")

# Specify features and labels
X = df[["Message_Length", "Word_Count", "Punctuation_Count", "Capital_Letter_Ratio", "Spam_Keyword_Count"]]
y = df["Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the model
model = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the model
model.fit(X_train, y_train)







