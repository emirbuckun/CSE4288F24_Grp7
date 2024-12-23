import pandas as pd, numpy as np, string, os
import matplotlib.pyplot as plt, seaborn as sns
import sys
from logger import Logger

# Redirect console output to a text file
sys.stdout = Logger("logs/analyze_log.txt")

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
df = pd.read_csv("data/cleaned_enron_dataset.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset info:")
print(df.info())

# Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Add a new column for the length of each message
df["Message_Length"] = df["Message"].apply(len)

# Display basic statistics for the message length
print("\nMessage Length Statistics:")
print(df["Message_Length"].describe())

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)

# Plot the distribution of message lengths
plt.figure(figsize=(10, 6))
sns.histplot(df["Message_Length"], bins=50, kde=True, color='blue')
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.savefig("images/distribution_of_message_lengths.png")
plt.close()  # Close the plot instead of showing it

# Display the distribution of classes (spam vs ham)
print("\nClass distribution:")
print(df["Label"].value_counts())

# Plot the distribution of classes
plt.figure(figsize=(8, 5))
sns.countplot(x="Label", data=df, palette="viridis", hue="Label", legend=False)
plt.title("Class Distribution (Ham vs Spam)")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
plt.savefig("images/distribution_of_classes.png")
plt.close()  # Close the plot instead of showing it

# Analyze the relationship between message length and label
plt.figure(figsize=(10, 6))
sns.boxplot(x="Label", y="Message_Length", data=df, palette="muted", hue="Label", legend=False)
plt.title("Message Length by Class (Ham vs Spam)")
plt.xlabel("Label")
plt.ylabel("Message Length")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
plt.savefig("images/relationship_between_length_and_label.png")
plt.close()  # Close the plot instead of showing it

# Word frequency analysis for clean messages
print("\nTop 10 most frequent words in cleaned messages:")
word_counter = Counter(" ".join(df["Clean_Message"]).split())
most_common_words = word_counter.most_common(10)
print(most_common_words)

# Visualize top 10 words
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette="coolwarm", hue=list(counts), legend=False)
plt.title("Top 10 Words in Cleaned Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.savefig("images/top_10.png")
plt.close()  # Close the plot instead of showing it

# Calculate the word count in the clean message
df["Word_Count"] = df["Clean_Message"].apply(lambda x: len(x.split()))

# Count punctuation marks in the original message
df["Punctuation_Count"] = df["Message"].apply(lambda x: sum([1 for char in x if char in string.punctuation]))

# Calculate the ratio of capital letters to total letters
df["Capital_Letter_Ratio"] = df["Message"].apply(
    lambda x: sum(1 for char in x if char.isupper()) / (len(x) + 1e-5)
)

# Filter spam and ham messages
spam_messages = df[df["Label"] == 1]["Clean_Message"]
ham_messages = df[df["Label"] == 0]["Clean_Message"]

# Use TfidfVectorizer to identify important keywords in spam messages
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words="english")  # Limit to top 50 terms
tfidf_matrix_spam = tfidf_vectorizer.fit_transform(spam_messages)

# Get feature names (keywords) and their importance
spam_keywords_scores = zip(tfidf_vectorizer.get_feature_names_out(), np.asarray(tfidf_matrix_spam.sum(axis=0)).flatten())
spam_keywords_sorted = sorted(spam_keywords_scores, key=lambda x: x[1], reverse=True)

# Extract top 10 spam-indicative keywords
spam_keywords = [keyword for keyword, score in spam_keywords_sorted[:10]]
print("\nTop Spam Keywords:", spam_keywords)

# Add Spam Keyword Count Feature using the dynamically generated spam keywords
df["Spam_Keyword_Count"] = df["Clean_Message"].apply(
    lambda x: sum(word in x for word in spam_keywords)
)

# Display a preview of the new feature
print("\nSpam Keyword Count Preview:")
print(df[["Clean_Message", "Spam_Keyword_Count"]].head())

# Display first few rows of engineered features
print("\nEngineered Features:")
print(df[["Message_Length", "Word_Count", "Punctuation_Count", "Capital_Letter_Ratio", "Spam_Keyword_Count"]].head())

# Select relevant features for the model
selected_features = df[["Clean_Message", "Message_Length", "Word_Count", "Punctuation_Count", "Capital_Letter_Ratio", "Spam_Keyword_Count", "Label"]]

# Save selected features to a new CSV
selected_features.to_csv("data/selected_features_dataset.csv", index=False, encoding="utf-8")
print("\nSelected features saved to 'data/selected_features_dataset.csv'.")

# Close the log file
sys.stdout.log.close()
