import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot the distribution of message lengths
plt.figure(figsize=(10, 6))
sns.histplot(df["Message_Length"], bins=50, kde=True, color='blue')
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# Display the distribution of classes (spam vs ham)
print("\nClass distribution:")
print(df["Label"].value_counts())

# Plot the distribution of classes
plt.figure(figsize=(8, 5))
sns.countplot(x="Label", data=df, palette="viridis")
plt.title("Class Distribution (Ham vs Spam)")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
plt.show()

# Analyze the relationship between message length and label
plt.figure(figsize=(10, 6))
sns.boxplot(x="Label", y="Message_Length", data=df, palette="muted")
plt.title("Message Length by Class (Ham vs Spam)")
plt.xlabel("Label")
plt.ylabel("Message Length")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
plt.show()

# Word frequency analysis for clean messages
print("\nTop 10 most frequent words in cleaned messages:")
from collections import Counter
word_counter = Counter(" ".join(df["Clean_Message"]).split())
most_common_words = word_counter.most_common(10)
print(most_common_words)

# Visualize top 10 words
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette="coolwarm")
plt.title("Top 10 Words in Cleaned Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()