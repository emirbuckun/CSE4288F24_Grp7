import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data/enron_dataset.csv")

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
plt.hist(df["Message_Length"], bins=50, alpha=0.7, color='blue')
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# Display the distribution of classes (spam vs ham)
print("\nClass distribution:")
print(df["Label"].value_counts())

# Plot the distribution of classes
df["Label"].value_counts().plot(kind="bar", color=['green', 'red'], alpha=0.7)
plt.title("Class Distribution (Ham vs Spam)")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.show()

# Display the first few rows of the dataset again
print("First few rows of the dataset:")
print(df.head())