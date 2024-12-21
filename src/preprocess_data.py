import os, re, zipfile, pandas as pd, nltk
from nltk.corpus import stopwords
import sys
from logger import Logger

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Redirect console output to a text file
sys.stdout = Logger("logs/preprocess_log.txt")

# Define paths
zip_path = "data/enron-spam.zip"
extract_to_path = "data/enron_data"

# Extract only the enron1 directory from the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if member.startswith("enron1/"):
            zip_ref.extract(member, extract_to_path)
print(f"Files from 'enron1' have been extracted to '{extract_to_path}' folder.")

# Define folder paths for ham and spam messages
ham_path = os.path.join(extract_to_path, "enron1", "ham")
spam_path = os.path.join(extract_to_path, "enron1", "spam")

data = []

# Function to load messages from a directory
def load_messages(directory, label):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            message = f.read()
            data.append({"Message": message, "Label": label})

# Load ham and spam messages
load_messages(ham_path, "ham")
load_messages(spam_path, "spam")

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Save DataFrame to CSV file
df.to_csv("data/enron_dataset.csv", index=False, encoding="utf-8", escapechar="\\")
print("Data has been saved to 'enron_dataset.csv'.")

# Data Cleaning
# Convert labels to numerical values
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

# Check and clean missing data
print("Missing values per column:")
print(df.isnull().sum())
df.dropna(inplace=True)

# Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

# Function to clean messages
def clean_message(message):
    message = re.sub(r"[^a-zA-Z]", " ", message)  # Keep only letters
    message = message.lower()  # Convert to lowercase
    message = " ".join(word for word in message.split() if word not in stop_words)  # Remove stopwords
    return message

# Clean messages
df["Clean_Message"] = df["Message"].apply(clean_message)

# Display cleaned messages
print(df[["Message", "Clean_Message"]].head())

# Reorder columns
df = df[["Message", "Clean_Message", "Label"]]

# Save cleaned data to a new CSV file
df.to_csv("data/cleaned_enron_dataset.csv", index=False, encoding="utf-8", escapechar="\\")
print("Cleaned data has been saved to 'cleaned_enron_dataset.csv'.")

# Close the log file
sys.stdout.log.close()
