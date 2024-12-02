# CSE4288 Introduction to Machine Learning – Team Project – Fall 2024

## Project Overview

This project is part of the CSE4288 course for Fall 2024. The objective is to provide hands-on experience in applying machine learning techniques to solve real-world problems. Our team will work on spam detection, classifying emails as spam or not spam using the [Enron dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam).

## Project Structure

```
CSE4288F24_Grp7/
├── data/                            # Data directory
│   ├── enron-spam.zip               # Enron spam data (not included in repo since contains large files)
├── docs/                            # Documentation files
│   ├── proposal.pdf                 # Proposal of the project
│   ├── term_project.pdf             # Term project document
├── src/                             # Source directory
│   ├── preprocess_data.py           # Preprocessing script
│   ├── analyze_data.py              # Analyzing script
├── .gitignore                       # Ignore unnecessary files
└── README.md                        # Project overview
```

## Installation Guide

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Setup Instructions

1. **Clone the repository:**

```bash
    git clone https://github.com/emirbuckun/CSE4288F24_Grp7.git
```

2. **Create a virtual environment:**

```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install the required dependencies:**

```bash
    pip install -r requirements.txt
```

4. **Download the dataset:**

- Download the enron-spam.zip file from the provided source and place it to the directory named "data" in the project root directory.
- [Click to download the data](https://drive.google.com/drive/folders/1RKMYE1GoeCbj6Y3QbkSqymXuWpeK2PAn?usp=sharing)

## Running the Project

1. **Preprocess the data:**

```bash
    python src/preprocess_data.py
```

2. **Analyze the data:**

```bash
    python src/analyze_data.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
