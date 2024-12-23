# CSE4288 Introduction to Machine Learning – Team Project – Fall 2024

## Project Overview

This project is part of the CSE4288 course for Fall 2024. The objective is to provide hands-on experience in applying machine learning techniques to solve real-world problems. Our team will work on spam detection, classifying emails as spam or not spam using the [Enron dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam).

## Project Structure

```
CSE4288F24_Grp7/
├── data/                           # Data directory
├── docs/                           # Documentation files
├── images/                         # Images from analyze
├── logs/                           # Logs for the scripts' outputs
├── models/                         # Contains models
├── predictions/                    # Contains model predictions
├── src/                            # Source directory
│   ├── analyze_data.py             # Analyzing script
│   ├── logger.py                   # Logger for console and file
│   ├── model_development.py        # Model development script
│   ├── preprocess_data.py          # Preprocessing script
├── .gitignore                      # Ignore unnecessary files
└── README.md                       # Project overview
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

3. **Model development:**

```bash
    python src/model_development.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
