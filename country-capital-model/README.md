# Country Capital Prediction Model (Educational)

This repository contains a Python script for training a simple AI model that can predict the capital city of a country when asked a question like "What is the capital of France?".  This project is designed to be an educational resource for those learning about basic AI/Machine Learning concepts, specifically:

*   **Data Loading and Preprocessing:** How to load data from a JSON file and prepare it for model training.
*   **Natural Language Processing (NLP) Basics:**  Using libraries like SpaCy and NLTK for text tokenization, lemmatization, and stop word removal.
*   **Text Vectorization:**  Understanding and using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors that machine learning models can understand.
*   **Machine Learning Model Training:**  Training a simple Logistic Regression classifier for text classification.
*   **Prediction and Evaluation:**  Using the trained model to make predictions on new questions.
*   **Model Persistence:** Saving and loading trained models for later use.
*   **Command Line Interface (CLI):**  Creating a basic CLI to interact with the script for training and prediction.

## Features

*   **Trains a model** to answer questions about country capitals.
*   **Uses NLP techniques** for preprocessing questions.
*   **Employs TF-IDF** for text vectorization.
*   **Trains a Logistic Regression classifier.**
*   **Saves and loads** the trained model.
*   **Provides a command-line interface** for easy training and prediction.
*   **Includes error handling** and informative messages.

## Learning Objectives

By exploring this repository and running the script, you can learn about:

*   **Data Handling:**  Loading and structuring data from JSON.
*   **Text Preprocessing:**  Techniques for cleaning and preparing text data for NLP tasks.
*   **Feature Engineering:**  Using TF-IDF to create numerical features from text.
*   **Supervised Learning:**  Training a classification model using labeled data.
*   **Model Evaluation (Implicit):** Observing how well the model predicts capitals.
*   **Model Deployment (Basic):**  Saving and loading a model for reuse.
*   **Python Scripting for ML:**  Structuring a Python script for a machine learning workflow.

## Getting Started

### Prerequisites

*   **Python 3.x** installed on your system.
*   **pip** package installer for Python.

### Installation and Setup

1.  **Clone this repository:**
    ```bash
    git clone [repository-url]
    cd country-capital-model
    ```
    (Replace `[repository-url]` with the actual URL of your GitHub repository)

2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install libraries like `scikit-learn`, `nltk`, `spacy`, `fuzzywuzzy`, and `python-Levenshtein`.  If you encounter issues with `spacy`, you might need to download the English language model separately:
    ```bash
    python -m spacy download en_core_web_sm
    ```

3.  **Data File:** Ensure you have the `capitals.json` file in the same directory as `train.py`. This file should be a JSON list of dictionaries, each with "country" and "city" keys (see example in the repository).

### Usage

**1. Training the Model:**

   To train the model, run the `train.py` script in `train` mode, providing the path to your data file:

   ```bash
   python train.py --mode train --data capitals.json