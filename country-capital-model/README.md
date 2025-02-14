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
   ```

   This command will:
   *   Load the country-capital data from `capitals.json`.
   *   Preprocess the data and prepare training examples.
   *   Vectorize the text data using TF-IDF.
   *   Train a Logistic Regression classifier.
   *   Save the trained model to the `saved_model` directory.

**2. Making Predictions:**

   Once the model is trained, you can ask questions about capitals using the `predict` mode:

   ```bash
   python train.py --mode predict --question "What is the capital of France?"
   ```

   Replace `"What is the capital of France?"` with your question. The script will:
   *   Load the trained model from the `saved_model` directory.
   *   Preprocess your question.
   *   Use the trained model to predict the capital.
   *   Print the answer.

**Customizing Data and Model:**

*   **Data:** You can modify the `capitals.json` file to use a different dataset of country-capital pairs. Ensure the JSON format remains consistent.
*   **Model:**  While this script uses `LogisticRegression` for simplicity, you could experiment with other classifiers from `scikit-learn` (e.g., `MLPClassifier`, `SVC`, etc.) by modifying the `CountryCapitalModel` class.
*   **Preprocessing:**  Explore different NLP preprocessing steps or TF-IDF parameters to see how they affect performance.

## Code Explanation (Brief Highlights)

*   **`CountryCapitalModel` class:**  Encapsulates the entire model logic.
*   **`load_data()`:**  Loads country-capital pairs from the JSON file.
*   **`prepare_data()`:** Generates training questions and labels from the country-capital data.
*   **`preprocess_text()`:**  Performs NLP preprocessing on text inputs.
*   **`train()`:**  Trains the machine learning model using TF-IDF and Logistic Regression.
*   **`predict()`:**  Makes predictions for new questions.
*   **`save_model()` and `load_model()`:**  Handle saving and loading the trained model components.
*   **`main()` function:**  Parses command-line arguments and orchestrates the training and prediction processes.

## Further Exploration

*   **Experiment with different datasets:** Try using a larger or more diverse dataset of countries and capitals.
*   **Try different classifiers:**  Replace `LogisticRegression` with other classifiers like Support Vector Machines (SVC), Random Forests, or more complex Neural Networks (MLPClassifier) and compare their performance.
*   **Improve preprocessing:**  Explore more advanced NLP techniques like stemming, handling synonyms, or using word embeddings.
*   **Add evaluation metrics:** Implement metrics like accuracy or precision/recall to quantitatively evaluate the model's performance on a test dataset.
*   **Create a user interface:**  Build a simple web interface or GUI to make the model more interactive.
