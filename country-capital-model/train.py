#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Country Capital Prediction Model - Educational Script

This script demonstrates how to train a simple AI model to predict the capital city
of a country based on questions. It's designed for educational purposes to illustrate:

- Data Loading and Preprocessing
- Text Vectorization (TF-IDF)
- Machine Learning Model Training (Logistic Regression)
- Prediction and Evaluation
- Model Saving and Loading

This script is part of a learning resource and is intentionally kept relatively simple
for clarity and educational value.

Author: [Your Name or GitHub Username]
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
import json
import spacy
import re
import pickle
import os
from dataclasses import dataclass, asdict
from pathlib import Path

def check_requirements():
    """
    Checks if required Python packages are installed and installs them if missing.
    Currently, it checks for 'python-Levenshtein' and installs it using pip if not found.
    This is done to improve fuzzy matching performance.
    """
    try:
        import Levenshtein
    except ImportError:
        print("Installing python-Levenshtein for better performance...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "python-Levenshtein"])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install python-Levenshtein automatically. Please try installing it manually using: `pip install python-Levenshtein`. Error details: {e}")
        except FileNotFoundError:
            raise RuntimeError("`pip` command not found. Please ensure pip is installed and in your system's PATH, or install python-Levenshtein manually.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during python-Levenshtein installation: {e}. You may need to install it manually.")

@dataclass
class ProcessedText:
    """
    Data class to hold processed text, extracted entities, and SpaCy Doc object.
    """
    processed_text: str
    entities: List[str]
    doc: Any

class CountryCapitalModel:
    """
    Country Capital Prediction Model.

    This class encapsulates the model's functionalities: loading data, preprocessing text,
    training a machine learning classifier, making predictions, and saving/loading the model.
    """
    def __init__(self, data_file: Optional[str] = None):
        """Initialize model with improved matching"""
        check_requirements() # Ensure required packages are installed

        # Initialize class variables
        self.countries: List[str] = []
        self.capitals: List[str] = []
        self.data_file = data_file
        self.is_trained = False
        self.country_capital_pairs: Dict[str, str] = {} # Dictionary to store country-capital pairs
        self.normalized_countries = {}
        self.country_variations = {}
        self.capital_label_encoder = LabelEncoder() # LabelEncoder to encode capital names into numerical labels
        self.encoded_capitals_list: List[str] = [] # List to store encoded capital names in order
        self.capital_to_country_map: Dict[str, str] = {} # Dictionary for reverse lookup: capital -> country

        # Initialize NLP components using SpaCy for text processing
        try:
            self.nlp = spacy.load('en_core_web_sm') # Load small English SpaCy model
        except OSError:
            raise RuntimeError("SpaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")

        self.lemmatizer = WordNetLemmatizer() # Lemmatizer to reduce words to their base form
        self.stop_words = set(stopwords.words('english')) # Set of common English stop words

        # Initialize ML components: TF-IDF Vectorizer and Logistic Regression Classifier
        self.vectorizer = TfidfVectorizer( # TF-IDF to convert text questions into numerical vectors
            max_features=1000, # Consider only top 1000 features to limit vocabulary size
            ngram_range=(1, 2), # Use unigrams and bigrams
            stop_words='english' # Remove common English stop words during vectorization
        )
        self.classifier = LogisticRegression( # Logistic Regression classifier for prediction
            random_state=42, # For reproducibility
            max_iter=300 # Maximum iterations for training
        )

        if data_file:
            self.load_data() # Load data from the specified JSON file
            self.normalized_countries = { # Create normalized country names for matching
                self.normalize_text(country): country
                for country in self.country_capital_pairs.keys()
            }
            self.country_variations = self._generate_country_variations() # Generate variations of country names
            self.capital_to_country_map = {capital: country for country, capital in self.country_capital_pairs.items()} # Create capital to country reverse map


    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching: lowercase, remove punctuation, and extra spaces."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower()) # Remove non-alphabetic characters
        return ' '.join(text.split()) # Remove extra spaces

    def _generate_country_variations(self) -> Dict[str, str]:
        """Generate common variations of country names for improved matching."""
        variations = {}
        prefixes_to_remove = ['the ', 'republic of ', 'democratic republic of '] # Prefixes to remove for variations

        for country in self.country_capital_pairs.keys():
            normalized = self.normalize_text(country)
            variations[normalized] = country # Add normalized name as variation

            for prefix in prefixes_to_remove:
                if normalized.startswith(prefix):
                    variations[normalized.replace(prefix, '').strip()] = country # Add variation without prefix

        return variations

    def load_data(self) -> None:
        """Load country-capital pairs from a JSON file with error handling."""
        if not self.data_file:
            raise ValueError("No data file specified. Please provide a path to a JSON data file using --data.")

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f) # Load JSON data from file

            if not isinstance(data, list): # Expecting a list of dictionaries
                raise ValueError(f"Expected data in JSON list format (array of objects), but found: {type(data)}. Please ensure your JSON file contains a list of country-capital pairs.")

            if not data: # Check if the loaded list is empty
                raise ValueError("The provided JSON data file is empty or contains no data.")

            pairs = {}
            for item in data:
                if not isinstance(item, dict):
                    print(f"Warning: Skipping invalid entry (not a dictionary): {item}. Expected format: {{'country': 'Country Name', 'city': 'Capital City'}}")
                    continue

                country = item.get('country')
                capital = item.get('city')

                if not isinstance(country, str) or not isinstance(capital, str):
                    print(f"Warning: Skipping invalid entry (country or city not a string): {item}. Expected format: {{'country': 'Country Name', 'city': 'Capital City'}}")
                    continue

                country = country.strip()
                capital = capital.strip()
                if country and capital:
                    pairs[country] = capital
                else:
                    print(f"Warning: Skipping entry with empty country or capital name: {item}")


            if not pairs:
                raise ValueError("No valid country-capital pairs found in the data after processing. Please check the format and content of your JSON file.")

            self.country_capital_pairs = pairs # Store loaded country-capital pairs
            return

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {self.data_file}: {str(e)}. Please ensure the file is valid JSON.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {self.data_file}. Please check the file path.")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.data_file}: {str(e)}")

    def prepare_data(self) -> Tuple[List[str], List[str]]:
        """Prepare training data by generating questions and corresponding capital labels."""
        X_data = [] # List to store question texts
        y_data = [] # List to store capital labels
        ordered_capitals = [] # List to maintain order of capitals for label encoding

        question_patterns = [ # Question patterns to generate training examples
            "What is the capital of {}?",
            "Which city serves as {}'s capital?"
        ]

        for country, capital in self.country_capital_pairs.items():
            if not isinstance(country, str) or not isinstance(capital, str): # Skip invalid entries
                continue

            country = country.strip()
            capital = capital.strip()

            if not country or not capital: # Skip empty country or capital names
                continue

            for pattern in question_patterns:
                question = pattern.format(country) # Create question using pattern and country name
                processed = self.preprocess_text(question) # Preprocess the question text
                if processed and processed.processed_text: # Check if preprocessing was successful and text is not empty
                    X_data.append(processed.processed_text) # Add processed question to training data
                    y_data.append(capital) # Add capital name as label
                    ordered_capitals.append(capital) # Maintain order of capitals
                elif processed:
                    print(f"Warning: Preprocessed text is empty for question: '{question}'. Skipping this example.")
                else:
                    print(f"Warning: Preprocessing failed for question: '{question}'. Skipping this example.")


        if not X_data or not y_data:
            raise ValueError("No valid training data generated after preprocessing. Please check your data and preprocessing steps.")

        return X_data, y_data, ordered_capitals # Return question texts, capital labels, and ordered capitals

    def preprocess_text(self, text: str) -> ProcessedText:
        """Preprocess text for training and prediction: normalize, tokenize, lemmatize, remove stop words and punctuation."""
        text = str(text) # Ensure input is string
        text = self.normalize_text(text) # Normalize text
        doc = self.nlp(text) # Process text with SpaCy NLP pipeline

        tokens = [
            str(self.lemmatizer.lemmatize(token.text)) # Lemmatize each token
            for token in doc
            if not token.is_stop and not token.is_punct # Remove stop words and punctuation
        ]

        entities = [
            str(ent.text) for ent in doc.ents # Extract named entities
            if ent.label_ in ['GPE', 'LOC'] # Consider only Geographical locations (GPE) and Locations (LOC)
        ]

        return ProcessedText(
            processed_text=' '.join(tokens), # Join processed tokens back into a string
            entities=entities, # List of extracted entities
            doc=doc # SpaCy Doc object for further analysis if needed
        )

    def train(self) -> None:
        """Train the model: prepare data, vectorize text, encode labels, and train the classifier."""
        if not self.country_capital_pairs:
            raise ValueError("No training data available. Please load data first using `--data` argument and ensure the data file is correctly formatted and contains valid country-capital pairs.")

        print("Preparing training data...")
        X_data, y_data, ordered_capitals = self.prepare_data() # Prepare data and get ordered capitals

        print(f"Generated {len(X_data)} training examples")

        try:
            print("Vectorizing text data...")
            X_vectors = self.vectorizer.fit_transform(X_data) # Vectorize question texts using TF-IDF
            X_dense = X_vectors.toarray() # Convert sparse matrix to dense array (for Logistic Regression)

            # Encode the labels (capitals), fitting on the ordered list to ensure consistent mapping
            print("Encoding labels (capitals)...")
            self.capital_label_encoder.fit(ordered_capitals) # Fit LabelEncoder on ordered capital names
            y_encoded = self.capital_label_encoder.transform(y_data) # Transform capital names to numerical labels
            self.encoded_capitals_list = list(self.capital_label_encoder.classes_) # Store encoded capital names in order

            print("Training classifier...")
            self.classifier.fit(X_dense, y_encoded) # Train Logistic Regression classifier
            self.is_trained = True # Mark model as trained

            print("Training completed successfully")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(self, question: str) -> str:
        """Make a prediction: preprocess question, vectorize, predict encoded label, decode label, and find country."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        processed = self.preprocess_text(question) # Preprocess the input question

        try:
            X_vector = self.vectorizer.transform([processed.processed_text]) # Vectorize the preprocessed question
            X_dense = X_vector.toarray()

            # Get encoded prediction and decode back to capital name using the ordered list
            predicted_encoded = self.classifier.predict(X_dense)[0] # Predict encoded label
            predicted_capital = self.encoded_capitals_list[predicted_encoded] # Decode label using encoded_capitals_list

            # Find corresponding country using the capital_to_country_map
            country = self.capital_to_country_map.get(predicted_capital)
            if country:
                return f"The capital of {country} is {predicted_capital}."
            else:
                print(f"Warning: Predicted capital '{predicted_capital}' not found in capital_to_country_map.") # Debugging
                # Fallback to fuzzy matching in case of "Bucuresti" vs "Bucharest" issue.
                for capital_db, country_db in self.capital_to_country_map.items():
                    if fuzz.ratio(predicted_capital, capital_db) > 90: # High threshold fuzzy match
                        return f"The capital of {country_db} is {capital_db} (fuzzy matched)."
                return f"The capital of {country} is {predicted_capital}." # Return best effort if country not found directly


        except Exception as e:
            print(f"ML prediction failed: {str(e)}, falling back to rule-based matching. Error: {e}")

            # Fallback to rule-based matching using entity recognition and fuzzy matching
            for entity in processed.entities:
                country = self.find_closest_country(entity)
                if country:
                    return f"The capital of {country} is {self.country_capital_pairs[country]}."

        return "I'm sorry, I couldn't identify the country in your question with sufficient confidence."


    def find_closest_country(self, query: str) -> Optional[str]:
        """Find closest matching country using fuzzy matching."""
        query = self.normalize_text(query) # Normalize the query text

        # Direct match in variations
        if query in self.country_variations:
            return self.country_variations[query]

        # Fuzzy matching to find the closest country name variation
        best_match = None
        highest_score = 0

        for variation, country in self.country_variations.items():
            score = fuzz.ratio(query, variation) # Calculate fuzzy ratio between query and country variation
            if score > highest_score:
                highest_score = score
                best_match = country # Update best match if score is higher

        return best_match if highest_score > 70 else None # Return best match if score is above threshold, otherwise None

    def save_model(self, directory: str = 'saved_model') -> None:
        """Save the trained model components to a directory."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(directory, exist_ok=True) # Create directory if it doesn't exist

        try:
            # Save ML components using pickle for vectorizer and classifier
            with open(os.path.join(directory, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)

            with open(os.path.join(directory, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)

            # Save label encoder and encoded capitals list using pickle
            with open(os.path.join(directory, 'capital_label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.capital_label_encoder, f)

            with open(os.path.join(directory, 'encoded_capitals_list.pkl'), 'wb') as f:
                pickle.dump(self.encoded_capitals_list, f)


            # Save other model data (country-capital pairs, variations, etc.) as JSON
            model_data = {
                'country_capital_pairs': self.country_capital_pairs,
                'normalized_countries': self.normalized_countries,
                'country_variations': self.country_variations,
                'is_trained': self.is_trained
            }

            with open(os.path.join(directory, 'model_data.json'), 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, directory: str = 'saved_model') -> 'CountryCapitalModel':
        """Load a trained model and all components from a directory."""
        instance = cls() # Create instance of the class

        try:
            # Load ML components using pickle
            with open(os.path.join(directory, 'vectorizer.pkl'), 'rb') as f:
                instance.vectorizer = pickle.load(f)

            with open(os.path.join(directory, 'classifier.pkl'), 'rb') as f:
                instance.classifier = pickle.load(f)

            # Load label encoder and encoded capitals list using pickle
            with open(os.path.join(directory, 'capital_label_encoder.pkl'), 'rb') as f:
                instance.capital_label_encoder = pickle.load(f)

            with open(os.path.join(directory, 'encoded_capitals_list.pkl'), 'rb') as f:
                instance.encoded_capitals_list = pickle.load(f)


            # Load other model data from JSON
            with open(os.path.join(directory, 'model_data.json'), 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            instance.country_capital_pairs = model_data['country_capital_pairs']
            instance.normalized_countries = model_data['normalized_countries']
            instance.country_variations = model_data['country_variations']
            instance.is_trained = model_data['is_trained']
            instance.capital_to_country_map = {capital: country for country, capital in instance.country_capital_pairs.items()} # Rebuild reverse map

            return instance # Return loaded model instance

        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

def main():
    """Main function to handle command-line interface for training and prediction."""
    import argparse

    parser = argparse.ArgumentParser(description='Country Capital Model')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode of operation: train or predict')
    parser.add_argument('--data', type=str, help='Path to training data file (JSON)')
    parser.add_argument('--model-dir', type=str, default='saved_model',
                      help='Directory for model storage/loading (default: saved_model)')
    parser.add_argument('--question', type=str, help='Question for prediction (in predict mode)')

    args = parser.parse_args()

    try:
        if args.mode == 'train': # Training mode
            if not args.data:
                raise ValueError("Training mode requires --data argument to specify the training data file.")

            print(f"Loading data from {args.data}...")
            model = CountryCapitalModel(args.data) # Initialize model with data file

            print("Training model...")
            model.train() # Train the model

            print(f"Saving model to {args.model_dir}...")
            model.save_model(args.model_dir) # Save the trained model
            print(f"Model trained and saved to {args.model_dir}")

        else:  # Prediction mode
            if not args.question:
                raise ValueError("Prediction mode requires --question argument to ask a question.")

            if not os.path.exists(args.model_dir):
                raise ValueError(f"Model directory {args.model_dir} not found. Please train the model first using 'train' mode.")

            print(f"Loading model from {args.model_dir}...")
            model = CountryCapitalModel.load_model(args.model_dir) # Load trained model

            print("Making prediction...")
            answer = model.predict(args.question) # Get prediction for the question
            print(f"\nQ: {args.question}")
            print(f"A: {answer}") # Print question and answer

    except Exception as e:
        print(f"Error: {str(e)}") # Print error message
        return 1 # Indicate error exit

    return 0 # Indicate successful exit

if __name__ == "__main__":
    exit(main()) # Execute main function when script is run