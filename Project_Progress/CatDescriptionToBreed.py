import numpy as np
import pandas as pd
import google.generativeai as genai
import sys
import json
from typing import Dict, Any
import argparse
import os
from utils.CatAttributesToPrediction import predict_breed


def setup_llm() -> Any:
    """Configure and return the LLM model."""
    genai.configure(api_key="AIzaSyDEBKa-klOmaLZyXXUA-H1VnVvteEymFDk")
    return genai.GenerativeModel("gemini-1.5-flash")


def extract_attributes(description: str, model: Any) -> Dict[str, Any]:
    """Use LLM to extract cat attributes from description."""
    prompt = f"""
    Extract attributes from this cat description: "{description}"
    Return ONLY a JSON object (no additional text, do not use code block markers) with these exact fields (use null if not mentioned):
    {{
      "Age": integer between 0-12 or null,
      "Number_of_cats": integer or null,
      "Time_spent_outside": integer between 0-5 or null,
      "Time_with_owner": integer between 0-5 or null,
      "Shy": integer between 0-5 or null,
      "Calm": integer between 0-5 or null,
      "Skittish": integer between 0-5 or null,
      "Intelligent": integer between 0-5 or null,
      "Vigilant": integer between 0-5 or null,
      "Tenacious": integer between 0-5 or null,
      "Affectionate": integer between 0-5 or null,
      "Friendly": integer between 0-5 or null,
      "Loner": integer between 0-5 or null,
      "Ferocious": integer between 0-5 or null,
      "Territorial": integer between 0-5 or null,
      "Aggressive": integer between 0-5 or null,
      "Impulsive": integer between 0-5 or null,
      "Predictable": integer between 0-5 or null,
      "Inattentive": integer between 0-5 or null,
      "Abundance_of_natural_areas": integer between 0-3 or null,
      "Bird_captures": integer between 0-5 or null,
      "Small_mammal_captures": integer between 0-5 or null,
      "Housing_type": one of ["Apartment with balcony", "Apartment without balcony", "House in subdivision", "Individual house"] or null,
      "Zone": one of ["Urban", "Periurban", "Rural"] or null,
      "Gender": "M" or "F" or null,
      "Coat_length": one of ["Long hair", "Medium hair", "No hair", "Short hair"] or null,
      "Coat_pattern": one of ["Bicolor", "Colorpoint", "Solid", "Tabby", "Tortoiseshell", "Tricolor"] or null,
      "Flat_face": boolean or null
    }}
    """

    response = model.generate_content(prompt)

    try:
        # Remove markdown code block if present
        response_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {str(e)}")
        return {}


def map_categorical_values(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Map categorical values to one-hot encoded features."""
    mapped_features = {}

    # Housing type mapping
    if extracted.get('Housing_type'):
        housing_types = {
            'Type of housing_Apartment with balcony or terrace': False,
            'Type of housing_Apartment without balcony': False,
            'Type of housing_House in a subdivision': False,
            'Type of housing_Individual house zone': False
        }
        housing_map = {
            'Apartment with balcony': 'Type of housing_Apartment with balcony or terrace',
            'Apartment without balcony': 'Type of housing_Apartment without balcony',
            'House in subdivision': 'Type of housing_House in a subdivision',
            'Individual house': 'Type of housing_Individual house zone'
        }
        if extracted['Housing_type'] in housing_map:
            housing_types[housing_map[extracted['Housing_type']]] = True
        mapped_features.update(housing_types)

    # Zone mapping
    if extracted.get('Zone'):
        zones = {
            'Zone_Urban': False,
            'Zone_Periurban': False,
            'Zone_Rural': False
        }
        if extracted['Zone'] in ['Urban', 'Periurban', 'Rural']:
            zones[f'Zone_{extracted["Zone"]}'] = True
        mapped_features.update(zones)

    # Gender mapping
    if extracted.get('Gender'):
        mapped_features['Gender_F'] = extracted['Gender'] == 'F'
        mapped_features['Gender_M'] = extracted['Gender'] == 'M'

    # Coat length mapping
    if extracted.get('Coat_length'):
        coat_lengths = {
            'Coat Length_Long hair': False,
            'Coat Length_Medium hair': False,
            'Coat Length_No hair': False,
            'Coat Length_Short hair': False
        }
        coat_map = {
            'Long hair': 'Coat Length_Long hair',
            'Medium hair': 'Coat Length_Medium hair',
            'No hair': 'Coat Length_No hair',
            'Short hair': 'Coat Length_Short hair'
        }
        if extracted['Coat_length'] in coat_map:
            coat_lengths[coat_map[extracted['Coat_length']]] = True
        mapped_features.update(coat_lengths)

    # Coat pattern mapping
    if extracted.get('Coat_pattern'):
        coat_patterns = {
            'Coat Pattern_Bicolor': False,
            'Coat Pattern_Colorpoint': False,
            'Coat Pattern_Solid': False,
            'Coat Pattern_Tabby': False,
            'Coat Pattern_Tortoiseshell': False,
            'Coat Pattern_Tricolor': False
        }
        if extracted['Coat_pattern']:
            pattern_key = f'Coat Pattern_{extracted["Coat_pattern"]}'
            if pattern_key in coat_patterns:
                coat_patterns[pattern_key] = True
        mapped_features.update(coat_patterns)

    # Direct mappings
    direct_mappings = {
        'Age': 'Age',
        'Number_of_cats': 'Number of cats in the household',
        'Time_spent_outside': 'Time spent outside each day',
        'Time_with_owner': 'Time spent with the owner each day',
        'Abundance_of_natural_areas': 'The abundance of natural areas',
        'Bird_captures': 'Frequency of Bird Captures',
        'Small_mammal_captures': 'Frequency of Small Mammal Captures',
        'Flat_face': 'FlatFace'
    }

    for source, target in direct_mappings.items():
        if source in extracted and extracted[source] is not None:
            mapped_features[target] = extracted[source]

    # Personality traits (already 0-5 scale)
    personality_traits = [
        'Shy', 'Calm', 'Skittish', 'Intelligent', 'Vigilant', 'Tenacious',
        'Affectionate', 'Friendly', 'Loner', 'Ferocious', 'Territorial',
        'Aggressive', 'Impulsive', 'Predictable', 'Inattentive'
    ]

    for trait in personality_traits:
        if trait in extracted and extracted[trait] is not None:
            mapped_features[trait] = extracted[trait]

    return mapped_features


def generate_missing_attributes(probabilities: Dict[str, Any],
                                extracted_attrs: Dict[str, Any]) -> np.ndarray:
    """Generate missing attributes based on probability distributions."""
    features = []

    # Get mapped features from extracted attributes
    mapped_features = map_categorical_values(extracted_attrs)

    # Define value ranges for numeric attributes
    numeric_ranges = {
        'Age': (0, 12),
        'Time spent outside each day': (0, 5),
        'Time spent with the owner each day': (0, 5),
        'The abundance of natural areas': (0, 3),
        'Frequency of Bird Captures': (0, 5),
        'Frequency of Small Mammal Captures': (0, 5)
    }

    # All personality traits are 0-5
    personality_traits = [
        'Shy', 'Calm', 'Skittish', 'Intelligent', 'Vigilant', 'Tenacious',
        'Affectionate', 'Friendly', 'Loner', 'Ferocious', 'Territorial',
        'Aggressive', 'Impulsive', 'Predictable', 'Inattentive'
    ]

    for trait in personality_traits:
        numeric_ranges[trait] = (0, 5)

    # Define the order of features
    feature_order = [
        'Age', 'Number of cats in the household', 'Time spent outside each day',
        'Time spent with the owner each day', 'Shy', 'Calm', 'Skittish',
        'Intelligent', 'Vigilant', 'Tenacious', 'Affectionate', 'Friendly',
        'Loner', 'Ferocious', 'Territorial', 'Aggressive', 'Impulsive',
        'Predictable', 'Inattentive', 'The abundance of natural areas',
        'Frequency of Bird Captures', 'Frequency of Small Mammal Captures',
        'Type of housing_Apartment with balcony or terrace',
        'Type of housing_Apartment without balcony',
        'Type of housing_House in a subdivision',
        'Type of housing_Individual house zone', 'Zone_Periurban',
        'Zone_Rural', 'Zone_Urban', 'Gender_F', 'Gender_M',
        'Coat Length_Long hair', 'Coat Length_Medium hair',
        'Coat Length_No hair', 'Coat Length_Short hair',
        'Coat Pattern_Bicolor', 'Coat Pattern_Colorpoint',
        'Coat Pattern_Solid', 'Coat Pattern_Tabby',
        'Coat Pattern_Tortoiseshell', 'Coat Pattern_Tricolor',
        'FlatFace'
    ]

    for feature in feature_order:
        if feature in mapped_features:
            features.append(mapped_features[feature])
        else:
            if feature in numeric_ranges:
                # For numeric features, use the probability distribution if available
                if feature in probabilities and isinstance(probabilities[feature], dict):
                    mean = probabilities[feature]['mean']
                    std = probabilities[feature]['std']
                    value = int(np.clip(np.random.normal(mean, std),
                                      numeric_ranges[feature][0],
                                      numeric_ranges[feature][1]))
                else:
                    # Fallback to uniform distribution within range
                    min_val, max_val = numeric_ranges[feature]
                    value = int(np.random.uniform(min_val, max_val))
                features.append(value)
            else:
                # For boolean features
                if feature in probabilities and isinstance(probabilities[feature], (int, float)):
                    prob = probabilities[feature]
                else:
                    # Default probability of 0.5 if not available
                    prob = 0.5
                features.append(np.random.binomial(1, prob))

    return np.array(features)


def calculate_probabilities(dataset: pd.DataFrame) -> Dict[str, Any]:
    """Calculate probability distributions for dataset attributes."""
    probabilities = {}

    # For boolean columns (one-hot encoded features)
    bool_columns = dataset.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        probabilities[col] = float(dataset[col].mean())

    # For numeric columns
    numeric_columns = [
        'Age', 'Number of cats in the household', 'Time spent outside each day',
        'Time spent with the owner each day', 'Shy', 'Calm', 'Skittish',
        'Intelligent', 'Vigilant', 'Tenacious', 'Affectionate', 'Friendly',
        'Loner', 'Ferocious', 'Territorial', 'Aggressive', 'Impulsive',
        'Predictable', 'Inattentive', 'The abundance of natural areas',
        'Frequency of Bird Captures', 'Frequency of Small Mammal Captures'
    ]

    for col in numeric_columns:
        if col in dataset.columns:
            probabilities[col] = {
                'mean': float(dataset[col].mean()),
                'std': float(dataset[col].std())
            }

    return probabilities


def main():
    parser = argparse.ArgumentParser(description='Predict cat breed from description')
    parser.add_argument('description', type=str, help='Description of the cat')
    args = parser.parse_args()

    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'utils', 'Dataset_Preprocessed.xlsx')
    dataset = pd.read_excel(dataset_path)

    # Setup LLM and get predictions
    model = setup_llm()
    extracted_attrs = extract_attributes(args.description, model)

    # Calculate probability distributions
    probabilities = calculate_probabilities(dataset)

    # Generate feature vector with missing attributes filled in
    features = generate_missing_attributes(probabilities, extracted_attrs)

    # Predict breed
    predicted_breed, confidence = predict_breed(features)

    # Only output the prediction
    print(predicted_breed)


if __name__ == "__main__":
    main()