import pandas as pd
import google.generativeai as genai
import argparse
import os
from typing import Dict, Any


def setup_llm() -> Any:
    """Configure and return the LLM model."""
    genai.configure(api_key="AIzaSyDEBKa-klOmaLZyXXUA-H1VnVvteEymFDk")
    return genai.GenerativeModel("gemini-1.5-flash")


def extract_breed_name(query: str) -> str:
    """Extract breed name from the query using LLM."""
    model = setup_llm()
    prompt = f"""
    Extract just the cat breed name from this query: "{query}"
    Return ONLY the breed name, nothing else. If no specific breed is mentioned, return "unknown". 
    Example: "Please describe the cat breed Sphynx to me" -> "Sphynx"
    """
    response = model.generate_content(prompt)
    return response.text.strip()


def normalize_breed_name(breed: str, dataset: pd.DataFrame) -> str:
    """Normalize breed name to match dataset column names."""
    # Get all column names that start with 'Race_'
    race_columns = [col for col in dataset.columns if col.startswith('Race_')]
    
    # Remove 'Race_' prefix and create a mapping of normalized names to actual column names
    breed_mapping = {col.replace('Race_', '').lower(): col.replace('Race_', '') 
                    for col in race_columns}
    
    # Normalize the input breed name
    normalized_breed = breed.lower()
    
    # Return the matching breed name from our dataset, or the original if not found
    return breed_mapping.get(normalized_breed, breed)


def get_breed_averages(dataset: pd.DataFrame, breed: str) -> Dict[str, Any]:
    """Calculate average values for all attributes for a specific breed."""
    # Normalize the breed name to match dataset columns
    normalized_breed = normalize_breed_name(breed, dataset)
    
    # Filter dataset for the specific breed
    breed_data = dataset[dataset[f'Race_{normalized_breed}'] == True]

    if len(breed_data) == 0:
        raise ValueError(f"No data found for breed: {normalized_breed}")

    # Initialize averages dictionary
    averages = {}

    # Calculate numeric averages
    numeric_columns = [
        'Age', 'Number of cats in the household',
        'Time spent outside each day', 'Time spent with the owner each day',
        'Shy', 'Calm', 'Skittish', 'Intelligent', 'Vigilant', 'Tenacious',
        'Affectionate', 'Friendly', 'Loner', 'Ferocious', 'Territorial',
        'Aggressive', 'Impulsive', 'Predictable', 'Inattentive',
        'The abundance of natural areas', 'Frequency of Bird Captures',
        'Frequency of Small Mammal Captures'
    ]

    for col in numeric_columns:
        averages[col] = round(breed_data[col].mean(), 1)

    # Calculate most common values for categorical columns
    # Housing type
    housing_cols = [col for col in dataset.columns if col.startswith('Type of housing_')]
    housing_values = breed_data[housing_cols].idxmax(axis=1).mode()[0]
    averages['Housing_type'] = housing_values.replace('Type of housing_', '')

    # Zone
    zone_cols = [col for col in dataset.columns if col.startswith('Zone_')]
    zone_values = breed_data[zone_cols].idxmax(axis=1).mode()[0]
    averages['Zone'] = zone_values.replace('Zone_', '')

    # Gender
    averages['Gender'] = 'F' if breed_data['Gender_F'].mean() > breed_data['Gender_M'].mean() else 'M'

    # Coat length
    coat_length_cols = [col for col in dataset.columns if col.startswith('Coat Length_')]
    coat_length_values = breed_data[coat_length_cols].idxmax(axis=1).mode()[0]
    averages['Coat_length'] = coat_length_values.replace('Coat Length_', '')

    # Coat pattern
    coat_pattern_cols = [col for col in dataset.columns if col.startswith('Coat Pattern_')]
    coat_pattern_values = breed_data[coat_pattern_cols].idxmax(axis=1).mode()[0]
    averages['Coat_pattern'] = coat_pattern_values.replace('Coat Pattern_', '')

    # Flat face
    averages['Flat_face'] = bool(round(breed_data['FlatFace'].mean()))

    return averages


def generate_description(breed: str, attributes: Dict[str, Any]) -> str:
    """Generate a natural language description of the cat using LLM."""
    model = setup_llm()

    # Format attributes for the prompt
    attrs_str = "\n".join([f"{k}: {v}" for k, v in attributes.items()])

    prompt = f"""
    Given these average attributes for the {breed} cat breed:

    {attrs_str}

    Generate a natural, flowing description of this cat breed. The description should feel like it's written by a cat expert. 
    Include personality traits, physical characteristics, and living habits based on the data.
    For numeric traits (0-5 scale), 0 means not at all, and 5 means extremely.
    Describe all major characteristics shown in the data but make it sound natural, not like reading statistics.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(description='Generate cat breed description')
    parser.add_argument('query', type=str, help='Query asking about a cat breed')
    args = parser.parse_args()

    try:
        # Load dataset first (we need it for breed name normalization)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'utils', 'Dataset_Preprocessed.xlsx')
        dataset = pd.read_excel(dataset_path)

        # Extract breed name
        breed = extract_breed_name(args.query)

        # Get breed averages
        averages = get_breed_averages(dataset, breed)

        # Generate description
        description = generate_description(breed, averages)

        # Print the result
        print(description)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()