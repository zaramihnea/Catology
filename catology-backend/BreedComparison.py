# BreedComparison.py
import pandas as pd
import google.generativeai as genai
import argparse
import os
from typing import Dict, Any, Tuple


def setup_llm() -> Any:
    """Configure and return the LLM model."""
    genai.configure(api_key="AIzaSyDEBKa-klOmaLZyXXUA-H1VnVvteEymFDk")
    return genai.GenerativeModel("gemini-1.5-flash")


def extract_breed_names(query: str) -> Tuple[str, str]:
    """Extract two breed names from the query using LLM."""
    model = setup_llm()
    prompt = f"""
    Extract exactly two cat breed names from this query: "{query}"
    Return ONLY the two breed names separated by a comma, nothing else.
    Example: "Compare Sphynx and Persian cats" -> "Sphynx,Persian"
    Example: "What's the difference between Maine Coon and Ragdoll?" -> "Maine coon,Ragdoll"
    """
    response = model.generate_content(prompt)
    breeds = response.text.strip().split(',')
    if len(breeds) != 2:
        raise ValueError("Could not identify exactly two breeds in the query")
    return (breeds[0].strip(), breeds[1].strip())


def get_breed_attributes(dataset: pd.DataFrame, breed: str) -> Dict[str, Any]:
    """Calculate average values for all attributes for a specific breed."""
    breed_data = dataset[dataset[f'Race_{breed}'] == True]

    if len(breed_data) == 0:
        raise ValueError(f"No data found for breed: {breed}")

    attributes = {}

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
        attributes[col] = round(breed_data[col].mean(), 1)

    # Calculate most common values for categorical columns
    housing_cols = [col for col in dataset.columns if col.startswith('Type of housing_')]
    attributes['Housing_type'] = breed_data[housing_cols].idxmax(axis=1).mode()[0].replace('Type of housing_', '')

    zone_cols = [col for col in dataset.columns if col.startswith('Zone_')]
    attributes['Zone'] = breed_data[zone_cols].idxmax(axis=1).mode()[0].replace('Zone_', '')

    attributes['Gender'] = 'F' if breed_data['Gender_F'].mean() > breed_data['Gender_M'].mean() else 'M'

    coat_length_cols = [col for col in dataset.columns if col.startswith('Coat Length_')]
    attributes['Coat_length'] = breed_data[coat_length_cols].idxmax(axis=1).mode()[0].replace('Coat Length_', '')

    coat_pattern_cols = [col for col in dataset.columns if col.startswith('Coat Pattern_')]
    attributes['Coat_pattern'] = breed_data[coat_pattern_cols].idxmax(axis=1).mode()[0].replace('Coat Pattern_', '')

    attributes['Flat_face'] = bool(round(breed_data['FlatFace'].mean()))

    return attributes


def generate_comparison(breed1: str, breed2: str, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> str:
    """Generate a natural language comparison between two breeds using LLM."""
    model = setup_llm()

    # Format attributes for the prompt
    attrs1_str = "\n".join([f"{k}: {v}" for k, v in attrs1.items()])
    attrs2_str = "\n".join([f"{k}: {v}" for k, v in attrs2.items()])

    prompt = f"""
    Compare these two cat breeds based on their average attributes:

    {breed1}:
    {attrs1_str}

    {breed2}:
    {attrs2_str}

    Generate a detailed comparison focusing on:
    1. Key personality differences (using the 0-5 scale traits)
    2. Physical characteristics
    3. Living habits and preferences
    4. Care requirements based on their traits

    Make the comparison natural and flowing, like an expert explaining the differences.
    Highlight significant differences but also note important similarities.
    For numeric traits (0-5 scale), 0 means not at all, and 5 means extremely.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    parser = argparse.ArgumentParser(description='Compare two cat breeds')
    parser.add_argument('query', type=str, help='Query comparing two cat breeds')
    args = parser.parse_args()

    try:
        # Extract breed names
        breed1, breed2 = extract_breed_names(args.query)

        # Load dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'utils', 'Dataset_Preprocessed.xlsx')
        dataset = pd.read_excel(dataset_path)

        # Get attributes for both breeds
        attrs1 = get_breed_attributes(dataset, breed1)
        attrs2 = get_breed_attributes(dataset, breed2)

        # Generate comparison
        comparison = generate_comparison(breed1, breed2, attrs1, attrs2)

        # Print the result
        print(comparison)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()