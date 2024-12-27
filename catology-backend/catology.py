from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
from pathlib import Path

app = Flask(__name__)
CORS(app)

def run_cat_script(script_name: str, input_text: str) -> str:
    """
    Run a cat-related script and return its output.
    """
    try:
        # Use absolute path based on the script location
        script_dir = Path(__file__).parent
        script_path = script_dir / f'{script_name}.py'
        
        result = subprocess.run(
            [sys.executable, str(script_path), input_text],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return f"Error: Failed to run {script_name}"
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return "Error: Unexpected error occurred"

def process_cat_query(query: str) -> str:
    """
    Process a cat-related query by first classifying it and then running appropriate script.
    Matches the implementation in usage_examples.ipynb.
    """
    # First, classify the query
    query_type = run_cat_script('QueryClassifier', query)
    
    # Based on classification, run appropriate script
    if query_type == "Description":
        return run_cat_script('BreedDescription', query)
    elif query_type == "Comparison":
        return run_cat_script('BreedComparison', query)
    elif query_type == "Prediction":
        return run_cat_script('CatDescriptionToBreed', query)
    else:
        return query_type  # Returns the help message

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    try:
        # Process examples like in the notebook:
        # "Tell me about Sphynx cats" -> Description
        # "Compare Sphynx and Persian cats" -> Comparison
        # "My cat is hairless, very intelligent, affectionate and shy" -> Prediction
        response = process_cat_query(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)