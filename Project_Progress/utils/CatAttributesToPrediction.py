from pathlib import Path
import numpy as np
import pickle
import sys
from os import path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from breed_predictor_model.NeuralNetworkImplementation import NeuralNetwork

# Get the absolute path to the model file
MODEL_PATH = project_root / 'breed_predictor_model' / 'cat_breed_model.pkl'

def load_model(filepath=MODEL_PATH):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    input_dim = model_data['weights'][0]['weights'].shape[0]
    output_dim = model_data['weights'][-1]['weights'].shape[1]

    model = NeuralNetwork(input_dim, output_dim)
    model.set_weights(model_data['weights'])

    return model, model_data['scaler']

def predict_breed(input_data):
    model, scaler = load_model()

    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    scaled_input = scaler.transform(input_data)
    predictions = model.predict(scaled_input)
    predicted_class = np.argmax(predictions, axis=1)[0]

    race_columns = ['Bengal', 'Birman', 'British Shorthair', 'Chartreux', 'European',
                   'Maine coon', 'No breed', 'Other', 'Persian', 'Ragdoll',
                   'Savannah', 'Siamese', 'Sphynx', 'Turkish angora']

    confidence_score = float(predictions[0][predicted_class])
    return race_columns[predicted_class], confidence_score