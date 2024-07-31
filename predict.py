import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from utils.smiles_to_descriptors import smiles_to_descriptors
from utils.data_preprocessing import preprocess_descriptors

class EnhancedDNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
        super(EnhancedDNN, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



def load_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file]
    return smiles_list

def predict_bbb(input_descriptors: pd.DataFrame, model_path: str) -> np.ndarray:
    """Function to predict BBB permeability for a set of samples with descriptors.

    Args:
        input_descriptors (pd.DataFrame): Descriptors for compounds
        model_path (str): Path to the trained model .pth file

    Returns:
        np.ndarray: BBB class (1 permeable; 0 non-permeable)
    """

    input_size = input_descriptors.shape[1]
    output_size = 2
    hidden_layers = [100, 200, 300, 500, 800]  # Should match the architecture of the trained model
    dropout_rate = 0.2  # Should match the trained model's dropout rate

    model = EnhancedDNN(input_size, hidden_layers, output_size, dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = StandardScaler()
    scaled_descriptors = scaler.fit_transform(input_descriptors.to_numpy())

    inputs = torch.tensor(scaled_descriptors, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

    return predictions.numpy()

def run_prediction(smiles_file: str, model_path: str) -> None:
    """Function to calculate descriptors and generate predictions of BBB permeability.

    Args:
        smiles_file (str): Path to the .smi file containing SMILES strings
        model_path (str): Path to the trained model .pth file
    """
  
    smiles_list = load_smiles(smiles_file)
    descriptors = smiles_to_descriptors(smiles_list)
    preprocessed_descriptors = preprocess_descriptors(descriptors)
    predictions = predict_bbb(preprocessed_descriptors, model_path)

    results = pd.DataFrame({'SMILES': smiles_list, 'Predicted_class': predictions})
    results['Predicted_class'] = results['Predicted_class'].map({1: 'Pass', 0: 'Fail'})

    results.to_csv('BBB_predictions.csv', index=False)
    print("Predictions saved to BBB_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict BBB permeability from SMILES strings.')
    parser.add_argument('smiles_file', type=str, help='Path to the .smi file containing SMILES strings.')
    parser.add_argument('model_path', type=str, help='Path to the trained model .pth file.')
    args = parser.parse_args()

    run_prediction(args.smiles_file, args.model_path)

