"""Model prediction utilities"""

import joblib
from pathlib import Path


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Trained model object
    """
    model = joblib.load(model_path)
    return model


def save_model(model, model_path):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_path: Path where to save the model
    """
    joblib.dump(model, model_path)


def make_predictions(model, X):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Feature data
        
    Returns:
        Predictions
    """
    return model.predict(X)
