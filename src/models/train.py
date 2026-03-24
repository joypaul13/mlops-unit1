"""Model training and evaluation"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Linear Regression model.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test data
        random_state: Random seed
        
    Returns:
        dict: Contains trained model and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluate_model(y_test, y_pred)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': metrics
    }


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
